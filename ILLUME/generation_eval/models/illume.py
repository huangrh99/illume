import torch
import os
import re
import numpy as np
from transformers import LogitsProcessorList, set_seed
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any

from .builder import EVAL_MODELS
from .inference_utils import CFGLogits, DualVQImageTokenProcessor, DynamicSamplingProcessor, InterleavedLogitsProcessor

from illume.constants import IMAGE_TOKEN_INDEX
from illume.conversation import conv_templates
from illume.data.aspect_ratio_utils import RATIOS, AspectRatioCrop
from illume.data.data_utils import unpad_and_resize_back
from illume.model.builder import load_pretrained_model
from illume.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token

from utils.registry_utils import read_config

from tokenizer.builder import build_vq_model
from tokenizer.dualvitok_model import RESOLUTION_MAPPING


@dataclass
class InferenceConfig:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    llm_cfg_scale: float = 1.0
    diffusion_cfg_scale: float = 1.5
    diffusion_num_inference_steps: float = 50
    # Image semantic token generation parameters
    image_semantic_temperature: float = None  # Will default to temperature if None
    image_semantic_top_k: int = None  # Will default to top_k if None
    image_semantic_top_p: float = None  # Will default to top_p if None
    # Image pixel token generation parameters
    image_pixel_temperature: Optional[float] = None  # Will default to image_semantic_temperature if None
    image_pixel_top_k: Optional[int] = None  # Will default to image_semantic_top_k * 3 if None
    image_pixel_top_p: Optional[float] = None  # Will default to image_semantic_top_p if None
    # Other parameters
    dataset_name: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None
    unconditional_prompt: Optional[Any] = None
    max_new_tokens: Optional[int] = None  # Added max_new_tokens

    def __post_init__(self):
        if self.image_semantic_temperature is None:
            self.image_semantic_temperature = self.temperature
        if self.image_semantic_top_k is None:
            self.image_semantic_top_k = self.top_k
        if self.image_semantic_top_p is None:
            self.image_semantic_top_p = self.top_p

        if self.image_pixel_temperature is None:
            self.image_pixel_temperature = self.image_semantic_temperature
        if self.image_pixel_top_k is None:
            self.image_pixel_top_k = self.image_semantic_top_k * 3
        if self.image_pixel_top_p is None:
            self.image_pixel_top_p = self.image_semantic_top_p


# qwen2.5
special_tokens_ids = [151665, 151666, 151667, 151668, 151669, 151670, 151671]
start_token = 151672 + 32
level0_range = (start_token, start_token + 32768)  # Level 0 token ID 范围
level1_range = (start_token + 32768, start_token + 32768 * 4)  # Level 1 token ID 范围

special_tokens_dict = {
    "start_of_image": 151665,
    "end_of_image": 151666,
    "start_of_level0": 151668,
    "end_of_level0": 151669,
    "start_of_level1": 151670,
    "end_of_level1": 151671,
    "end_of_line": 151667,
    "end_of_text": 151645,
    #
    "level0_range": level0_range,
    "level1_range": level1_range,
}


def rank0_print(content):
    if int(os.getenv('WORLD_SIZE', '1')) > 1 and torch.distributed.get_rank() == 0:
        print(content)
    elif int(os.getenv('WORLD_SIZE', '1')) == 1:
        print(content)


def calculate_image_token_num(h, w, downsample_rate_per_level=[28, 16]):
    # Level‑0 ─ Semantic tokens
    mapped_w, mapped_h = RESOLUTION_MAPPING[(w, h)]
    w1 = mapped_w // downsample_rate_per_level[0]
    h1 = mapped_h // downsample_rate_per_level[0]
    semantic_token_num = w1 * h1

    # Level‑1 ─ Pixel tokens
    w2 = w // downsample_rate_per_level[1]
    h2 = h // downsample_rate_per_level[1]
    pixel_token_num = w2 * h2

    # rank0_print(f"semantic_token_num:{semantic_token_num}, pixel_token_num:{pixel_token_num}")
    max_token_length = (h1 * (w1 + 1) + 2) + (h2 * (w2 + 1) + 2) + 2 + 2 + 1 + 1
    return [semantic_token_num, pixel_token_num], max_token_length, h1, w1, h2, w2


def pad_sequence(tokenizer, input_ids, batch_first, padding_value):
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids


@EVAL_MODELS.register_module()
class ILLUME():
    def __init__(self,
                 config,
                 tokenizer_config,
                 diffusion_decoder_path=None,
                 tokenizer_checkpoint=None,
                 torch_dtype="fp32",
                 seed=42,
                 **kwargs):
        self.config = read_config(config)
        self.tokenizer_config = read_config(tokenizer_config)
        self.diffusion_decoder_path = diffusion_decoder_path
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.seed = seed

        self.output_dir = self.config.training_args.output_dir

        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.rank = int(os.getenv('RANK', '0'))
        self.local_rank = int(os.getenv('LOCAL_RANK', 0))
        self.device = self.local_rank

        torch_dtype_mapping = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
        self.torch_dtype = torch_dtype_mapping[torch_dtype]

        self.build_mllm_model()
        self.build_detokenizer()

        self._default_generation_template = "Generate an image of {resolution_tag}, the content of image is {content}\n"
        self._default_generation_unconditional_template = "Generate a random image of {resolution_tag}\n"

        self._default_editing_template = "{resolution_tag}<image>\nPlease edit the image according to the instruction: {content}\n"
        self._default_editing_unconditional_template = "{resolution_tag}<image>\nReconstruct the image according to the given image\n"

    @property
    def default_generation_template(self):
        return self._default_generation_template

    @property
    def default_generation_unconditional_template(self):
        return self._default_generation_unconditional_template

    @property
    def default_editing_template(self):
        return self._default_editing_template

    @property
    def default_editing_unconditional_template(self):
        return self._default_editing_unconditional_template

    def prepare_inference_config(self,
                                 temperature=1.0,
                                 top_k=2048,
                                 top_p=1.0,
                                 llm_cfg_scale=2.0,
                                 diffusion_cfg_scale=1.5,
                                 diffusion_num_inference_steps=50,
                                 image_semantic_temperature=None,
                                 image_semantic_top_k=None,
                                 image_semantic_top_p=None,
                                 image_pixel_temperature=None,
                                 image_pixel_top_k=None,
                                 image_pixel_top_p=None,
                                 resolution=None,
                                 unconditional_prompt=None,
                                 max_new_tokens=1024,
                                 ):

        return InferenceConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            llm_cfg_scale=llm_cfg_scale,
            diffusion_cfg_scale=diffusion_cfg_scale,
            diffusion_num_inference_steps=diffusion_num_inference_steps,
            image_semantic_temperature=image_semantic_temperature,
            image_semantic_top_k=image_semantic_top_k,
            image_semantic_top_p=image_semantic_top_p,
            image_pixel_temperature=image_pixel_temperature,
            image_pixel_top_k=image_pixel_top_k,
            image_pixel_top_p=image_pixel_top_p,
            resolution=resolution,
            unconditional_prompt=unconditional_prompt,
            max_new_tokens=max_new_tokens,
        )

    def build_mllm_model(self):
        rank0_print("build mllm")
        model_path = self.config.training_args.output_dir
        model_path = os.path.expanduser(model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = None
        if self.config.training_args.get('lora_enable',
                                         False) or self.config.model_args.language_model.trainable == False:
            model_base = self.config.model_args.language_model.pretrained_model_name_or_path
        rank0_print(f"model_path, {model_path}")
        rank0_print(f"model_base, {model_base}")
        rank0_print(f"model_name, {model_name}")

        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                               device_map=self.local_rank,
                                                                               device=self.local_rank,
                                                                               config=self.config)
        rank0_print(f"tokenizer_length: {len(tokenizer)}")
        model = model.eval()
        rank0_print(f'mllm model dtype: {model.dtype}')
        rank0_print(f'model device: {model.device}')

        model.config.image_aspect_ratio = self.config.data_args.image_aspect_ratio
        model.config.special_tokens_ids = self.config.model_args.get("special_tokens_ids", [])

        self.mllm_model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        rank0_print("build mllm done")

    def build_detokenizer(self):
        rank0_print("build detokenizer")
        vq_model = build_vq_model(self.tokenizer_config.vq_model)
        vq_model = vq_model.to(self.torch_dtype).to(self.device)
        vq_model.eval()  # important
        if self.tokenizer_checkpoint is not None:
            checkpoint = torch.load(self.tokenizer_checkpoint, map_location="cpu")
            if "ema" in checkpoint and self.tokenizer_config.get("use_ema", False):  # ema
                model_weight = checkpoint["ema"]
                rank0_print("Using ema params for evaluation.")
            elif "model" in checkpoint:  # ddp
                model_weight = checkpoint["model"]
            elif "state_dict" in checkpoint:
                model_weight = checkpoint["state_dict"]
            else:
                model_weight = checkpoint

            msg = vq_model.load_state_dict(model_weight, strict=False)
            rank0_print(msg)
            del checkpoint
        self.vq_model = vq_model

        rank0_print(f"use diffusion decoder")
        from tokenizer.sdxl_decoder_pipe import StableDiffusionXLDecoderPipeline
        self.diffusion_decoder_pipe = StableDiffusionXLDecoderPipeline.from_pretrained(
            self.diffusion_decoder_path,
            add_watermarker=False,
            vq_config=self.tokenizer_config,
            vq_model=vq_model,
            torch_dtype=torch.float32,
        ).to(self.device)

        rank0_print("build detokenizer done")

    def get_resolution_tag_from_resolution(self, resolution):
        return f"<height_{resolution[0]}><width_{resolution[1]}>"

    def transform_image_nearest_resolution_ratio(self, image, ratios=RATIOS):
        arc = AspectRatioCrop(ratios, crop_percent_thresh=self.crop_percent_thresh)
        image, original_size, target_size, flag_matched = arc(image, is_inference=True)
        return image

    def unpad_and_resize_back(self, padded_image, original_width, original_height):
        return unpad_and_resize_back(padded_image, original_width, original_height)

    def prepare_mllm_batch_data(self, batch, is_img_gen_task=True):
        # add system template
        prompts = []
        for b in batch:
            qs = b["prompt"]

            if "images_data" in b and '<image>' not in qs:
                qs = f"<image>\n{qs}"

            conv = conv_templates[self.config.model_args.version].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)

        output = {'prompts': prompts}

        if "images_data" in batch[0]:
            images = []
            for b in batch:
                images.extend(b["images_data"])
            original_sizes = [image.size for image in images]
            # images = [b["images_data"] for b in batch]
            images, image_sizes = process_images(images,
                                                 self.image_processor,
                                                 self.mllm_model.config,
                                                 self.config,
                                                 is_gen_task=is_img_gen_task)
            output["images"] = images
            output["image_sizes"] = image_sizes
            output["original_sizes"] = original_sizes

        for k in batch[0]:
            if k in ['images_data']:
                continue
            tmp = [_[k] for _ in batch]
            output[k] = tmp

        return output

    def prepare_logit_processor(self, inference_config: InferenceConfig, unconditional_input_ids, images, image_sizes):
        cfg_logitprocessor = CFGLogits(
            inference_config.llm_cfg_scale,
            unconditional_input_ids.to(self.device),
            self.mllm_model,
            images=images,
            image_sizes=image_sizes
        )

        strict_image_token_logitprocessor = DualVQImageTokenProcessor(
            level0_range=level0_range,
            level1_range=level1_range,
            num_level0_rows=self.h1,
            num_level0_tokens=self.w1,
            num_level1_rows=self.h2,
            num_level1_tokens=self.w2,
            special_tokens=special_tokens_dict
        )

        dynamicSampling_logitprocessor = DynamicSamplingProcessor(
            special_tokens=special_tokens_dict,
            default_temp=inference_config.temperature,
            level0_temp=inference_config.image_semantic_temperature,
            level1_temp=inference_config.image_pixel_temperature,
            default_top_k=inference_config.top_k,
            level0_top_k=inference_config.image_semantic_top_k,
            level1_top_k=inference_config.image_pixel_top_k,
            default_top_p=inference_config.top_p,
            level0_top_p=inference_config.image_semantic_top_p,
            level1_top_p=inference_config.image_pixel_top_p,
        )

        logits_processor = [
            cfg_logitprocessor,
            strict_image_token_logitprocessor,
            dynamicSampling_logitprocessor,
        ]

        return logits_processor

    def prepare_interleaved_logit_processor(self,
                                            inference_config: InferenceConfig,
                                            unconditional_input_ids=None,
                                            images=None,
                                            image_sizes=None
                                            ):
        if inference_config.resolution is not None:
            _, _, h1, w1, h2, w2 = calculate_image_token_num(*inference_config.resolution)
        else:
            h1, w1, h2, w2 = 0, 0, 0, 0

        return [InterleavedLogitsProcessor(
            guidance_scale=inference_config.llm_cfg_scale,
            uncond=unconditional_input_ids,
            model=self.mllm_model,
            level0_range=level0_range,
            level1_range=level1_range,
            num_level0_rows=h1, num_level0_tokens=w1,
            num_level1_rows=h2, num_level1_tokens=w2,
            special_tokens=special_tokens_dict,
            default_temp=inference_config.temperature, level0_temp=inference_config.image_semantic_temperature,
            level1_temp=inference_config.image_pixel_temperature,
            default_top_k=inference_config.top_k, level0_top_k=inference_config.image_semantic_top_k,
            level1_top_k=inference_config.image_pixel_top_k,
            default_top_p=inference_config.top_p, level0_top_p=inference_config.image_semantic_top_p,
            level1_top_p=inference_config.image_pixel_top_p,
            images=images,
            image_sizes=image_sizes
        )]

    def prepare_conversation_prompt(self, prompt):
        assert isinstance(prompt, str)
        conv = conv_templates[self.config.model_args.version].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def inference_mllm(self, batch_data, inference_config, is_img_gen_task=True, **kwargs):
        batch_data = self.prepare_mllm_batch_data(batch_data, is_img_gen_task=is_img_gen_task)
        prompts = batch_data["prompts"]

        if inference_config.resolution is not None:
            h, w = inference_config.resolution
            if h < 0 or w < 0:  # editing setting, for anyres, only support bs=1
                image_sizes = batch_data["image_sizes"]
                w, h = image_sizes[0]
        else:
            image_sizes = batch_data["image_sizes"]
            w, h = image_sizes[0]

        if is_img_gen_task:
            inference_config.resolution = (h, w)

            self.token_nums, max_new_tokens, self.h1, self.w1, self.h2, self.w2 = calculate_image_token_num(h, w)
            inference_config.max_new_tokens = max_new_tokens
            self.resolution_tag = f"<height_{h}><width_{w}>"

            prompts = [prompt.replace("<image>", self.resolution_tag + "<image>")
                       if self.resolution_tag + "<image>" not in prompt else prompt
                       for prompt in prompts]
            prompts = [prompt.format(resolution_tag=self.resolution_tag) for prompt in prompts]
            prompts = [prompt + self.resolution_tag for prompt in prompts]  # append <height_i><width_j> in prompt

        if "images" in batch_data:
            images = batch_data["images"]
            if not isinstance(images, list):
                images = images.to(torch.float32).to(self.device)  # dtype
            image_sizes = batch_data["image_sizes"]
        else:
            images = None
            image_sizes = None

        input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for
                          prompt in prompts]

        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = pad_sequence(self.tokenizer, input_ids_list, batch_first=True, padding_value=pad_token_ids).to(
            self.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.device)

        # prepare unconditional_prompt
        if inference_config.unconditional_prompt:
            unconditional_prompt = self.prepare_conversation_prompt(inference_config.unconditional_prompt)
            if is_img_gen_task:
                if self.resolution_tag + "<image>" not in unconditional_prompt:
                    unconditional_prompt = unconditional_prompt.replace("<image>", self.resolution_tag + "<image>")
                unconditional_prompt = unconditional_prompt.format(
                    resolution_tag=self.resolution_tag) + self.resolution_tag
            unconditional_token_ids = tokenizer_image_token(unconditional_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                            return_tensors="pt")
            unconditional_token_ids = unconditional_token_ids.repeat(input_ids.shape[0], 1)
        else:
            unconditional_token_ids = None

        # prepare logits processor
        # logit_processor = self.prepare_logit_processor(inference_config, unconditional_token_ids, images, image_sizes)
        logit_processor = self.prepare_interleaved_logit_processor(inference_config, unconditional_token_ids,
                                                                   images, image_sizes,)

        set_seed(self.seed, deterministic=False)
        # do_sample = True if inference_config.temperature > 0 else False
        if 'do_sample' not in kwargs:
            kwargs['do_sample'] = True
        output_ids = self.mllm_model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=images,
            image_sizes=image_sizes,
            pad_token_id=pad_token_ids,
            # do_sample=do_sample,
            temperature=inference_config.temperature,
            top_k=inference_config.top_k,
            top_p=inference_config.top_p,
            max_new_tokens=inference_config.max_new_tokens,
            logits_processor=LogitsProcessorList(logit_processor),
            use_cache=True,
            **kwargs
        )

        text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # parse vision token from llm output
        batch_outputs = []
        for i, output in enumerate(text_outputs):
            image_embed_inds = []
            for level in range(self.config.model_args.vision_tokenizer_levels):
                pattern = r'<\|image_level{}_(\d+)\|>'.format(level)
                matches = re.findall(pattern, output)
                image_embed_ind = [int(num) for num in matches]
                # rank0_print(f"level {level}, {len(image_embed_ind)}")
                image_embed_inds.append(image_embed_ind)

                output = re.sub(pattern, '', output)

            tmp = {
                "image_embed_inds": image_embed_inds,
                "output_text": output
            }

            for k, v in batch_data.items():
                if k in ["image", "images", "prompts", "prompt"]:
                    continue
                tmp[k] = v[i]
            batch_outputs.append(tmp)
        return batch_outputs

    @torch.inference_mode()
    def inference_tokenizer_decoder(self, batch_llm_output,
                                    inference_config: InferenceConfig,
                                    use_diffusion_decoder=False):
        batch_decode_images = []
        for one_output in batch_llm_output:
            image_embed_inds = one_output["image_embed_inds"]
            semantic_code = torch.as_tensor([image_embed_inds[0]])
            texture_code = torch.as_tensor([image_embed_inds[1]])
            semantic_code = semantic_code.view(semantic_code.shape[0], self.h1, self.w1)
            texture_code = texture_code.view(texture_code.shape[0], self.h2, self.w2)

            if use_diffusion_decoder:
                h, w = inference_config.resolution
                diffusion_outputs = self.diffusion_decoder_pipe(
                    vq_indices=(semantic_code, texture_code),
                    height=h * 2,
                    width=w * 2,
                    guidance_scale=inference_config.diffusion_cfg_scale,
                    num_inference_steps=inference_config.diffusion_num_inference_steps,
                    generator=torch.Generator(self.device).manual_seed(self.seed),
                )
                samples = diffusion_outputs.images
                samples = [np.asarray(sample) for sample in samples]
            else:
                quant_semantic = self.vq_model.semantic_quantizer.indices_to_codes(semantic_code)
                quant_pixel = self.vq_model.pixel_quantizer.indices_to_codes(texture_code)
                samples = self.vq_model.decode(quant_semantic, quant_pixel)
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu",
                                                                                              dtype=torch.uint8).numpy()

            batch_decode_images.append(samples[0])
        return batch_decode_images

    def get_one_batch_results(self, batch_data, inference_config: InferenceConfig):
        set_seed(self.seed)

        # get mllm output results
        batch_llm_output = self.inference_mllm(batch_data, inference_config)

        # get image results
        # tokenizer decoder
        out_images_tokenizer = self.inference_tokenizer_decoder(batch_llm_output, inference_config,
                                                                use_diffusion_decoder=False)
        # diffusion decoder
        out_images_diffusion = self.inference_tokenizer_decoder(batch_llm_output, inference_config,
                                                                use_diffusion_decoder=True)

        output = {
            "batch_llm_output": batch_llm_output,
            "out_images_tokenizer": out_images_tokenizer,
            "out_images_diffusion": out_images_diffusion,
        }
        return output
