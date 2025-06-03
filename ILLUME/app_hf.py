import argparse
import os
import traceback
import logging
from functools import partial
from threading import Thread

from PIL import Image

# --- Add necessary imports from your ILLUME codebase ---
import torch
torch.backends.cudnn.allow_tf32 = True
from transformers import LogitsProcessorList, TextIteratorStreamer

from illume.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from illume.conversation import conv_templates, default_conversation  # Import Conversation class
from illume.mm_utils import process_images, tokenizer_image_token
from illume.data.data_utils import unpad_and_resize_back

from generation_eval.models.builder import build_eval_model
from generation_eval.models.inference_utils import CFGLogits, DualVQImageTokenProcessor, DynamicSamplingProcessor, \
    InterleavedLogitsProcessor, parse_interleaved_text_image, calculate_image_token_num, check_image_token_num

# --- End ILLUME Imports ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger("http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("PIL").setLevel(logging.WARNING) # Optional: Reduce PIL logging noise

import gradio as gr

# Use the Conversation class directly instead of the public one
# from conversation_public import default_conversation, conv_templates, SeparatorStyle # Remove this

# --- Global Variables and Model Loading ---
eval_model = None  # Global variable to hold the loaded ILLUME model
args = None  # Global variable to hold command line args
streamer = None  # Global variable to hold command line args

# Define common resolutions
DEFAULT_RESOLUTIONS = [
    (256, 256), (512, 512), (384, 640), (640, 384), (512, 384),
    (384, 512), (256, 384), (384, 256), (256, 512), (512, 256)
]

DEFAULT_DIFFUSION_RESOLUTIONS = [
    (512, 512), (1024, 1024), (768, 1280), (1280, 768), (1024, 768),
    (768, 1024), (512, 768), (768, 512), (512, 1024), (1024, 512)
]

# qwen2.5
special_tokens_ids = [151665, 151666, 151667, 151668, 151669, 151670, 151671]
start_token = 151672 + 32
level0_range = (start_token, start_token + 32768)  # Level 0 token ID 鑼冨洿
level1_range = (start_token + 32768, start_token + 32768 * 4)  # Level 1 token ID 鑼冨洿

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


def pad_sequence(tokenizer, input_ids, batch_first, padding_value):
    # Assuming input_ids is a list of Tensors
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    # Manually pad if needed, or use torch utils if input_ids are tensors
    # This assumes input_ids are already tensors
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids_padded = torch.flip(input_ids_padded, [1])
    return input_ids_padded


# --- Gradio UI Functions ---
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)
server_error_msg = "**NETWORK ERROR OR SERVER ISSUE. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
server_oom_msg = "**OUT OF GPU MEMORY DETECTED. PLEASE DECREASE THE MAX OUTPUT TOKENS OR IMAGE RESOLUTION AND REGENERATE.**"


def load_demo_refresh_model_list():
    logging.info("load_demo.")
    # Use the conversation template from the loaded model/config
    # Ensure eval_model is loaded before this runs
    if eval_model and hasattr(eval_model, 'config'):
        conv_template_name = eval_model.config.model_args.version
        if conv_template_name in conv_templates:
            state = conv_templates[conv_template_name].copy()
            logging.info(f"Using conversation template: {conv_template_name}")
        else:
            logging.warning(f"Conversation template '{conv_template_name}' not found. Using default.")
            # Find a default template name from conv_templates or define one
            default_template_name = next(iter(conv_templates))  # Get the first available template
            state = conv_templates[default_template_name].copy()
    else:
        logging.error("Eval model not loaded. Cannot initialize conversation state.")
        # Fallback or raise error
        state = default_conversation().copy()
    return state


def regenerate(state):
    logging.info("regenerate.")
    if not state.messages or len(state.messages) < 2:
        return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2  # Use state's image

    # Clear the last assistant message
    state.messages[-1][-1] = None

    state.skip_next = False
    # Return state, updated chatbot display, refill textbox, keep image
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2


def http_bot_conditional_then(state, temperature, top_k, top_p, max_output_tokens,
                              llm_cfg_scale, resolution_wh, use_diffusion, diffusion_cfg_scale,
                              diffusion_num_inference_steps):
    if state.mode == 'chat':
        result = yield from http_chat_bot(state, temperature, top_k, top_p, max_output_tokens)
    else:
        result = yield from http_gen_edit_bot(state, temperature, top_k, top_p, max_output_tokens,
                                              llm_cfg_scale, resolution_wh, use_diffusion, diffusion_cfg_scale,
                                              diffusion_num_inference_steps)
    return result


def clear_history():
    logging.info("clear_history.")
    state = load_demo_refresh_model_list()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2


def add_text(state, text, image, mode):
    global eval_model  # Ensure we use the loaded model

    logging.info(f"add_text. Text len: {len(text)}, Image provided: {image is not None}")
    if len(text.strip()) == 0 and image is None:
        state.skip_next = True
        # Keep image in the imagebox if only image was present
        return (state, state.to_gradio_chatbot(), "", image) + (no_change_btn,) * 2

    if state.messages and state.messages[-1][1] and \
            isinstance(state.messages[-1][1], str) and state.messages[-1][1].startswith("**"):
        state = load_demo_refresh_model_list()  # Start fresh after error
    if mode in ['image-generation']:
        state = load_demo_refresh_model_list()

    image_process_mode = "Default"

    if image is not None:
        if '<image>' not in text:
            text = f'<image>\n{text}'
        text = (text, image, image_process_mode)

    # Append user message
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)  # Placeholder for assistant
    state.skip_next = False
    state.mode = mode
    logging.info(f"Updated state messages: {len(state.messages)}")

    # Return new state, updated chatbot, clear textbox, clear imagebox
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2


def stream_response(model, inputs, streamer, prompt, gen_kwargs):
    thread = Thread(target=model.generate, kwargs=dict(
        streamer=streamer,
        **inputs,
        **gen_kwargs
    ))
    thread.start()

    generated_text = prompt

    for new_text in streamer:
        generated_text += new_text
        yield generated_text


# @spaces.GPU
def http_chat_bot(state, temperature, top_k, top_p, max_new_tokens):
    global eval_model, args, streamer  # Use global model and args
    logging.info("http_chat_bot.")

    if state.skip_next:
        logging.warning("Skipping bot generation. skip_next or model not ready.")
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
        return

    if len(state.messages) < 2:
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
        return

    # --- Prepare Inputs for ILLUME ---
    # Get the full prompt from the conversation state
    prompt = state.get_prompt()
    all_images = state.get_images(return_pil=True)

    logging.info(f"Raw Prompt: {prompt}")

    if len(all_images):
        # Use the process_images function from illume.mm_utils

        # process_images expects a list of PIL Images
        # Ratios might be needed depending on your config - using None for default
        images_tensor, image_sizes = process_images(
            all_images,  # Pass as a list
            eval_model.image_processor,
            eval_model.mllm_model.config,
            eval_model.config,
            is_gen_task=False,  # Assuming this is correct for general use
            ratios=DEFAULT_RESOLUTIONS  # Or get ratios based on UI selection if needed
        )

        if isinstance(images_tensor, list):
            logging.info(
                f"Processed image. Tensor shape: {[img.shape for img in images_tensor]}, Size info: {image_sizes}")
        else:
            logging.info(f"Processed image. Tensor shape: {images_tensor.shape}, Size info: {image_sizes}")
    else:
        # Clear any previously processed image if no new image is provided
        images_tensor = None
        image_sizes = None

    # Tokenize the prompt
    # Use tokenizer_image_token for potential image placeholder
    input_ids_list = [tokenizer_image_token(prompt, eval_model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
    pad_token_ids = eval_model.tokenizer.pad_token_id if eval_model.tokenizer.pad_token_id is not None else eval_model.tokenizer.eos_token_id
    input_ids = pad_sequence(eval_model.tokenizer, input_ids_list, batch_first=True, padding_value=pad_token_ids).to(
        eval_model.device)
    attention_masks = input_ids.ne(pad_token_ids).to(eval_model.device)
    logging.info(f"Input IDs shape: {input_ids.shape}")

    # Set initial response placeholder
    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2

    # --- MLLM Generation ---
    inputs = dict(
        inputs=input_ids,
        attention_mask=attention_masks,
        images=images_tensor,  # Pass the processed input image tensor
        image_sizes=image_sizes,  # Pass image sizes
    )

    logit_processor = InterleavedLogitsProcessor(
        # enable_generate_image=False,
        guidance_scale=1.0,
        uncond=None,
        model=eval_model.mllm_model,
        images=images_tensor,  # Pass input image if present
        image_sizes=image_sizes,
        level0_range=level0_range,
        level1_range=level1_range,
        num_level0_rows=0,
        num_level0_tokens=0,
        num_level1_rows=0,
        num_level1_tokens=0,
        special_tokens=special_tokens_dict,
        default_temp=temperature, level0_temp=1.0,level1_temp=1.0,
        default_top_k=top_k, level0_top_k=2048, level1_top_k=2048 * 3,
        default_top_p=top_p, level0_top_p=1.0, level1_top_p=1.0,
    )
    logits_processor = LogitsProcessorList([logit_processor])

    gen_kwargs = dict(
        pad_token_id=pad_token_ids,
        do_sample=True if temperature > 0 else False,  # Controlled by dynamic sampler now, but keep flag
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=eval_model.tokenizer.eos_token_id,
        logits_processor=logits_processor,
    )

    logging.info(f"==== request kwargs====\n{gen_kwargs}")

    if max_new_tokens < 1:
        state.messages[-1][-1] = "Exceeds max token length. Please start a new conversation, thanks."
        yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2
        return

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2

    # Stream output
    try:
        for generated_text in stream_response(eval_model.mllm_model, inputs, streamer, prompt, gen_kwargs):
            output = generated_text[len(prompt):].strip()
            state.messages[-1][-1] = output
            yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2
    except Exception as e:
        os.system("nvidia-smi")
        logging.info(traceback.print_exc())
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
    return (state, state.to_gradio_chatbot()) + (enable_btn,) * 2


def http_gen_edit_bot(state, temperature, top_k, top_p, max_output_tokens,
                      llm_cfg_scale, resolution_wh, use_diffusion, diffusion_cfg_scale, diffusion_num_inference_steps):
    global eval_model, args  # Use global model and args
    logging.info("http_gen_edit_bot.")

    if state.skip_next:
        logging.warning("Skipping bot generation. skip_next or model not ready.")
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
        return

    if len(state.messages) < 2:
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2

        return

    # --- Prepare Inputs for ILLUME ---
    # Get the full prompt from the conversation state
    all_images = state.get_images(return_pil=True)

    # Add image token to text if not present
    img_token = eval_model.config.model_args.get("image_token", DEFAULT_IMAGE_TOKEN)  # Get from config or use default

    # read resolution from user defined.
    h_str, w_str = resolution_wh.split('x')
    h_out, w_out = int(h_str), int(w_str)
    if use_diffusion:
        h_out, w_out = (h_out // 2, w_out // 2)
    else:
        h_out, w_out = (h_out, w_out)

    # Calculate ratio tag based on base resolution from config
    h_tag, w_tag = h_out, w_out
    ratio_tag = f"<height_{h_tag}><width_{w_tag}>"
    logging.info(f"Target Resolution: {h_out}x{w_out}, Ratio Tag: {ratio_tag}")

    input_state = state.copy()

    # prepare the text.
    original_image_sizes = None
    if len(all_images):  # image editing.
        # Use the process_images function from illume.mm_utils
        # Ensure eval_model and its components are loaded

        # process_images expects a list of PIL Images
        # Ratios might be needed depending on your config - using None for default
        original_image_sizes = [image.size for image in all_images]
        logging.info(f"original_image_sizes: {original_image_sizes}")
        images_tensor, image_sizes = process_images(
            all_images,  # Pass as a list
            eval_model.image_processor,
            eval_model.mllm_model.config,
            eval_model.config,
            is_gen_task=True,  # Assuming this is correct for general use
            ratios=DEFAULT_RESOLUTIONS  # Or get ratios based on UI selection if needed, # [(256, 256)]
        )

        w, h = image_sizes[0]
        ratio_tag = f"<height_{h}><width_{w}>"
        h_out, w_out = h, w
        logging.info(f"Target Resolution: {h_out}x{w_out}, Ratio Tag: {ratio_tag}")

        if isinstance(images_tensor, list):
            logging.info(
                f"Processed image. Tensor shape: {[img.shape for img in images_tensor]}, Size info: {image_sizes}")
        else:
            logging.info(f"Processed image. Tensor shape: {images_tensor.shape}, Size info: {image_sizes}")

        unconditional_text = f"{ratio_tag}{img_token}\nReconstruct the image according to the given image\n"  # of {ratio_tag}

        instruction, img, image_process_type = input_state.messages[-2][-1]
        print("instruction", instruction)
        instruction = instruction.replace(img_token, '').lstrip("\n")
        text = f"{ratio_tag}{img_token}\nPlease edit the image according to the instruction: {instruction}\n"
        input_state.messages[-2][-1] = text, img, image_process_type
    else:
        # image generation
        images_tensor = None
        image_sizes = None

        text = input_state.messages[-2][-1]
        logging.info(f"Current text is {text}")
        text = f"Generate an image of {ratio_tag}, the content of image is {text}\n"
        input_state.messages[-2][-1] = text
        logging.info(f"After padding. current text is {text}")

        unconditional_text = f"Generate a random image of {ratio_tag}\n"

    prompt = input_state.get_prompt()
    prompt += ratio_tag
    logging.info(f"Raw Prompt: {prompt}")

    # Tokenize the prompt
    # Use tokenizer_image_token for potential image placeholder
    input_ids_list = [tokenizer_image_token(prompt, eval_model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
    pad_token_ids = eval_model.tokenizer.pad_token_id if eval_model.tokenizer.pad_token_id is not None else eval_model.tokenizer.eos_token_id
    input_ids = pad_sequence(eval_model.tokenizer, input_ids_list, 
                             batch_first=True, padding_value=pad_token_ids).to(eval_model.device)
    attention_masks = input_ids.ne(pad_token_ids).to(eval_model.device)
    logging.info(f"Input IDs shape: {input_ids.shape}")

    # --- Prepare Logits Processors (Adapted from prepare_logit_processor) ---
    logits_processor_list = []

    # 1. CFG Logits Processor
    if llm_cfg_scale > 1.0:
        # Prepare unconditional prompt
        # Use a fixed unconditional prompt or get from dataset/config if available

        # Empty unconditional prompt common for generation
        # empty prompt
        conv_uncond = conv_templates[eval_model.config.model_args.version].copy()
        conv_uncond.append_message(conv_uncond.roles[0], unconditional_text)
        conv_uncond.append_message(conv_uncond.roles[1], None)
        unconditional_prompt_str = conv_uncond.get_prompt() + ratio_tag  # Add ratio tag
        print("unconditional_prompt_str", unconditional_prompt_str)
        unconditional_input_ids = tokenizer_image_token(unconditional_prompt_str, eval_model.tokenizer,
                                                        IMAGE_TOKEN_INDEX, return_tensors="pt")
        unconditional_input_ids = unconditional_input_ids.repeat(input_ids.shape[0], 1)

        # Pad unconditional prompt if needed to match batch size (1 in this case)
        unconditional_input_ids = unconditional_input_ids.to(eval_model.device)

        cfg_processor = CFGLogits(
            guidance_scale=llm_cfg_scale,
            uncond=unconditional_input_ids,
            model=eval_model.mllm_model,
            images=images_tensor,  # Pass input image if present
            image_sizes=image_sizes
        )
        logits_processor_list.append(cfg_processor)
        logging.info(f"Added CFGLogits with scale {llm_cfg_scale}")

    # 2. Strict Image Token Processor (DualVQImageTokenProcessor)
    token_nums, max_new_tokens, h1, w1, h2, w2 = calculate_image_token_num(h_out, w_out)

    global special_tokens_dict
    special_tokens_dict['start_of_vision_answer'] = eval_model.tokenizer.encode('<answer>')
    image_token_processor = DualVQImageTokenProcessor(
        level0_range=level0_range,
        level1_range=level1_range,
        num_level0_rows=h1,
        num_level0_tokens=w1,
        num_level1_rows=h2,
        num_level1_tokens=w2,
        special_tokens=special_tokens_dict  # Pass the whole dict
    )
    logits_processor_list.append(image_token_processor)
    logging.info("Added DualVQImageTokenProcessor")

    # 3. Dynamic Sampling Processor
    dynamic_sampling_processor = DynamicSamplingProcessor(
        special_tokens=special_tokens_dict,
        default_temp=0.7, level0_temp=temperature, level1_temp=temperature,
        default_top_k=20, level0_top_k=top_k, level1_top_k=top_k * 3,  # As per your code
        default_top_p=0.8, level0_top_p=top_p, level1_top_p=top_p,
    )
    logits_processor_list.append(dynamic_sampling_processor)
    logging.info(f"Added DynamicSamplingProcessor with temp={temperature}, top_k={top_k}, top_p={top_p}")

    # Combine into LogitsProcessorList
    final_logits_processor = LogitsProcessorList(logits_processor_list)

    # Set initial response placeholder
    state.messages[-1][-1] = "image generating..."
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2

    # --- MLLM Generation ---
    generated_image = None
    generated_text = ""
    try:
        with torch.inference_mode():  # Ensure no gradients are calculated
            output_ids = eval_model.mllm_model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=images_tensor,  # Pass the processed input image tensor
                image_sizes=image_sizes,  # Pass image sizes
                pad_token_id=pad_token_ids,
                do_sample=True if temperature > 0 else False,  # Controlled by dynamic sampler now, but keep flag
                temperature=1.0,  # Set to 1.0 as dynamic sampler handles it
                top_k=0,  # Set to 0 as dynamic sampler handles it
                top_p=1.0,  # Set to 1.0 as dynamic sampler handles it
                max_new_tokens=max_new_tokens,
                logits_processor=final_logits_processor,  # Use the combined processor
                use_cache=True,
                eos_token_id=eval_model.tokenizer.eos_token_id  # Ensure EOS token is set
            )

        logging.info(f"Generated output IDs shape: {output_ids.shape}")

        # Decode the generated IDs, skipping prompt and special tokens
        # We need to decode the full output first to parse image tokens
        # output_ids shape is likely (batch_size, seq_len), batch_size=1 here
        generated_ids = output_ids[0]  # Get only generated tokens
        full_output_text = eval_model.tokenizer.decode(
            generated_ids, skip_special_tokens=True)  # Decode WITH special tokens first for parsing
        logging.info(f"Full decoded output: {full_output_text}")

        # --- Parse Output for Image Tokens and Text ---
        # Ensure levels are sorted and create the final list
        num_levels = eval_model.config.model_args.get("vision_tokenizer_levels", 2)  # Get expected levels

        generated_text, image_embed_inds_list, list_image_token_parts = parse_interleaved_text_image(full_output_text,
                                                                                                     num_levels)

        assert len(image_embed_inds_list) == 1, 'The number of generated image should be 1.'
        image_embed_inds = image_embed_inds_list[0]
        logging.info(f"The generated text: {full_output_text}")
        logging.info(f"Parsed generated text (image presents as <image>): {generated_text}")

        # Update chat with generated text first
        state.messages[-1][-1] = "vision tokenizer decoding..."  # Remove cursor
        yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2  # Yield text update

        # --- Image Detokenization ---
        if any(image_embed_inds):
            logging.info("Image tokens found. Attempting detokenization...")
            yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2

            # Check and pad/truncate token numbers
            checked_embed_inds = check_image_token_num(image_embed_inds, token_nums, identifier="Bot")
            if checked_embed_inds is None:
                raise ValueError("Image token number mismatch during generation.")

            semantic_code_list = [checked_embed_inds[0]]  # Batch size 1
            texture_code_list = [checked_embed_inds[1]]  # Batch size 1

            with torch.inference_mode() and torch.cuda.amp.autocast(dtype=eval_model.torch_dtype):
                semantic_code = torch.as_tensor(semantic_code_list).to(eval_model.vq_device)
                texture_code = torch.as_tensor(texture_code_list).to(eval_model.vq_device)

                if use_diffusion:
                    semantic_code = semantic_code.view(semantic_code.shape[0], h1, w1)
                    texture_code = texture_code.view(texture_code.shape[0], h2, w2)

                    diffusion_outputs = eval_model.diffusion_decoder_pipe(
                        vq_indices=(semantic_code, texture_code),
                        height=h_out * 2,  # Use target output height
                        width=w_out * 2,  # Use target output width
                        guidance_scale=diffusion_cfg_scale,
                        num_inference_steps=diffusion_num_inference_steps  # Or make configurable
                    )
                    samples = diffusion_outputs.images  # List of PIL Images
                    if samples:
                        generated_image = samples[0]  # Get the first (only) image
                    logging.info(
                        f"Using Diffusion Decoder (cfg: {diffusion_cfg_scale}, steps: {diffusion_num_inference_steps}) Image size: {generated_image.size}")
                else:
                    semantic_code = semantic_code.view(semantic_code.shape[0], h1, w1)
                    texture_code = texture_code.view(texture_code.shape[0], h2, w2)

                    samples_tensor = eval_model.vq_model.decode_code(semantic_code, texture_code)

                    # Convert tensor to PIL Image
                    samples_tensor = torch.clamp(127.5 * samples_tensor + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu",
                                                                                                                dtype=torch.uint8).numpy()
                    assert samples_tensor.shape[0] == 1
                    generated_image = Image.fromarray(samples_tensor[0])
                    logging.info(f"Using VQ Tokenizer Decoder. Image size: {generated_image.size}")

            if generated_image:
                if original_image_sizes is not None and len(original_image_sizes) == 1:
                    # editing task, unpad and resize image to original size
                    original_size = original_image_sizes[0]
                    logging.info(f"original size: {original_size}. Output Image size: {generated_image.size}")
                    generated_image = unpad_and_resize_back(generated_image, original_size[0], original_size[1])
                    logging.info(f"final image size: {generated_image.size}")
                logging.info("Image successfully generated.")
                # <image> is placeholder.
                state.messages[-1][-1] = ('<image>', [generated_image], list_image_token_parts)
            else:
                logging.error("Image generation failed or produced no output.")
                state.messages[-1][-1] = "(Image generation failed)"
        else:
            # No image tokens generated or no decoder enabled
            state.messages[-1][-1] = generated_text  # Final text without image

        # Final yield with potentially updated message (text + image)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"CUDA OutOfMemoryError during generation: {e}\n{traceback.format_exc()}")
        state.messages[-1][-1] = server_oom_msg
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
    except Exception as e:
        logging.error(f"Error during model generation or detokenization: {e}\n{traceback.format_exc()}")
        state.messages[-1][-1] = f"{server_error_msg}\n```\n{traceback.format_exc()}\n```"  # Show traceback in error
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2

    logging.info(f"Final Assistant Message Length: {len(state.messages[-1][-1])}")


def update_resolution_dropdown(diffusion_enabled, current_resolution_str):
    logging.info(f"Updating resolution dropdown. Diffusion: {diffusion_enabled}, Current: {current_resolution_str}")
    current_h_str, current_w_str = current_resolution_str.split('x')
    current_h, current_w = int(current_h_str), int(current_w_str)

    new_value_str = None
    if diffusion_enabled:
        new_h, new_w = int(current_h) * 2, int(current_w) * 2

        if (new_h, new_w) not in DEFAULT_DIFFUSION_RESOLUTIONS:
            new_h, new_w = DEFAULT_DIFFUSION_RESOLUTIONS[0]
        new_value_str = f"{new_h}x{new_w}"
        return gr.Dropdown.update(choices=[f'{h}x{w}' for h, w in DEFAULT_DIFFUSION_RESOLUTIONS],
                                  value=new_value_str)
    else:
        new_h, new_w = int(current_h) // 2, int(current_w) // 2

        if (new_h, new_w) not in DEFAULT_RESOLUTIONS:
            new_h, new_w = DEFAULT_RESOLUTIONS[0]
        new_value_str = f"{new_h}x{new_w}"

        return gr.Dropdown.update(choices=[f'{h}x{w}' for h, w in DEFAULT_RESOLUTIONS],
                                  value=new_value_str)


# --- Gradio Layout ---
title_markdown = """
<div style="display: flex; align-items: center; padding: 20px; border-radius: 10px; background-color: #f0f0f0;">
  <div>
    <h1 style="margin: 0;"> ILLUME+: Illuminating Unified MLLM with Dual Visual Tokenization and Diffusion Refinement</h1>
    <h2 style="margin: 10px 0;">
      <a href="https://arxiv.org/abs/2504.01934" target="_blank" rel="noopener noreferrer">Paper</a> |
      <a href="https://github.com/illume-unified-mllm/ILLUME_plus" target="_blank" rel="noopener noreferrer">Code</a> |
      <a href="https://huggingface.co/ILLUME-MLLM/illume_plus-qwen2_5-3b-hf" target="_blank" rel="noopener noreferrer">Model</a> |
      <a href="https://illume-unified-mllm.github.io/" target="_blank" rel="noopener noreferrer">Project Page</a>
    </h2>
    <ul style="margin: 20px 0; padding-left: 20px;">
      <li><strong>1.</strong> Enter text and/or upload an image.</li>
      <li><strong>2.</strong> Click the 💬 <strong>Chat</strong> button for image inputted conversations</li>
      <li><strong>3.</strong> Click the 🖼️ <strong>Generate</strong> for image generation and image editing.</li>
      <li><strong>4.</strong> (Optional) Enable Diffusion Decoder for image super resolution decoding. 
      <li><strong>5.</strong> Adjust generation parameters if needed. 
        <br/><strong>💡 Tip 1:</strong> For better image generation quality, we recommend setting <code>temperature = 1.0</code>, <code>top_k = 2048</code>, <code>top_p = 1.0</code>, <code>llm_cfg = 2.0</code>.    
        <br/><strong>💡 Tip 2:</strong> For better image editing quality, we recommend setting <code>temperature = 0.7</code>, <code>top_k = 512</code>, <code>top_p = 0.8</code>, <code>llm_cfg = 1.5</code>.
        <br/><strong>💡 Tip 3:</strong> For diffusion decoder, CFG scale of 1.5 or 2.0 is enough.
      </li>
    </ul>
  </div>
</div>
"""


learn_more_markdown = ("""
## Citation


    @article{huang2025illume_plus,
      title={ILLUME+: Illuminating Unified MLLM with Dual Visual Tokenization and Diffusion Refinement},
      author={Huang, Runhui and Wang, Chunwei and Yang, Junwei and Lu, Guansong and Yuan, Yunlong and Han, Jianhua and Hou, Lu and Zhang, Wei and Hong, Lanqing and Zhao, Hengshuang and Xu, Hang}
      journal={arXiv preprint arXiv:2504.01934},
      year={2025}
    }
""")

block_css = """


#buttons button {
    min-width: min(120px,100%);
}
.message-row img {
    max-width: 80%;
    max-height: 400px;
    height: auto;
    display: block;
    margin-top: 10px;
    margin-bottom: 5px;
    border-radius: 5px;
    border: 1px solid #e0e0e0; /* Add a light border */
}
.avatar-container img {
    padding: 0px !important;
}
/* Style for resolution dropdown */
#resolution_dropdown .gradio-dropdown {
    min-width: 150px !important;
}
"""


# (Keep load_demo_refresh_model_list as it is)

def load_initial_state_and_example1():
    """
    Loads the initial Conversation state and prepares the inputs
    for the first example to populate the UI on startup.
    """
    logging.info("Loading initial state and Example 1 inputs for UI.")

    # 1. Get the base initial state object
    initial_state = load_demo_refresh_model_list()
    # At this point, initial_state is a Conversation object with empty messages.
    initial_state = 'chat'

    # 2. Define Example 1 inputs
    image_path = "../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/1.png"  # Make sure this path is correct relative to where you run the script
    text_prompt = "Describe this scene in detail."
    image_pil = None

    # 3. Load the example image
    try:
        # Ensure the image file exists and load it
        if os.path.exists(image_path):
            image_pil = Image.open(image_path)
            logging.info(f"Successfully loaded example image: {image_path}")
        else:
            logging.warning(f"Example image not found at: {image_path}. Image box will be empty.")
            # Optionally provide a placeholder blank image?
            # image_pil = Image.new('RGB', (60, 30), color = 'red') # Example placeholder
    except Exception as e:
        logging.error(f"Error loading example image {image_path}: {e}")
        image_pil = None  # Ensure it's None on error

    # 4. Return values to populate the UI components
    #    - state: The initial Conversation object
    #    - chatbot: The initial empty chatbot display ([]) derived from the initial state
    #    - textbox: The example text prompt
    #    - imagebox: The loaded PIL image (or None)
    return initial_state, initial_state.to_gradio_chatbot(), text_prompt, image_pil


def build_demo(embed_mode):
    textbox = gr.Textbox(label="Text Input / Prompt", show_label=False,
                         placeholder="Enter text prompt. Ask about the image or request image generation...",
                         container=False, scale=8)

    with gr.Blocks(title="ILLUME Demo", theme=gr.themes.Default(), css=block_css) as demo:
        conversation_state = gr.State()  # Holds conversation state (instance of illume.conversation.Conversation)

        if not embed_mode:
            gr.HTML(title_markdown)

        with gr.Row():
            with gr.Column(scale=2):
                imagebox = gr.Image(type="pil", label="Input Image", height=300)

                # Text Generation Parameters
                with gr.Accordion("Text Generation Parameters", open=True):
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.5, value=1.0, step=0.1,
                        label="Temperature",
                        info="Controls randomness of the output (higher = more diverse)."
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=4096, value=128, step=1,
                        label="Top-K",
                    )
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                        label="Top-P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=128, maximum=8192, value=1024, step=128,
                        label="Max Output Tokens",
                    )

                # Image Generation Parameters
                with gr.Accordion("Image Generation Parameters", open=True):
                    image_gen_temperature = gr.Slider(
                        minimum=0.0, maximum=1.5, value=1.0, step=0.1,
                        label="Temperature",
                    )
                    image_gen_top_k = gr.Slider(
                        minimum=1, maximum=4096 * 2, value=2048, step=32,
                        label="Top-K",
                        info="Recommended value for better image generation: 2048."
                    )
                    image_gen_top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                        label="Top-P",
                    )

                    resolution_wh_dropdown = gr.Dropdown(
                        [f'{h}x{w}' for h, w in DEFAULT_RESOLUTIONS],
                        value="512x512",
                        label="Output Resolution (HxW)",
                        elem_id="resolution_dropdown",
                        info="Select target size for generated images."
                    )

                    llm_cfg_scale = gr.Slider(
                        minimum=1.0, maximum=10.0, value=2.0, step=0.1,
                        label="LLM CFG Scale",
                        info="Guidance for text-to-image conditioning (higher = stricter to prompt)."
                    )

                    with gr.Accordion("Diffusion Decoder (Optional)", open=False):
                        use_diffusion_checkbox = gr.Checkbox(
                            value=False, interactive=True,
                            label="Use diffusion decoder for image generation",
                            info="Enable diffusion decoder."
                        )
                        diffusion_cfg_scale = gr.Slider(
                            minimum=1.0, maximum=15.0, value=2.0, step=0.1,
                            label="Diffusion CFG Scale",
                            info="Guidance strength for diffusion decoder."
                        )
                        diffusion_num_inference_steps = gr.Slider(
                            minimum=5, maximum=100, value=20, step=5,
                            label="Diffusion Inference Steps",
                            info="Number of steps during denoising."
                        )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="ILLUME Chat",
                    layout="bubble",
                    height=650,  # Increased height
                    bubble_full_width=False,
                    render_markdown=True  # Crucial for images
                )
                with gr.Row():
                    textbox.render()
                with gr.Row(elem_id="buttons") as button_row:
                    chat_btn = gr.Button(value="💬 Chat", variant="primary")
                    gen_btn = gr.Button(value="🖼️ Generate", variant="secondary")
                with gr.Row(elem_id="additional-buttons") as button_row_additional:
                    regenerate_btn = gr.Button(value="🔄 Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️ Clear History", interactive=False)

        # Update examples for ILLUME
        with gr.Accordion("Examples (Click to Load)", open=True):
            with gr.Row():
                gr.Examples(examples=[
                    ["../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/1.png",
                     "What are they doing?"],
                    ["../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/2.png",
                     "Depict the image in detail."],
                    ["../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/3.png",
                     "Parse the table"],
                ], inputs=[imagebox, textbox], label='Image Understanding Examples')

                gr.Examples(examples=[
                    [None, "a cat with a hat."],
                    [None, "a smiling child."],
                    [None, "tiger cub playing with soccer ball"],
                    [None, "screenshot from a 16 bit platform game in a lush green landscape"],
                    [None, "Old car in kandy sri lanka,lake road,flower, bright, sunny, orange sky"],
                    [None, "Create a vibrant painting of a tropical beach at sunset."],
                ], inputs=[imagebox, textbox], label='Image Generation Examples')

                gr.Examples(examples=[
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/0.jpg",
                     "Change the color of the boots to a deep forest green"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/2.jpg",
                     "Remove the dried flowers"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/3.jpg",
                     "Change it into winter"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/4.jpg",
                     "Delete the tennis racket from the man's hand"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/5.jpg",
                     "Show me this as it would appear in a comic book"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/1.jpg",
                     "Add a hat on the dog"],
                ], inputs=[imagebox, textbox], label='Image Editing Examples')

        if not embed_mode:
            gr.Markdown(learn_more_markdown)

        # Register listeners
        btn_list = [regenerate_btn, clear_btn]
        parameter_chat_inputs = [temperature, top_k, top_p, max_output_tokens]
        parameter_gen_edit_inputs = [image_gen_temperature, image_gen_top_k, image_gen_top_p, max_output_tokens,
                                     llm_cfg_scale, resolution_wh_dropdown,
                                     use_diffusion_checkbox, diffusion_cfg_scale, diffusion_num_inference_steps]

        regenerate_btn.click(
            regenerate,
            [conversation_state],
            [conversation_state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_bot_conditional_then,
            [conversation_state] + parameter_gen_edit_inputs,  # Pass state and all params
            [conversation_state, chatbot] + btn_list,
        )

        clear_btn.click(
            clear_history,
            None,
            [conversation_state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        # Default use chat.
        textbox.submit(
            partial(add_text, mode="chat"),
            [conversation_state, textbox, imagebox],
            [conversation_state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_chat_bot,
            [conversation_state] + parameter_chat_inputs,
            [conversation_state, chatbot] + btn_list,
        )

        # Regular Vision-language Chat
        chat_btn.click(partial(add_text, mode="chat"),
                       [conversation_state, textbox, imagebox],
                       [conversation_state, chatbot, textbox, imagebox] + btn_list,
                       queue=False
                       ).then(
            http_chat_bot,
            [conversation_state] + parameter_chat_inputs,
            [conversation_state, chatbot] + btn_list,
        )

        # Image Generation
        gen_btn.click(
            partial(add_text, mode="image-generation"),
            [conversation_state, textbox, imagebox],
            [conversation_state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_gen_edit_bot,
            [conversation_state] + parameter_gen_edit_inputs,
            [conversation_state, chatbot] + btn_list
        )

        use_diffusion_checkbox.change(
            fn=update_resolution_dropdown,
            inputs=[use_diffusion_checkbox, resolution_wh_dropdown],
            outputs=[resolution_wh_dropdown],
            queue=False
        )

        # Load initial state when demo starts
        demo.load(
            load_demo_refresh_model_list,
            None,
            conversation_state,
            queue=False
        )
    return demo


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- Add arguments for ILLUME configs and checkpoints ---
    parser.add_argument("--model_name", type=str, default="ILLUME", help="Name for builder.")
    parser.add_argument("--config", type=str, required=True, help="Path to MLLM config file (e.g., config.py).")
    parser.add_argument("--torch_dtype", type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'],
                        help="Computation data type.")

    # Add diffusion/tokenizer version names if needed by builder, or read from config
    parser.add_argument("--diffusion_decoder_path", type=str, default='/path/to/dualvitok_sdxl_decoder',
                        help="Path to Diffusion Decoder checkpoint. Required if using diffusion.")

    parser.add_argument("--tokenizer_config", type=str, required=True,
                        help="Path to Tokenizer config file (e.g., tokenizer_config.py).")
    parser.add_argument("--tokenizer_checkpoint", type=str, default=None,
                        help="Path to VQ Tokenizer checkpoint (.pth).")

    # --- End ILLUME arguments ---
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--embed", action="store_true", help="Run in embed mode (minimal UI)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda, cpu).")

    args = parser.parse_args()

    # --- Model Loading ---
    # Set device
    if "cuda" in args.device and torch.cuda.is_available():
        device = args.device
        local_rank = 0  # Assume single GPU for Gradio unless configured otherwise
        torch.cuda.set_device(local_rank)  # Set default CUDA device
    else:
        device = "cpu"
        local_rank = -1  # Indicate CPU
    logging.info(f"Using device: {device}")

    # Prepare eval_model_cfg dictionary from args
    eval_model_cfg = dict(
        type=args.model_name,
        config=args.config,
        tokenizer_config=args.tokenizer_config,
        diffusion_decoder_path=args.diffusion_decoder_path,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        torch_dtype=args.torch_dtype
        # Add other necessary fields expected by your builder if any
    )

    # Build the ILLUME model instance
    logging.info("Building ILLUME model...")
    try:
        # Pass device info to the builder if it accepts it, otherwise models are moved later
        # Add device=device, local_rank=local_rank if builder supports them
        eval_model = build_eval_model(eval_model_cfg)

        num_gpus = torch.cuda.device_count()
        logging.info(f"Detected {num_gpus} CUDA devices.")
        # Determine device assignments
        if num_gpus >= 3:
            # Assign to separate GPUs if we have at least 3
            mllm_device = torch.device('cuda:0')
            vq_device = torch.device('cuda:1')
            diffusion_device = torch.device('cuda:2')
            logging.info("Assigning models: MLLM -> cuda:0, VQ -> cuda:1, Diffusion -> cuda:2")
        elif num_gpus == 2:
            # Assign MLLM to GPU 0, VQ and Diffusion to GPU 1
            mllm_device = torch.device('cuda:0')
            vq_device = torch.device('cuda:1')
            diffusion_device = torch.device('cuda:1')
            logging.info("Assigning models: MLLM -> cuda:0, VQ & Diffusion -> cuda:1")
        elif num_gpus == 1:
            # Assign all to the single available GPU
            mllm_device = torch.device('cuda:0')
            vq_device = torch.device('cuda:0')
            diffusion_device = torch.device('cuda:0')
            logging.info("Assigning all models to cuda:0")
        else:
            # Fallback to CPU if no GPUs are available
            mllm_device = torch.device('cpu')
            vq_device = torch.device('cpu')
            diffusion_device = torch.device('cpu')
            logging.info("Warning: No CUDA devices found. Assigning all models to CPU.")

        # Manually move components to device if not handled by builder/load_pretrained_model
        if hasattr(eval_model, 'mllm_model') and eval_model.mllm_model:
            eval_model.mllm_model.to(mllm_device)
        if hasattr(eval_model, 'vq_model') and eval_model.vq_model:
            eval_model.vq_model.to(vq_device)
        if hasattr(eval_model, 'diffusion_decoder_pipe') and eval_model.diffusion_decoder_pipe:
            eval_model.diffusion_decoder_pipe.to(diffusion_device)  # Move the whole pipeline

        # Assign device to eval_model for later use
        eval_model.device = device
        eval_model.mllm_device = mllm_device
        eval_model.vq_device = vq_device
        eval_model.diffusion_device = diffusion_device
        eval_model.local_rank = local_rank

        streamer = TextIteratorStreamer(eval_model.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        logging.info("ILLUME model built successfully.")

    except Exception as e:
        logging.error(f"Failed to build ILLUME model: {e}\n{traceback.format_exc()}")
        print("Error: Model loading failed. Please check config paths, checkpoints, and dependencies.")
        exit(1)  # Exit if model loading fails

    demo = build_demo(args.embed)
    demo.queue(
        max_size=10,
        api_open=False
    ).launch(
        share=args.share,
        server_name="0.0.0.0"  # Allow network access if not using --share
    )