# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import argparse
import os
import copy
from dataclasses import dataclass, field, fields
import json
import logging
import pathlib

import numpy as np
import torch

import transformers
import tokenizers

from illume.data.dataset import make_supervised_data_module
from illume.model.language_model.builder import build_language_model
from illume.train.illume_trainer import ILLUMETrainer
from illume import conversation as conversation_lib
from illume.model import *

from illume.utils import read_config, set_local_rank, rank0_print, smart_tokenizer_and_embedding_resize, \
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer, \
    find_all_linear_names

from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    print('import torch_npu successfully')
except Exception as e:
    print("import torch_npu failed.")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    def __post_init__(self):
        super(TrainingArguments, self).__post_init__()


def parse_args():
    parser = argparse.ArgumentParser(description="ILLUME Args.")
    parser.add_argument("config", type=str, help='config')
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args, unknown = parser.parse_known_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args, unknown


def read_args():
    args, unknown = parse_args()
    cfg = read_config(args.config)

    overrides = []
    for item in unknown:
        if item.startswith('--'):
            item = item.strip('--')
        k, v = item.split('=')
        overrides.append((k, v))

    # Apply overrides
    for key, value in overrides:
        try:
            cfg.set_nested(key, value)
        except AttributeError as e:
            print(f"Warning: {e}")

    fields_name = set([f.name for f in fields(TrainingArguments)])

    extract_args = dict()
    for k, v in cfg.training_args.items():
        if k in fields_name:
            extract_args[k] = v

    training_args = TrainingArguments(**extract_args)
    for k, v in cfg.training_args.items():
        if not hasattr(training_args, k):
            setattr(training_args, k, v)

    cfg.training_args = training_args

    if hasattr(cfg.training_args, 'use_liger_kernel') and cfg.training_args.use_liger_kernel:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2()
        rank0_print("Using liger kernel for qwen models.")
    return cfg


def train(attn_implementation=None):
    cfg = read_args()

    model_args, data_args, training_args = cfg.model_args, cfg.data_args, cfg.training_args

    local_rank = training_args.local_rank
    set_local_rank(local_rank)
    rank0_print(model_args, data_args, training_args)

    # Avoid resume the old checkpoint. But it's dangerous!!!!
    if local_rank == 0 and hasattr(training_args, 'clean_cache_checkpoint'):
        if training_args.clean_cache_checkpoint and os.path.exists(training_args.output_dir):
            import shutil
            shutil.rmtree(training_args.output_dir)
            os.makedirs(training_args.output_dir)

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if hasattr(training_args, 'bits') and training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    training_args.tune_language_model = model_args.language_model.get('trainable', False)
    training_args.tune_vision_tower = model_args.mm_vision_tower.get('trainable', False)
    training_args.tune_mm_mlp_adapter = model_args.mm_projector.get('trainable', True)
    is_pretrain_stage = not training_args.tune_language_model

    model = build_language_model(model_args.language_model,
                                 cache_dir=training_args.cache_dir,
                                 **bnb_model_from_pretrained_args)
    print("build language model done")

    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    pretrained_model_name_or_path = model_args.language_model.pretrained_model_name_or_path
    if 'mpt' in pretrained_model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif 'glm' in pretrained_model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
        )
    elif 'qwen' in pretrained_model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=True,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    else:
        if 'think' in model_args.version:
            # It is thinking mode. We need to add <think>, </think>, <answer>, </answer>
            rank0_print(f"Adding think tokens as '<think>', '</think>', '<answer>', '</answer>' if not already present")
            original_embed_size = len(tokenizer)
            special_tokens = ['<think>', '</think>', '<answer>', '</answer>']

            # to check if the model already has this token.
            tokens_to_add = []
            for token in special_tokens:
                if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
                    tokens_to_add.append(tokenizers.AddedToken(token, single_word=True, rstrip=False, lstrip=False))

            if tokens_to_add:
                added_tokens_num = tokenizer.add_tokens(tokens_to_add)
                new_tokenizer_size = len(tokenizer)
                if new_tokenizer_size > original_embed_size:
                    new_embed_size = int(np.ceil(new_tokenizer_size / 128) * 128)
                    model.resize_token_embeddings(new_embed_size)

            for token in ['<think>', '</think>', '<answer>', '</answer>']:
                rank0_print(f'Add token {token} and token id is {tokenizer.encode(token)}')

        if tokenizer.pad_token is None:
            rank0_print(f"Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
        rank0_print(f"Using conversation format: {conversation_lib.default_conversation.version}")
    print("build tokenizer done")

    if model_args.mm_vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.special_tokens_ids = model_args.get("special_tokens_ids", [])
        if data_args.image_aspect_ratio == 'anyres' or data_args.image_aspect_ratio == 'anyres_qwen_patchfy':
            try:
                base_size = vision_tower.config.image_size
            except:  # for qwen2vit
                base_size = model_args.mm_vision_tower.base_resolution
                data_args.base_resolution = base_size

            if data_args.get('max_num_slices', None):
                from illume.mm_utils import set_max_image_pixels
                from illume.utils import find_possible_grids
                max_num_slices = data_args.get('max_num_slices')
                set_max_image_pixels(max_num_slices * base_size * base_size)
                grids = find_possible_grids(max_num_slices)
            else:
                if hasattr(data_args, 'grids'):
                    grids = data_args.grids
                else:
                    grids = [[1, 2], [2, 1], [2, 2], [3, 1], [1, 3]]
            rank0_print(f"Enabling any-resolution training. Grid: {grids}")
            model.config.image_grid_pinpoints = data_args.image_grid_pinpoints = [
                [g[0] * base_size, g[1] * base_size] for g in grids]

            if data_args.get('dynamic_max_num_slices', None) and data_args.get('dynamic_max_num_slices_probability',
                                                                               None):  # dynamic max num slice during training
                from illume.mm_utils import set_dynamic_max_num_slices
                from illume.utils import find_possible_grids
                dynamic_max_num_slices = data_args.get('dynamic_max_num_slices')
                dynamic_grids = find_possible_grids(dynamic_max_num_slices)
                dynamic_image_grid_pinpoints = [[g[0] * base_size, g[1] * base_size] for g in dynamic_grids]
                dynamic_max_num_slices_probability = data_args.get('dynamic_max_num_slices_probability')
                set_dynamic_max_num_slices(dynamic_image_grid_pinpoints, dynamic_max_num_slices_probability)
                rank0_print(
                    f"Setting the dynamic_max_num_slices {dynamic_max_num_slices} with probability {dynamic_max_num_slices_probability}")

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        if training_args.tune_vision_tower:
            model.get_vision_tower().tune()
        else:
            model.get_vision_tower().freeze()

        if training_args.tune_mm_mlp_adapter:
            model.get_mm_projector().tune()
        else:
            model.get_mm_projector().freeze()

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        rank0_print(
            f"==============================training_args.mm_projector_lr{training_args.mm_projector_lr}=============================")
        if hasattr(training_args, 'mm_vision_tower_lr'):
            rank0_print(
                f"==============================training_args.mm_vision_tower_lr{training_args.mm_vision_tower_lr}=============================")

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if hasattr(training_args, 'enable_skip_ignore_index_in_lm_head'):
        model.enable_skip_ignore_index_in_lm_head()

    if hasattr(training_args, 'record_sample_loss'):
        model.enable_record_sample_loss()

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    rank0_print(model)
    for n, p in model.named_parameters():
        if p.requires_grad:
            rank0_print("Trainable Param", n, p.dtype)
        else:
            rank0_print("Non-trainable Param", n, p.dtype)

    data_args.group_by_modality_length = training_args.group_by_modality_length
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    trainer = ILLUMETrainer(model=model,
                            tokenizer=tokenizer,
                            args=training_args,
                            **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print(f"Resuming from {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir,
                                       is_pretrain_stage=is_pretrain_stage)

    try:
        import moxing as mox
        if hasattr(training_args, 'remote_work_dir'):
            # mox.file.copy_parallel(training_args.output_dir,
            #                        training_args.remote_work_dir)
            for item in os.listdir(training_args.output_dir):
                local_path = os.path.join(training_args.output_dir, item)
                remote_path = os.path.join(training_args.remote_work_dir, item)
                if os.path.isdir(local_path) and item.startswith("checkpoint-"):
                    rank0_print(f"Skipping checkpoint directory: {local_path}")
                    continue
                mox.file.copy_parallel(local_path, remote_path)
            rank0_print(f"Moxing: copy work dir from {training_args.output_dir} to {training_args.remote_work_dir}.")
    except Exception as e:
        rank0_print(f"Checkpoint saving in {training_args.output_dir}")


if __name__ == "__main__":
    train()
