#!/bin/bash

# Inference on text-to-image generation task
# Ensure you are in the ILLUME directory before running this script,
# or adjust paths accordingly.

CODE_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CODE_DIR
export PYTHONPATH=$PYTHONPATH:$CODE_DIR/../vision_tokenizer/

# --- IMPORTANT: Make sure you have downloaded these checkpoints ---
TOKENIZER_CKPT=../checkpoints/dualvitok/pytorch_model.bin
DIFFUSION_CKPT=../checkpoints/dualvitok-sdxl-decoder
# ------ 

# MLLM Model setting
CONFIG_NAME=../configs/example/illume_plus_3b/illume_plus_qwen2_5_3b_stage3.py
# Tokenizer setting
TOKENIZER_CONFIG=../configs/example/dualvitok/dualvitok_anyres_max512.py

# Dataset and inference parameter setting
INFERENCE_BATCH_SIZE=8
DATASET_NAME=Text2ImageExampleDataset # See example format: ../configs/data_configs/test_data_examples/Text2ImageExample/t2i_test_examples.jsonl
TEMPERATURE=1.0
TOP_K=2048
TOP_P=1.0
LLM_CFG_SCALE=2.0
DIFFUSION_CFG_SCALE=2.0
TORCH_DTYPE=bf16 # or fp16, fp32

echo "Starting text-to-image generation inference..."

torchrun --nproc_per_node=8 generation_eval/main.py \
    --mllm_config $CONFIG_NAME --tokenizer_config $TOKENIZER_CONFIG \
    --batch_size $INFERENCE_BATCH_SIZE --torch_dtype $TORCH_DTYPE \
    --chosen_datasets $DATASET_NAME \
    --temperature $TEMPERATURE --top_k $TOP_K --top_p $TOP_P \
    --llm_cfg_scale $LLM_CFG_SCALE --diffusion_cfg_scale $DIFFUSION_CFG_SCALE \
    --tokenizer_checkpoint $TOKENIZER_CKPT \
    --diffusion_decoder_path $DIFFUSION_CKPT

echo "Text-to-image inference finished." 