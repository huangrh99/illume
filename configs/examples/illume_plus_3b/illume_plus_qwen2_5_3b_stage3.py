_base_ = [
    '../../_base_/models/qwen2_5_dualencoder.py',
    '../../_base_/training/default.py',
]

data_args = dict(
    meta_data_info_list=[
        dict(
            meta_configs=["../configs/data_configs/train_data_examples/examples_meta_data_config.json"],
            dataset_dtype="DefaultDataset",  # for understanding and text-only task
            codebook_dir="jsonl_after_tokenizer/dualvitok/fixed_256/",
            data_infos={
                ### list the dataset you want to use, and the corresponding sample num
                #
                # "dataset_name": sample_num, # sample_num < 0 mean sample all data
                #
                "examples_understanding": -1,
            },
            image_aspect_ratio='anyres_dualvitok',  # using naive image resolution for understanding task
        ),
        dict(
            meta_configs=["../configs/data_configs/train_data_examples/examples_meta_data_config.json"],
            dataset_dtype="Text2ImageDataset",  # for text-to-image generation task
            codebook_dir="jsonl_after_tokenizer/dualvitok/fixed_256/",
            data_infos={
                "examples_generation": -1,
            },
            image_aspect_ratio='anyres_dualvitok_fix_anchors',  # for generation/editing task, we set multiple aspect ratios, image will be matched to one
        ),
        dict(
            meta_configs=["../configs/data_configs/train_data_examples/examples_meta_data_config.json"],
            dataset_dtype="SingleTurnEditDataset",  # for image editing task
            codebook_dir="jsonl_after_tokenizer/dualvitok/fixed_256/",
            data_infos={
                "examples_editing": -1,
            },
            image_aspect_ratio='anyres_dualvitok_fix_anchors',  # for generation/editing task, we set multiple aspect ratios, image will be matched to one
        ),
    ],
    lazy_preprocess=True,
    is_multimodal=False,
    base_resolution=256,
    cfg_text_dropout_probability=0.1,  # probability for random dropout text conditions
    image_aspect_ratio='anyres_dualvitok',  # image processing method for understanding in inference
    image_aspect_ratio_generation="anyres_dualvitok_fix_anchors"  # image processing method for generation in inference
)

model_args = dict(
    version="qwen2",
    tune_mm_mlp_adapter=True,

    vision_tokenizer_levels=2,
    text_token_num=151665,
    unfreeze_vision_embedding=True,
    unfreeze_text_embedding=True,

    language_model=dict(
        type='IllumeQwen2ForCausalLM',
        # your can extend the llm's vocabulary by using ILLUME/scripts/prepare_llm_with_extended_vision_tokenizer.py
        pretrained_model_name_or_path='../checkpoints/Qwen2.5-3B-Instruct-with-vision-tokenizer-32k-96k-level2',
        attn_implementation="sdpa",  # flash_attention_2
        trainable=True
    ),
    mm_vision_tower=dict(
        type='DualVisionTower',
        min_pixels=32 * 32,
        max_pixels=768 * 768,
        trainable=True,
        unfreeze_mm_vision_tower=True,
        tune_vit_from_layer=16
    ),
    mm_projector=dict(
        trainable=True
    )
)

training_args = dict(
    output_dir="./logdir/illume_plus_3b/illume_plus-qwen2_5-3b_stage3/",
    deepspeed="./scripts/zero2.json",
    bf16=True,
    tf32=True,
    fp16=False,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,  # 256 bs, 256 npu
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=1,
    save_on_each_node=True,
    mm_vision_tower_lr=2e-6,
    learning_rate=2e-5,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    model_max_length=4096,
    gradient_checkpointing=True,
    dataloader_num_workers=1,
    group_by_modality_length=False,
    report_to="tensorboard",
    seed=423,
    adam_beta2=0.95,  ##
)
