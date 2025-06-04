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
            image_aspect_ratio='anyres_dualvitok',
        ),
        dict(
            meta_configs=["../configs/data_configs/train_data_examples/examples_meta_data_config.json"],
            dataset_dtype="Text2ImageDataset",  # for text-to-image generation task
            codebook_dir="jsonl_after_tokenizer/dualvitok/fixed_256/",
            data_infos={
                "examples_generation": -1,
            },
            image_aspect_ratio='anyres_dualvitok_fix_anchors',
        ),
        dict(
            meta_configs=["../configs/data_configs/train_data_examples/examples_meta_data_config.json"],
            dataset_dtype="SingleTurnEditDataset",  # for image editing generation task
            codebook_dir="jsonl_after_tokenizer/dualvitok/fixed_256/",
            data_infos={
                "examples_editing": -1,
            },
            image_aspect_ratio='anyres_dualvitok_fix_anchors',
        ),
    ],
    lazy_preprocess=True,
    is_multimodal=False,
    base_resolution=256,
    cfg_text_dropout_probability=0.1,
    image_aspect_ratio='anyres_dualvitok',
    image_aspect_ratio_generation="anyres_dualvitok_fix_anchors"
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
        pretrained_model_name_or_path='../checkpoints/Qwen2.5-3B-Instruct-with-vision-tokenizer-32k-96k-level2',
        attn_implementation="sdpa",
        trainable=True
    ),
    mm_vision_tower=dict(
        type='DualVisionTower',
        vq_config="configs/example/dualvitok/dualvitok_anyres_max512.py",
        vq_ckpt="../checkpoints/dualvitok/pytorch_model.bin",
        min_pixels=256 * 256,
        max_pixels=256 * 256,
        trainable=False
    ),
    mm_projector=dict(
        trainable=True
    )
)

training_args = dict(
    output_dir="./logdir/illume_plus_3b/illume_plus-qwen2_5-3b_debug/",
    deepspeed="./scripts/zero3.json",
    bf16=False,
    tf32=False,
    fp16=True,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,  # 256bs, 128gpu, 20k step/epoch
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=1,
    save_on_each_node=True,
    # mm_projector_lr=1e-3,
    # vision_token_lr=1e-4,
    # mm_vision_tower_lr=2e-6,
    learning_rate=2e-5,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    model_max_length=2048,
    gradient_checkpointing=True,
    dataloader_num_workers=1,
    group_by_modality_length=False,
    report_to="tensorboard",
    seed=423,
    adam_beta2=0.95,  ##
)
    