model_args = dict(
    version="qwen2",
    freeze_backbone=False,

    pretrain_mm_mlp_adapter=None,
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,

    mm_patch_merge_type='flat',

    special_tokens_ids=[151665, 151666, 151667, 151668, 151669, 151670, 151671],

    language_model=dict(
        type='IllumeQwen2ForCausalLM',
        pretrained_model_name_or_path='../checkpoints/Qwen2.5-3B-Instruct-with-vision-tokenizer-32k-96k-level2',
        attn_implementation="sdpa",
        from_pretrained=True,
    ),
    mm_vision_tower=dict(
        type='DualVisionTower',
        vq_config="../configs/example/dualvitok/dualvitok_anyres_max512.py",
        vq_ckpt="../checkpoints/dualvitok/pytorch_model.bin",
        min_pixels=256 * 256,
        max_pixels=256 * 256,
        use_ema=False,
        trainable=False
    ),
    mm_projector=dict(
        type='MixedProjector',
        projector_cfg1=dict(
            type='MLPProjector',
            mlp_depth=2,
        ),
        projector_cfg2=dict(
            type='MLPProjector',
            mlp_depth=2,
        ),
        trainable=True
    )
)
