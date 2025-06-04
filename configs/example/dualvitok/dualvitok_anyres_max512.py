vq_model = dict(
    type='DualViTok',
    config=dict(
        semantic_encoder=dict(
            semantic_encoder='Emova-ollm/qwen2vit600m',
            num_blocks=4,
            proj_layer='linear',
            z_channels=32,

            embed_dim=1280,
            attn_implementation='sdpa',
            target_mlp='norm',
        ),
        semantic_quantizer=dict(
            type='SimVQ',
            dim=32,
            codebook_size=32768,
            channel_first=True,
            rotation_trick=False,
        ),
        semantic_decoder=dict(
            z_channels=32,
            proj_layer='linear_norm',
            embed_dim=1280,
            output_channels=3584,
            num_blocks=4,
            attn_implementation='sdpa',
        ),

        pixel_encoder=dict(
            ch=128,
            attn_resolutions=[4],
            ch_mult=[1, 1, 2, 2, 4],
            codebook_size=32768 * 3,
            double_z=False,
            dropout=0.0,
            embed_dim=32,
            in_channels=3,
            num_res_blocks=2,
            out_channels=3,
            z_channels=32,
            use_dc_up_down_blocks=True,
        ),
        pixel_quantizer=dict(
            type='SimVQ',
            dim=32,
            codebook_size=32768 * 3,
            channel_first=True,
            rotation_trick=False,
        ),
        pixel_decoder=dict(
            ch=384,
            attn_resolutions=[4],
            ch_mult=[1, 1, 2, 2, 4],
            codebook_size=32768 * 3,
            double_z=False,
            dropout=0.0,
            embed_dim=64,
            in_channels=3,
            num_res_blocks=2,
            out_channels=3,
            z_channels=64,
            use_dc_up_down_blocks=True,
        ),

        pixel_drop_rate=None,
        pixel_drop_batch_rate=0.1,
        pixel_noise_type='random',
    )
)

vq_loss = dict(
    type='DualViTokLoss',
    disc_start=20000,
    disc_loss="hinge",
    disc_dim=64,
    disc_type='patchgan',
    image_size=512,
    disc_num_layers=3,
    disc_in_channels=3,
    disc_weight=0.5,
    disc_adaptive_weight=False,
    gen_adv_loss='hinge',
    reconstruction_loss='l2',
    reconstruction_weight=1.0,
    codebook_weight=1.0,
    perceptual_weight=1.0,
    semantic_reconstruction='cos',
)

data_args = dict(
    global_batch_size=256,
    image_size=512,
    image_size_eval=256,

    train=dict(
        resolution=512,
        augment=dict(type='dualvitok_anyres_resolution_train',
                     min_pixels=32 * 32, max_pixels=512 * 512),
        dataset=[
            # imagenet
            dict(dataset='folder',
                 data_path='./data/imagenet_train/',
                 # specific this for the image resolution bucketing.
                 json_file='./data/json_files/imagenet_train.json',
                 # generate by `vision_tokenizer/scripts/read_folder_image_sizes.py`
                 shard_data=True, global_sharding=True),

            # Define other datasets here
            # dict(dataset='folder',
            #      data_path='path/to/your/dataset/',
            #      json_file='path/to/your/json_file.json',
            #      shard_data=True, global_sharding=True),
        ],
    ),

    val=dict(
        resolution=256,
        dataset='imagenet',
        data_path='./data/imagenet_val/',
        augment=dict(type='dualvitok_fix_resolution_val'),
    ),

    # For codebook inference.
    inference=dict(
        resolution=256,
        dataset='codebook_inference',
        data_path='',
        augment=dict(type='dualvitok_anyres_inference'),
        batch_size_for_inference=1,
    ),

    per_proc_batch_size=2,
    infer_interpolate=True
)

# =======================
# General settings
# =======================
no_local_save = False  # Do not save checkpoints to local path
ema = False  # Whether to use EMA training
finetune_decoder = False
finetune = False

vq_ckpt = None
aug_fade_steps = 0

# =======================
# Training settings
# =======================
compile = False  # Whether to compile the model (PyTorch 2.0)
results_dir = './logdir/dualvitok/dualvitok_anyres_max512/'  # Directory for results
sample_dir = f'{results_dir}sample_dir/'
epochs = 21  # Number of epochs
lr = 1e-4  # Learning rate
weight_decay = 5e-2  # Weight decay
beta1 = 0.9  # Adam optimizer beta1
beta2 = 0.95  # Adam optimizer beta2
max_grad_norm = 1.0  # Max gradient norm
global_seed = 3  # Global random seed
num_workers = 8  # Number of data loading workers
log_every = 100  # Log every x iterations
ckpt_every = 2000  # Save checkpoint every x iterations
gradient_accumulation_steps = 1  # Gradient accumulation steps
mixed_precision = 'bf16'  # Mixed precision training ('none', 'fp16', 'bf16')

use_local_data = True
training_with_anyres = True
use_group_image_resolution = True
