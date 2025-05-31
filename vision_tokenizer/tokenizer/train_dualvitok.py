# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/
import random
import shutil

import torch
from utils.registry_utils import Config

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn.functional as F

import os
import time
import argparse

from glob import glob
from einops import rearrange

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad, SimpleDistributedEMA
from dataset.build import build_dataset
from tokenizer.builder import build_vq_model
from tokenizer.scheduler import AnnealingLR
from utils.sampler import ImageResolutionGroupedSampler

try:
    import torch_npu
    from torch_npu.npu import amp
    from torch_npu.contrib import transfer_to_npu

    print('Successful import torch_npu')
except Exception as e:
    print(e)


def collate_anyres(batch):
    """Non-Padding Anyres data collate_fn
    Recursively collates a batch of items that can be dicts, lists/tuples, or tensors.

    Special handling:
      - If a dict contains the key "image_gen" whose value is a tensor, pad them to the maximum
        height and width in the batch and also generate a mask. If all shapes are identical, simply
        stack and create an all-ones mask.

    Parameters:
        batch (list): A list of items (dicts, lists/tuples, or tensors).

    Returns:
        Collated batch in the same structure.
    """
    if not batch:
        return batch

    # If batch items are dictionaries, process each key recursively.
    if isinstance(batch[0], dict):
        out = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            # Special handling for "image"
            if key == "image" and isinstance(values[0], torch.Tensor):
                out.update(process_images(values))
            else:
                out[key] = collate_anyres(values)
        return out

    # If batch items are lists or tuples (but not strings), transpose and recursively collate.
    elif isinstance(batch[0], (list, tuple)) and not isinstance(batch[0], (str, bytes)):
        return [collate_anyres(list(items)) for items in zip(*batch)]

    # If batch items are tensors.
    elif isinstance(batch[0], torch.Tensor):
        shapes = [x.shape for x in batch]
        if all(s == shapes[0] for s in shapes):
            return torch.stack(batch)
        return batch  # If shapes differ and no special key is provided, return as list.

    # For other types (int, float, str, etc.), return the list.
    else:
        return batch


def pad_and_stack(tensors):
    """
    Pads a list of image tensors so that all have the same spatial dimensions (H and W).
    This function handles both:
      - 3D tensors with shape [C, H, W] and returns a mask of shape [N, 1, H, W], and
      - 4D tensors with shape [T, C, H, W] and returns a mask of shape [N, T, 1, H, W].

    If all tensor shapes are identical, the tensors are stacked and a full ones mask is returned.

    Parameters:
        tensors (list of torch.Tensor): List of image tensors.

    Returns:
        A tuple of:
            - Padded images as a tensor of shape [N, ...] where ... is either (C, max_H, max_W)
              for 3D images or (T, C, max_H, max_W) for 4D images.
            - Masks as a tensor of shape [N, ...] where ... is either (1, max_H, max_W) for 3D images
              or (T, 1, max_H, max_W) for 4D images.
    """
    tensor_dim = len(tensors[0].shape)
    shapes = [v.shape for v in tensors]
    if all(s == shapes[0] for s in shapes):
        # If all shapes are the same, simply stack and create an all-ones mask.
        if tensor_dim == 3:
            N, C, H, W = len(tensors), *tensors[0].shape
            images = torch.stack(tensors)
            mask = torch.ones((N, 1, H, W), dtype=torch.uint8, device=tensors[0].device)
            return images, mask
        elif tensor_dim == 4:
            N, T, C, H, W = len(tensors), *tensors[0].shape
            images = torch.stack(tensors)
            mask = torch.ones((N, T, 1, H, W), dtype=torch.uint8, device=tensors[0].device)
            return images, mask
        else:
            raise ValueError(f"Unsupported tensor dimension: expected 3 or 4, got {tensor_dim}")

    # If shapes differ, assume differences are only in spatial dimensions (H, W).
    if tensor_dim == 3:
        C = tensors[0].shape[0]
        max_H = max(t.shape[1] for t in tensors)
        max_W = max(t.shape[2] for t in tensors)

        padded = []
        masks = []
        for t in tensors:
            _, H, W = t.shape
            padded_tensor = torch.zeros((C, max_H, max_W), dtype=t.dtype, device=t.device)
            padded_tensor[:, :H, :W] = t
            padded.append(padded_tensor)
            mask = torch.zeros((1, max_H, max_W), dtype=torch.uint8, device=t.device)
            mask[:, :H, :W] = 1
            masks.append(mask)
        return torch.stack(padded), torch.stack(masks)

    elif tensor_dim == 4:
        # For tensors with shape [T, C, H, W], assume T is fixed and pad H and W.
        T = tensors[0].shape[0]
        C = tensors[0].shape[1]
        max_H = max(t.shape[2] for t in tensors)
        max_W = max(t.shape[3] for t in tensors)

        padded = []
        masks = []
        for t in tensors:
            _, C, H, W = t.shape  # t is (T, C, H, W)
            padded_tensor = torch.zeros((T, C, max_H, max_W), dtype=t.dtype, device=t.device)
            padded_tensor[:, :, :H, :W] = t
            padded.append(padded_tensor)
            mask = torch.zeros((T, 1, max_H, max_W), dtype=torch.uint8, device=t.device)
            mask[:, :, :H, :W] = 1
            masks.append(mask)
        return torch.stack(padded), torch.stack(masks)
    else:
        raise ValueError(f"Unsupported tensor dimension: expected 3 or 4, got {tensor_dim}")


def process_images(videos):
    """
    Resize and randomly crop videos so that all have matching spatial dimensions.
    Each video is expected to have shape [T, C, H, W].
    Returns a dict with:
      - 'image': tensor of shape [N, T, C, target_short, min_long]
    """
    # Compute the target short side across all videos
    sizes = [vid.shape[-2:] for vid in videos]
    target_short = min(min(H, W) for H, W in sizes)

    resized, new_sizes, orients = [], [], []
    for vid in videos:
        T, C, H, W = vid.shape
        scale = target_short / min(H, W)
        new_H, new_W = round(H * scale), round(W * scale)
        # Resize video using bilinear interpolation
        resized_vid = F.interpolate(vid, size=(new_H, new_W), mode='bilinear', align_corners=False)
        resized.append(resized_vid)
        new_sizes.append((new_H, new_W))
        # If height equals target_short, assume vertical; otherwise horizontal.
        orients.append('vertical' if new_H == target_short else 'horizontal')

    # Compute the minimum long side across videos and adjust to a multiple of 16
    min_long = min(max(h, w) for h, w in new_sizes)
    min_long = (min_long // 16) * 16

    outs = []
    for vid, (h, w), orient in zip(resized, new_sizes, orients):
        if orient == 'vertical':
            start = random.randint(0, w - min_long) if w > min_long else 0
            crop = vid[:, :, :, start:start + min_long]
        else:
            start = random.randint(0, h - min_long) if h > min_long else 0
            crop = vid[:, :, start:start + min_long, :]
            crop = crop.transpose(2, 3)  # Ensure final shape [T, C, target_short, min_long]
        outs.append(crop)

    out_tensor = torch.stack(outs)
    return dict(image=out_tensor)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(args)
    assert args.data_args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.vq_model.type.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{model_string_name}"  # Create an experiment folder
        # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        log_dir = f"{experiment_dir}/logs"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        writer = SummaryWriter(log_dir=log_dir)  # TensorBoard log
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = build_vq_model(args.vq_model)

    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    logger.info(f"VQ Model Trainable Parameters: {sum(p.numel() for p in vq_model.parameters() if p.requires_grad):,}")
    if args.ema:
        # ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        # requires_grad(ema, False)
        ema = SimpleDistributedEMA(vq_model, distributed=False)  # Create an EMA of the model for use after training
        logger.info(f"RANK {dist.get_rank()} VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)

    vq_loss = build_vq_model(args.vq_loss)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    for n, p in vq_model.named_parameters():
        logger.info(f'Param {n} trainable: {p.requires_grad} Shape: {p.data.shape}')

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))

    # Setup optimizer
    optimizer = torch.optim.Adam([p for p in vq_model.parameters() if p.requires_grad], lr=args.lr,
                                 betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr,
                                      betas=(args.beta1, args.beta2))

    # Setup data:
    dataset = build_dataset(args.data_args.train)

    local_batch_size = int(args.data_args.global_batch_size // dist.get_world_size())

    if args.get('use_local_data', False):
        sampler = RandomSampler(dataset)
        logger.info("Using data in local node for training.")
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )

    if args.get('use_group_image_resolution', False):
        assert args.use_local_data, 'group image resolution only support use_local_data=True'
        group_world_size = 8 if args.get('use_local_data', False) else dist.get_world_size()
        image_sizes = None
        if isinstance(dataset, ConcatDataset):
            if hasattr(dataset.datasets[0], 'image_sizes'):
                image_sizes = []
                for sub_dataset in dataset.datasets:
                    image_sizes.extend(sub_dataset.image_sizes)
        else:
            if hasattr(dataset, 'image_sizes'):
                image_sizes = dataset.image_sizes

        if image_sizes:
            sampler = ImageResolutionGroupedSampler(local_batch_size, group_world_size, image_sizes)
            logger.info("Grouping the image resolutions to accelerate the training.")
        else:
            logger.info("dataset.image_sizes is None. Cannot grouping the image resolutions.")

    if args.get('training_with_anyres', None) is None:
        collate_fn = None
    else:
        collate_fn = collate_anyres

    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=4,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_args.train})")

    args.train_iters = args.epochs * len(loader)
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.get('warmup', 0.) * args.train_iters,
        num_iters=args.train_iters,
        decay_style=args.get('lr_decay_style', 'constant'),
        last_iter=-1,
        decay_ratio=args.get('lr_decay_ratio', 0.1)
    )
    lr_scheduler_disc = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.get('warmup', 0.) * args.train_iters,
        num_iters=args.train_iters,
        decay_style=args.get('lr_decay_style', 'constant'),
        last_iter=-1,
        decay_ratio=args.get('lr_decay_ratio', 0.1)
    )

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location='cuda')
        if "model" in checkpoint:
            model_state = checkpoint["model"]
        else:
            model_state = checkpoint

        vq_model.load_state_dict(model_state, strict=True)
        logger.info("Loaded model from checkpoint.")

        if args.ema:
            if 'ema' in checkpoint:
                ema.load_state_dict(checkpoint["ema"])
            else:
                logger.info("Try to load ema parameters. But ema model is not in the CKPT. "
                            "Init the ema model with current params.")
                # update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights
                ema.update(vq_model, decay=0)  # Ensure EMA is initialized with synced weights

        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            # optimizer.to(device)
            logger.info("Optimizer starting from checkpoint.")
        except Exception as e:
            print(e)
            logger.info("Optimizer starting from scratch.")
        try:
            vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
            vq_loss.discriminator.to(device)
            logger.info("Discriminator starting from checkpoint.")
        except Exception as e:
            print(e)
            logger.info("Discriminator starting from scratch.")
        try:
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
            # optimizer_disc.to(device)
            logger.info("Discriminator optimizer starting from checkpoint.")
        except Exception as e:
            print(e)
            logger.info("Discriminator optimizer starting from scratch.")

        try:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
            # optimizer_disc.to(device)
            logger.info("Scheduler starting from checkpoint.")
        except Exception as e:
            print(e)
            logger.info("scheduler starting from scratch.")

        try:
            lr_scheduler_disc.load_state_dict(checkpoint["scheduler_disc"])
            # optimizer_disc.to(device)
            logger.info("Discriminator scheduler starting from checkpoint.")
        except Exception as e:
            print(e)
            logger.info("Discriminator scheduler starting from scratch.")

        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(
            args.vq_ckpt.split('/')[-1].split('.')[0])

        if args.get('use_local_data', False):
            start_epoch = int(train_steps / int(len(dataset) / (args.data_args.global_batch_size // dist.get_world_size())))
            train_steps = int(start_epoch * int(len(dataset) / (args.data_args.global_batch_size // dist.get_world_size())))
        else:
            start_epoch = int(train_steps / int(len(dataset) / args.data_args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.data_args.global_batch_size))

        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            # update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights
            ema.update(vq_model, decay=0)  # Ensure EMA is initialized with synced weights

    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model)  # requires PyTorch 2.0

    vq_model_without_ddp = vq_model.to(device)
    logger.info(vq_model_without_ddp)
    vq_model = DDP(vq_model_without_ddp, device_ids=[args.gpu])
    vq_model.train()
    # if args.ema:
    #     ema.eval()  # EMA model should always be in eval mode
    vq_loss_without_ddp = vq_loss.to(device)
    vq_loss = DDP(vq_loss_without_ddp, device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    if ptdtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        logger.info("Current GPU not support bf16. Switch to fp16.")
        ptdtype = torch.float16

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()
    count = 0
    # use_discriminator= args.get('use_discriminator', True)
    logger.info(f"Training for {args.epochs} epochs...")
    logger.info(f"One epoch has {len(loader)} steps...")
    for epoch in range(start_epoch, args.epochs):
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        optimizer.zero_grad()

        logger.info(f"Beginning epoch {epoch}...")
        for batch_idx, batch in enumerate(loader):
            inputs = vq_model_without_ddp.get_input(batch)

            imgs = inputs['image']

            if args.get('aug_fade_steps', 0) >= 0:
                fade_blur_schedule = 0 if train_steps < args.vq_loss.disc_start else min(1.0, (train_steps - args.vq_loss.disc_start) / (args.get('aug_fade_steps', 0) + 1))
                fade_blur_schedule = 1 - fade_blur_schedule
            else:
                fade_blur_schedule = 0

            # generator training
            with torch.cuda.amp.autocast(dtype=ptdtype):
                (recons_semantic, codebook_loss_semantic, imgs_semantic), \
                (recons_imgs, codebook_loss_detail) = vq_model(**inputs)

                if imgs.ndim == 5:
                    imgs = rearrange(imgs, 'b t c h w -> (b t) c h w')
                if recons_imgs.ndim == 5:
                    recons_imgs = rearrange(recons_imgs, 'b t c h w -> (b t) c h w')
                if imgs_semantic.ndim == 5:
                    imgs_semantic = rearrange(imgs_semantic, 'b t c h w -> (b t) c h w')
                if recons_semantic.ndim == 5:
                    recons_semantic = rearrange(recons_semantic, 'b t c h w -> (b t) c h w')

                loss_gen, loss_dict_gen = vq_loss(
                    codebook_loss_detail, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps + 1,
                    last_layer=vq_model.module.pixel_decoder.last_layer,
                    logger=logger, log_every=args.log_every,
                    fade_blur_schedule=fade_blur_schedule)

                semantic_loss, semantic_loss_dict = vq_loss_without_ddp.compute_semantic_loss(
                    codebook_loss_semantic, imgs_semantic, recons_semantic)
                loss_gen += semantic_loss

                if torch.isnan(recons_semantic).any():
                    raise RuntimeError("Meet NaN in recons_semantic.")

                if torch.isnan(semantic_loss):
                    raise RuntimeError("Meet NaN in semantic_loss.")

                if (train_steps + 1) % args.log_every == 0:
                    print_str = f"(Semantic Generator)"
                    for k, v in semantic_loss_dict.items():
                        print_str += f" {k}: {v:.4f}"
                    logger.info(print_str)

            loss_gen = loss_gen / args.gradient_accumulation_steps
            if torch.isnan(loss_gen):
                raise RuntimeError("Meet NaN during training.")
            scaler.scale(loss_gen).backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)

                lr_scheduler.step()
                scaler.step(optimizer)
                scaler.update()
                if args.ema:
                    # update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)
                    ema.update(vq_model.module._orig_mod if args.compile else vq_model.module)  # Ensure EMA is initialized with synced weights

                optimizer.zero_grad()

            # discriminator training            
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc, loss_dict_disc = vq_loss(
                    codebook_loss_detail, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps + 1,
                    logger=logger, log_every=args.log_every,
                    fade_blur_schedule=fade_blur_schedule)

            loss_disc = loss_disc / args.gradient_accumulation_steps
            scaler_disc.scale(loss_disc).backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm != 0.0:
                    scaler_disc.unscale_(optimizer_disc)
                    total_grad_norm_dis = torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(),
                                                                         args.max_grad_norm)
                lr_scheduler_disc.step()
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()

            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(epoch={epoch} step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Total grad norm: {total_grad_norm}, Total grad norm for dis: {total_grad_norm_dis}, lr: {lr_scheduler.get_lr()}")

                if rank == 0 and writer:
                    # Log the losses to TensorBoard
                    for key, value in loss_dict_gen.items():
                        writer.add_scalar(f"gen_loss/{key}", value, train_steps)
                    for key, value in semantic_loss_dict.items():
                        writer.add_scalar(f"semantic_loss/{key}", value, train_steps)
                    for key, value in loss_dict_disc.items():
                        writer.add_scalar(f"disc_loss/{key}", value, train_steps)

                    writer.add_scalar(f"train_grad_norm", total_grad_norm, train_steps)
                    grid = (torch.cat([imgs[:4], recons_imgs[:4]], dim=0) + 1) / 2
                    grid = torchvision.utils.make_grid(grid.clamp(0, 1), nrow=4)
                    writer.add_image("train/reconstructed_imgs", grid, train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:

                if args.ema:  # all gpu need to call this, in case is DistributedEMA.
                    ema_state_dict = ema.state_dict()
                else:
                    ema_state_dict = None

                if rank == 0:
                    def convert_to_cpu(data):
                        if isinstance(data, dict):
                            return {k:convert_to_cpu(v) for k, v in data.items()}
                        elif isinstance(data, list):
                            return [convert_to_cpu(v) for v in data]
                        elif isinstance(data, torch.Tensor):
                            return data.cpu()

                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()
                    checkpoint = {
                        "model": convert_to_cpu(model_weight),
                        "optimizer": convert_to_cpu(optimizer.state_dict()),
                        "discriminator": convert_to_cpu(vq_loss.module.discriminator.state_dict()),
                        "optimizer_disc": convert_to_cpu(optimizer_disc.state_dict()),
                        "scheduler": lr_scheduler.state_dict(),
                        "scheduler_disc": lr_scheduler_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema_state_dict

                    if not args.no_local_save:
                        checkpoint_filename = f"{train_steps:07d}.pt"
                        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

                        def delete_old_checkpoints(checkpoint_dir, keep_checkpoint):
                            # List all checkpoint files in the directory
                            try:
                                checkpoint_files = glob(os.path.join(checkpoint_dir, "*.pt"))

                                for file_path in checkpoint_files:
                                    # Delete files that are not the current checkpoint
                                    if os.path.basename(file_path) != keep_checkpoint:
                                        os.remove(file_path)
                                        logger.info(f"Deleted old checkpoint: {file_path}")
                            except Exception as e:
                                logger.error(f"Error deleting checkpoint {file_path}: {e}")

                        # Delete old checkpoints before saving the new one
                        delete_old_checkpoints(checkpoint_dir, checkpoint_filename)

                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                dist.barrier()
    vq_model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    dist.destroy_process_group()
    if rank == 0 and writer:
        writer.close()


def read_config(file):
    # solve config loading conflict when multi-processes
    import time
    while True:
        config = Config.fromfile(file)
        if len(config) == 0:
            time.sleep(0.1)
            continue
        break
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    args, unknown = parser.parse_known_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    config = read_config(args.config)

    overrides = []
    for item in unknown:
        if item.startswith('--'):
            item = item.strip('--')
        k, v = item.split('=')
        overrides.append((k, v))

    # Apply overrides
    for key, value in overrides:
        try:
            config.set_nested(key, value)
        except AttributeError as e:
            print(f"Warning: {e}")

    main(config)
