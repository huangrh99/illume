import shutil
import torch_fidelity

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, PretrainedConfig

from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import argparse
import itertools
from einops import rearrange

from tokenizer.builder import build_vq_model
from utils.registry_utils import Config
from dataset.build import build_dataset

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

try:
    import torch_npu
    from torch_npu.npu import amp
    from torch_npu.contrib import transfer_to_npu

    print('Successful import torch_npu')
except Exception as e:
    print(e)


def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    sample_dir = sample_dir.rstrip('/')
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def main(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = build_vq_model(args.vq_model)

    if args.model_dtype == 'fp16':
        vq_model = vq_model.to(torch.float16)
        print("Convert the model dtype to float16")
    elif args.model_dtype == 'bf16':
        vq_model = vq_model.to(torch.bfloat16)
        print("Convert the model dtype to bfloat16")

    vq_model.to(device)
    vq_model.eval() # important
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    if "ema" in checkpoint and args.use_ema:  # ema
        model_weight = checkpoint["ema"]
        print("Using ema params for evaluation.")
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        model_weight = checkpoint
    msg = vq_model.load_state_dict(model_weight, strict=False)
    print(msg)
    del checkpoint

    if config.torch_dtype == 'fp16':
        torch_dtype = torch.float16
    elif config.torch_dtype == 'bf16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    config.torch_dtype = torch_dtype
    
    if args.use_sdxl_decoder:
        from tokenizer.sdxl_decoder_pipe import StableDiffusionXLDecoderPipeline
        diffusion_decoder_pipe = StableDiffusionXLDecoderPipeline.from_pretrained(
            args.sdxl_decoder_ckpt,
            torch_dtype=args.torch_dtype,
            add_watermarker=False,
            vq_config=config,
            vq_model=vq_model,
        )
        diffusion_decoder_pipe.enable_model_cpu_offload()

    sample_folder_dir = f"{args.sample_dir}/samples/"
    gt_folder_dir = f"{args.sample_dir}/gts/"
    grids_folder_dir = f"{args.sample_dir}/grids/"
    if rank == 0:
        if os.path.exists(sample_folder_dir):
            shutil.rmtree(sample_folder_dir)
        if os.path.exists(gt_folder_dir):
            shutil.rmtree(gt_folder_dir)
        if os.path.exists(grids_folder_dir):
            shutil.rmtree(grids_folder_dir)
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(gt_folder_dir, exist_ok=True)
        os.makedirs(grids_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Setup data:
    dataset = build_dataset(args.data_args.val)
    num_fid_samples = len(dataset)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.data_args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )    

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.data_args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()

    codebook_usage_pixel = set()
    codebook_usage_semantic = set()
    total_codebooks_pixel = vq_model.pixel_quantizer.codebook_size
    total_codebooks_semantic = vq_model.semantic_quantizer.codebook_size

    psnr_val_rgb = []
    ssim_val_rgb = []
    loader = tqdm(loader) if rank == 0 else loader
    total = 0
    for batch in loader:
        inputs = vq_model.get_input(batch)

        imgs = inputs['image']

        if args.data_args.val.resolution != imgs.shape[-1] or args.data_args.val.resolution != imgs.shape[-2]:
            resolution = args.data_args.val.resolution
            rgb_gts = F.interpolate(imgs, size=(resolution, resolution), mode='bicubic')
        else:
            rgb_gts = imgs

        rgb_gts = (rgb_gts.permute(0, 2, 3, 1).to("cpu").numpy() + 1.0) / 2.0 # rgb_gt value is between [0, 1]

        with torch.inference_mode() and torch.cuda.amp.autocast(dtype=torch_dtype):
            if args.use_sdxl_decoder:
                diffusion_outputs = diffusion_decoder_pipe(
                    images=inputs['image'],
                    height=args.data_args.val.resolution * 2,
                    width=args.data_args.val.resolution * 2,
                    guidance_scale=args.diffusion_cfg_value,
                    num_inference_steps=args.diffusion_steps)
                samples = diffusion_outputs.images
                indices_semantic = diffusion_outputs.indices_semantic
                indices_pixel = diffusion_outputs.indices_pixel

                samples = [np.asarray(sample.resize((args.data_args.val.resolution, args.data_args.val.resolution))) for sample in samples]
            else:
                (quant_semantic, diff_semantic, indices_semantic, target_semantic), \
                (quant_pixel, diff_pixel, indices_pixel) = vq_model.encode(**inputs)
                samples = vq_model.decode(quant_semantic, quant_pixel)

                if isinstance(samples, tuple):
                    samples = samples[1]
                if samples.ndim==5:
                    samples = rearrange(samples, 'b t c h w -> (b t) c h w')

                if args.data_args.val.resolution != samples.shape[-1] or args.data_args.val.resolution != samples.shape[-2]:
                    # print(f"Decoded samples has different resolution {samples.shape[-2:]} vs. Config's {args.data_args.image_size_eval}")
                    samples = F.interpolate(samples, size=(args.data_args.image_size_eval, args.data_args.image_size_eval), mode='bicubic')
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        codebook_usage_pixel.update(indices_pixel.cpu().view(-1).tolist())
        codebook_usage_semantic.update(indices_semantic.cpu().view(-1).tolist())

        # Save samples to disk as individual .png files
        for i, (sample, rgb_gt) in enumerate(zip(samples, rgb_gts)):
            index = i * dist.get_world_size() + rank + total

            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

            rgb_gt_img = np.clip(rgb_gt * 255, 0, 255).astype(np.uint8)
            Image.fromarray(rgb_gt_img).save(f"{gt_folder_dir}/{index:06d}.png")

            grid = np.concatenate([rgb_gt_img, sample], axis=0).astype(np.uint8)
            Image.fromarray(grid).save(f"{grids_folder_dir}/{index:06d}.png")

            # metric
            rgb_restored = sample.astype(np.float32) / 255. # rgb_restored value is between [0, 1]
            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)
            
        total += global_batch_size

    # ------------------------------------
    #       Summary
    # ------------------------------------
    # Make sure all processes have finished saving their samples
    dist.barrier()
    world_size = dist.get_world_size()
    gather_psnr_val = [None for _ in range(world_size)]
    gather_ssim_val = [None for _ in range(world_size)]
    dist.all_gather_object(gather_psnr_val, psnr_val_rgb)
    dist.all_gather_object(gather_ssim_val, ssim_val_rgb)

    codebook_usage_pixel_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(codebook_usage_pixel_list, codebook_usage_pixel)
    codebook_usage_pixel = set().union(*codebook_usage_pixel_list)

    if rank == 0:
        utilized_codebooks_pixel = len(codebook_usage_pixel)
        utilization_pixel = utilized_codebooks_pixel / total_codebooks_pixel * 100
        print(f"Utilization of Pixel Codebook: {utilization_pixel}")

        utilized_codebooks_semantic = len(codebook_usage_semantic)
        utilization_semantic = utilized_codebooks_semantic / total_codebooks_semantic * 100
        print(f"Utilization of Semantic Codebook: {utilization_semantic}")

        gather_psnr_val = list(itertools.chain(*gather_psnr_val))
        gather_ssim_val = list(itertools.chain(*gather_ssim_val))        
        psnr_val_rgb = sum(gather_psnr_val) / len(gather_psnr_val)
        ssim_val_rgb = sum(gather_ssim_val) / len(gather_ssim_val)
        print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))

        result_file = f"{args.sample_dir}/psnr_ssim_results.txt"
        print("writing results to {}".format(result_file))
        with open(result_file, 'w') as f:
            print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb), file=f)

        if not args.disable_torch_fidelity:
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=sample_folder_dir,
                input2=gt_folder_dir,
                cuda=True,
                isc=True,
                fid=True,
                kid=False,
                prc=False,
                verbose=False,
            )
            print(f"rFID: {metrics_dict}.")
            with open(result_file, 'a+') as f:
                print("rFID: {metrics_dict}.", file=f)
        else:
            npz_path = create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
            print(f"Done. Save npz file in {npz_path}")


    dist.barrier()
    dist.destroy_process_group()


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

    parser.add_argument("--vq-ckpt", type=str, help="ckpt path for vq model")
    parser.add_argument("--torch-dtype", type=str, default='fp32')
    parser.add_argument("--model-dtype", type=str, default='fp32')
    parser.add_argument("--use-ema", action='store_true')
    parser.add_argument("--use-sdxl-decoder", action='store_true')
    parser.add_argument("--sdxl-decoder-ckpt", type=str, default=None)
    parser.add_argument("--diffusion-cfg-value", type=int, default=2.0)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--disable-torch-fidelity", action='store_true')
    parser.add_argument("--verbose", action='store_true')

    args = parser.parse_args()

    config = read_config(args.config)
    config.vq_ckpt = args.vq_ckpt
    config.torch_dtype = args.torch_dtype
    config.model_dtype = args.model_dtype
    config.use_ema = args.use_ema
    config.verbose = args.verbose
    config.use_sdxl_decoder = args.use_sdxl_decoder
    config.sdxl_decoder_ckpt = args.sdxl_decoder_ckpt
    config.disable_torch_fidelity = args.disable_torch_fidelity
    config.diffusion_cfg_value = args.diffusion_cfg_value
    config.diffusion_steps = args.diffusion_steps

    main(config)
