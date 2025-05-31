import torch
import json
import time
import datetime

from tokenizer.builder import build_vq_model
from utils.registry_utils import read_config

from vision_tokenizer.dataset.build import make_transform

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
import os
import argparse
from collections import defaultdict

import json
import os
import copy
import orjson
from PIL import Image
import torch

from torch.utils.data import Dataset
from illume.data.aspect_ratio_utils import RATIOS, AspectRatioCrop
from illume.data.data_utils import write_to_jsonl, count_lines_in_jsonl_file


try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except Exception as e:
    print("import torch_npu failed.")


def is_distributed():
    return get_world_size() > 1


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def rank0_print(content):
    if is_distributed() and dist.get_rank() == 0:
        print(content)
    elif not is_distributed():
        print(content)


def data_collate_fn(batch):
    output = {}
    for k in batch[0]:
        tmp = []
        for info in batch:
            tmp.append(info[k] if k in info else None)
        output[k] = tmp
    return output




class CodebookInfereneDataset(Dataset):
    def __init__(self, jsonl_file, image_dir, transform=None, ratios=RATIOS, crop_percent_thresh=0.2):
        super().__init__()
        self.jsonl_file = jsonl_file
        self.transform = transform
        self.image_dir = image_dir

        if jsonl_file.endswith('jsonl'):
            with open(jsonl_file, "rb") as fr:
                self.infos = [item for item in fr]
        else:  # json file
            self.infos = json.load(open(jsonl_file))

        self.arc = AspectRatioCrop(ratios, crop_percent_thresh=crop_percent_thresh)

    def add_image_info_into_data(self, info, image_sizes, image_embed_inds, matched_ratios):
        info["image_sizes"] = image_sizes
        info["image_embed_inds"] = image_embed_inds
        info["matched_ratios"] = matched_ratios
        return info

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        info = copy.deepcopy(self.infos[idx])
        if isinstance(info, bytes):
            info = orjson.loads(info)

        # added keys: images, image_sizes, need_to_skip_data
        need_to_skip_data = False
        image_paths = info["images"]
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images_data, image_sizes, matched_ratios = [], [], []
        for img_idx, image_path in enumerate(image_paths):
            imagename = os.path.join(self.image_dir, image_path)
            try:
                image = Image.open(imagename).convert('RGB')
            except Exception as e:
                print("skip data:", e)
                need_to_skip_data = True
                image = Image.new('RGB', (256, 256), (255, 255, 255))

            w, h = image.size
            image_sizes.append((h, w))

            # match image into aspect ratio types
            image, original_size, target_size, flag_matched = self.arc(image)
            if not flag_matched:
                need_to_skip_data = True

            if self.transform:
                try:
                    image = self.transform(image)["image"]
                except Exception as e:
                    print(e)
                    need_to_skip_data = True
                    image = torch.zeros(1, 3, 256, 256)

            images_data.append(image)
            matched_ratios.append(target_size)  # (h, w)

        info["need_to_skip_data"] = need_to_skip_data
        info["images_data"] = images_data
        info["image_sizes"] = image_sizes
        info["matched_ratios"] = matched_ratios

        return info


def build_codebook_inference_dataset(args, transform):
    return CodebookInfereneDataset(args.input_file, args.image_dir,
                                   transform=transform, ratios=args.ratios,
                                   crop_percent_thresh=args.crop_percent_thresh)


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    # def __iter__(self):
    #     yield from self._local_indices
    def __iter__(self):
        # Check if the current rank has data
        if len(self._local_indices) == 0:
            return iter([])
        return iter(self._local_indices)

    def __len__(self):
        return len(self._local_indices)


def inference_one_dataset(vq_model, args):
    transform = make_transform(n_px=args.data_args.inference.resolution,
                               augment=args.data_args.inference.augment)
    dataset = build_codebook_inference_dataset(args.data_args.inference, transform=transform)

    if len(dataset) == 0:
        return

    kwargs = {}
    if is_distributed():
        sampler = InferenceSampler(len(dataset))
        kwargs["sampler"] = sampler
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.data_args.inference.batch_size_for_inference,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        collate_fn=data_collate_fn,
        **kwargs
    )

    start_time = time.time()

    assert args.data_args.inference.batch_size_for_inference == 1, "only support batch size=1 now"

    outputs = defaultdict(list)
    for batch in data_loader:
        batch_image_sizes = batch.pop("image_sizes")
        batch_matched_ratios = batch.pop("matched_ratios")
        batch_need_to_skip_data = batch.pop("need_to_skip_data")
        batch_images_data = batch.pop("images_data")

        for i, need_to_skip_data in enumerate(batch_need_to_skip_data):
            if need_to_skip_data:  # filter data
                continue
            image_sizes = batch_image_sizes[i]
            matched_ratios = batch_matched_ratios[i]
            images_data = batch_images_data[i]

            if len(images_data) == 0:
                print("no images! skip")
                continue

            image_embed_inds = []
            with torch.no_grad():
                for image_data in images_data:
                    image_data = image_data.to(device=args.device, dtype=vq_model.dtype, non_blocking=True)
                    out = vq_model.encode(image_data)
                    semantic_code, texture_code = out[0][2], out[1][2]
                    # rank0_print(f"texture_code, {semantic_code.shape}")
                    # rank0_print(f"semantic_code, {texture_code.shape}")
                    semantic_code = semantic_code.cpu().tolist()
                    texture_code = texture_code.cpu().tolist()
                    image_embed_inds.append([semantic_code[0], texture_code[0]])

            tmp = {}
            for k, v in batch.items():
                tmp[k] = v[i]
            dataset.add_image_info_into_data(tmp, image_sizes, image_embed_inds, matched_ratios)

            # split into different ratios and save in different files
            if all(ratio == matched_ratios[0] for ratio in matched_ratios):
                ratio_type = f"ratio_h{matched_ratios[0][0]}_w{matched_ratios[0][1]}"
            else:
                ratio_type = "ratio_mixed"

            outputs[ratio_type].append(tmp)

    if is_distributed():
        world_size = dist.get_world_size()
        outputs_merged = [None for _ in range(world_size)]
        dist.all_gather_object(outputs_merged, outputs)

        if dist.get_rank() == 0:
            final_outputs = defaultdict(list)
            for d in outputs_merged:
                for key, value_list in d.items():
                    final_outputs[key].extend(value_list)

        dist.barrier()
    else:
        final_outputs = outputs

    if not is_distributed() or (is_distributed() and dist.get_rank() == 0):
        results = []
        statistic = {}
        for ratio_type, output in final_outputs.items():
            results.extend(output)
            statistic[ratio_type] = len(output)

        filename = os.path.basename(args.data_args.inference.input_file)
        filename = filename if filename.endswith('.jsonl') else filename.replace('.json', '.jsonl')
        out_file = os.path.join(args.data_args.inference.output_dir, filename)
        write_to_jsonl(results, out_file)

        rank0_print(f"statistic: {statistic}")
        with open(args.data_args.inference.statistic_file, 'w', encoding='utf-8') as f_w:
            f_w.write(json.dumps(statistic, ensure_ascii=False))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    rank0_print(f'inference time {total_time_str}')

    dist.barrier()


def main(args):
    torch.set_grad_enabled(False)

    # Setup DDP:
    if int(os.getenv("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        args.rank = dist.get_rank()
        args.device = args.rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + args.rank
        torch.manual_seed(seed)
        torch.cuda.set_device(args.device)
        print(f"Starting rank={args.rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        args.rank = int(os.getenv('RANK', '0'))
        args.device = "cuda"

    # build model
    rank0_print("build vq model")
    vq_model = build_vq_model(args.vq_model)
    torch_type_mapping = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
    torch_dtype = args.torch_dtype if args.torch_dtype is not None else args.mixed_precision
    torch_dtype = torch_type_mapping[torch_dtype]
    vq_model = vq_model.to(torch_dtype)
    vq_model.to(args.device)
    vq_model.eval()  # important
    if args.vq_ckpt is not None:
        print(f'Loading tokenizer checkpoint from {args.vq_ckpt}')
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        if "ema" in checkpoint and args.use_ema:  # ema
            model_weight = checkpoint["ema"]
            print("load ema")
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            model_weight = checkpoint
        msg = vq_model.load_state_dict(model_weight, strict=False)
        print(msg)
        del checkpoint

    if is_distributed():
        dist.barrier()

    meta_infos = json.load(open(args.data_config, "r"))
    for dataset_name, dataset_info in meta_infos.items():
        if not dataset_info["tokenizer_inference"]:
            continue

        jsonl_dir = dataset_info["annotation_dir"]
        if args.resolution_type == "fixed":
            ratios = [(args.resolution, args.resolution)]
            aspect_ratio_version = f"{args.resolution_type}_{args.resolution}"
        elif args.resolution_type == "fixed_anchors":
            ratios = RATIOS
            aspect_ratio_version = args.resolution_type
        else:
            raise NotImplementedError

        filelist = [os.path.join(jsonl_dir, file) for file in os.listdir(jsonl_dir) if file.endswith((".jsonl", 'json'))]

        output_dir = os.path.join(dataset_info["root"], args.output_dirname, args.tokenizer_version, aspect_ratio_version)
        os.makedirs(output_dir, exist_ok=True)
        rank0_print(f"dataset name: {dataset_name}, file len: {len(filelist)}")

        for i, input_file in enumerate(filelist):
            filename, file_dir = os.path.basename(input_file), os.path.dirname(input_file)

            # skip existed files
            if args.resume and any(os.path.exists(os.path.join(output_dir, dir, os.path.basename(filename))) for dir in
                                   os.listdir(output_dir)):
                continue

            rank0_print(f"dataset name: {dataset_name}, {i} file {filename}")
            args.data_args.inference.input_file = input_file
            args.data_args.inference.output_dir = output_dir
            args.data_args.inference.statistic_file = os.path.join(dataset_info["root"], args.output_dirname, f"{aspect_ratio_version}_data_statistic.json")
            args.data_args.inference.image_dir = dataset_info["image_dir"]
            args.data_args.inference.ratios = ratios
            args.data_args.inference.crop_percent_thresh = args.crop_percent_thresh
            inference_one_dataset(vq_model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str,
                        default="../configs/example/dualvitok/dualvitok_stage3_anyres_max512.py", required=True)
    parser.add_argument("--tokenizer_version", type=str, default="dualvitok")
    parser.add_argument("--tokenizer_checkpoint", type=str, required=True)
    parser.add_argument("--output_dirname", type=str, default="jsonl_after_tokenizer")
    parser.add_argument("--data_config", type=str,
                        default='../configs/data_configs/train_data_examples/examples_meta_data_config.json')
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--resolution_type", type=str, default="fixed_anchors")  # fixed, fixed_anchors
    parser.add_argument("--crop_percent_thresh", type=float, default=0.2)
    parser.add_argument("--torch_dtype", type=str, default='fp32')
    parser.add_argument("--use-ema", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    config = read_config(args.model_config)
    config.output_dirname = args.output_dirname
    config.resolution_type = args.resolution_type
    config.resolution = args.resolution
    config.tokenizer_version = args.tokenizer_version
    config.vq_ckpt = args.tokenizer_checkpoint
    config.data_config = args.data_config
    config.torch_dtype = args.torch_dtype
    config.use_ema = args.use_ema
    config.resume = args.resume
    config.crop_percent_thresh = args.crop_percent_thresh
    main(config)
