import io
import os
import json
import time

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from pathlib import Path

from utils.dist_utils import get_local_rank, get_rank, get_world_size

try:
    import moxing as mox
    is_on_cloud = True
except Exception as e:
    is_on_cloud = False


def find_images_with_pathlib(root_dir):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    root_path = Path(root_dir)

    return [str(p) for p in root_path.rglob('*')
            if p.suffix.lower() in image_extensions and not p.name.startswith('.')]


def read_data_file(file):
    if not os.path.exists(file):
        print(f"{file} not exist!!!")
        return []
    if file.endswith('.json'):
        with open(file, 'r') as f:
            list_data_dict = json.load(f)
    elif file.endswith('.jsonl'):
        with open(file, "r") as fr:  # jsonl文件改用二进制读取-速度更快
            return [item for item in fr]
    else:
        raise RuntimeError(f"Unrecoginized file: {file}")
    return list_data_dict


class DatasetFolder(Dataset):
    def __init__(self, data_path, json_file=None, transform=None, shard_data=False, global_sharding=True, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.global_sharding = global_sharding  # New flag for controlling sharding behavior
        self.shard_data = shard_data

        world_size = get_world_size()
        rank = get_rank()
        local_rank = get_local_rank()

        if json_file:
            json_data = read_data_file(json_file)
            image_paths = [ item['image'] for item in json_data ]
            self.image_sizes = [ (item['height'], item['width']) for item in json_data ]
        else:
            # Get all image paths and sort them
            image_paths = sorted(find_images_with_pathlib(data_path))
            self.image_sizes = None
        self.image_paths = image_paths
        self.sync_image_paths(image_paths)

        if shard_data:
            if global_sharding:
                # Global sharding: split the dataset globally among all nodes
                chunk_size = len(self.image_paths) // world_size
                self.image_paths = self.image_paths[rank * chunk_size: (rank + 1) * chunk_size]
                if self.image_sizes:
                    self.image_sizes = self.image_sizes[rank * chunk_size: (rank + 1) * chunk_size]
            else:
                # Local sharding: each node gets its own local subset of data
                chunk_size = len(self.image_paths) // 8
                self.image_paths = self.image_paths[local_rank * chunk_size: (local_rank + 1) * chunk_size]
                if self.image_sizes:
                    self.image_sizes = self.image_sizes[local_rank * chunk_size: (local_rank + 1) * chunk_size]

        print(f"Dataset: Folder '{data_path}' has {len(self.image_paths)} images on rank {rank}. "
              f"Global sharding: {global_sharding}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {e}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def getdata(self, idx):
        image_path = self.image_paths[idx]
        image_path_full = os.path.join(self.data_path, image_path)
        image = Image.open(image_path_full).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

    def sync_image_paths(self, image_paths):
        """
        Check the size of the image path if its sizes are consistenct.
        """
        gathered_sizes = [None] * get_world_size()
        torch.distributed.all_gather_object(gathered_sizes, len(image_paths))

        if self.shard_data:
            if self.global_sharding:
                # Check if all nodes have the same set of image paths
                if len(set(gathered_sizes)) != 1:
                    raise ValueError(f"Inconsistent number of image paths across nodes. ", gathered_sizes, self.data_path)
            else:
                # Check if all nodes have the same set of image paths
                if len(set(gathered_sizes)) != 1:
                    print('The local sharding is not the same length. Cut into the shorest length', self.data_path, gathered_sizes)
                else:
                    if min(gathered_sizes) == 0:
                        raise ValueError(f"Meet empty folder in {self.data_path}", gathered_sizes,)
                self.image_paths = self.image_paths[:min(gathered_sizes)]
                if self.image_sizes:
                    self.image_sizes = self.image_sizes[:min(gathered_sizes)]
        else:
            # Check if all nodes have the same set of image paths
            if len(set(gathered_sizes)) != 1:
                raise ValueError(f"Inconsistent number of image paths across nodes. ", gathered_sizes, self.data_path)


def build_folder(args, transform):
    dataset_args = args
    if 'dataset' in dataset_args:
        dataset_args.pop('dataset')
    return DatasetFolder(**dataset_args, transform=transform)
