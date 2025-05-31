import os
import json
import math
import numpy as np
import imagesize
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Sampler


from utils.dist_utils import get_local_rank, get_rank, get_world_size

from .folder import read_data_file, find_images_with_pathlib


def get_image_size(entry, image_key='image'):
    """
    If the entry has 'height' and 'width', return them.
    Otherwise, use imagesize or PIL to get the image dimensions.
    """
    if 'height' in entry and 'width' in entry:
        return entry['width'], entry['height']
    path = entry[image_key]
    try:
        w, h = imagesize.get(path)
        return w, h
    except Exception:
        with Image.open(path) as img:
            return img.size  # returns (w, h)


def assign_ratio(height, width, base_sizes):
    """
    Determine the best matching group index based on the aspect ratio
    compared with each target base_size (e.g. 512/512, 640/480, etc.)
    """
    ratio = width / height
    best_index = None
    best_diff = float('inf')
    for i, (bh, bw) in enumerate(base_sizes):
        base_ratio = bw / bh
        diff = abs(ratio - base_ratio)
        if diff < best_diff:
            best_diff = diff
            best_index = i
    return best_index


# ----- Single resolution dataset: returns one complete batch per __getitem__ -----
class SingleResolutionDataset(Dataset):
    def __init__(self, entries, batch_size, transform=None):
        """
        entries: List of samples in the same resolution group. Each sample is a dict
                 containing at least 'image_path' and its 'width' and 'height'.
        batch_size: The internal batch size (i.e. number of images per batch).
        transform: Optional transform applied to each image.
        """
        self.entries = entries
        self.batch_size = batch_size
        self.transform = transform
        # Only keep samples that can form a complete batch.
        self.num_batches = len(self.entries) // self.batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch_entries = self.entries[index * self.batch_size: (index + 1) * self.batch_size]
        images = []
        for entry in batch_entries:
            image_path = entry['image']
            # Read image either from S3 or local disk
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
            # Get caption if available, otherwise use empty string.
        # If transform returns a tensor, stack the images into a batch
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        elif isinstance(images[0], dict):
            new_images = dict()
            for key in images[0]:
                if isinstance(images[0][key], torch.Tensor):
                    new_images[key] = torch.stack([image[key] for image in images])
                else:
                    new_images[key] = (image[key] for image in images)
            images = new_images
        return images, 0


class ResolutionConcatDataset(Dataset):
    def __init__(self,
                 data_path,
                 json_file=None,
                 transform=None,
                 shard_data=False,
                 global_sharding=True,
                 base_sizes=None,
                 image_key='image',
                 **kwargs):
        """
        data_path: Root directory for images.
        json_file: JSON file with image information (if provided, should contain image path, size, caption).
                   If not provided, images are scanned from data_path.
        transform: Optional image transform.
        shard_data: Whether to shard the data for distributed reading.
        global_sharding: True means global sharding among all nodes; otherwise, use local sharding.
        base_sizes: List of target sizes for resolution grouping.
        batch_size: The internal batch size. Each __getitem__ returns one complete batch.
        image_key: Key in the JSON for the image path (default 'image').
        """
        if base_sizes is None:
            base_sizes = [
                (1024, 1024), (768, 1024), (1024, 768), (512, 2048), (2048, 512), (640, 1920), (1920, 640), (768, 1536),
                (1536, 768), (768, 1152), (1152, 768)
            ]
        self.data_path = data_path
        self.transform = transform
        self.global_sharding = global_sharding
        self.shard_data = shard_data
        self.base_sizes = base_sizes

        world_size = get_world_size()
        rank = get_rank()
        local_rank = get_local_rank()

        # Load data: use JSON file if provided; otherwise, scan the data_path for images.
        if json_file:
            json_data = read_data_file(json_file)
            entries = []
            for item in json_data:
                # Image path can be absolute or relative to data_path.
                image_rel = item.get(image_key)
                image_path = image_rel if os.path.isabs(image_rel) else os.path.join(data_path, image_rel)
                entry = {'image': image_path}
                if 'height' in item and 'width' in item:
                    entry['height'] = item['height']
                    entry['width'] = item['width']
                # if 'caption' in item:
                #     entry['caption'] = item['caption']
                entries.append(entry)
        else:
            image_paths = sorted(find_images_with_pathlib(data_path))
            entries = [{'image': os.path.join(data_path, p)} for p in image_paths]

        # Ensure each entry has size information.
        for entry in entries:
            if 'height' not in entry or 'width' not in entry:
                w, h = get_image_size(entry, image_key=image_key)
                entry['width'] = w
                entry['height'] = h

        # Assign a group id (resolution group) for each entry.
        self.entries = []
        # List of group indices for each sample.
        self.group_ids = []
        for entry in entries:
            try:
                group_index = assign_ratio(entry['height'], entry['width'], base_sizes)
                self.entries.append(entry)
                self.group_ids.append(group_index)
            except Exception as e:
                pass

        self.group_to_indices = {}
        for i, group in enumerate(self.group_ids):
            self.group_to_indices.setdefault(group, []).append(i)

        # Shard the data if required.
        if shard_data:
            if global_sharding:
                self.group_to_indices.setdefault(group, []).append(i)

                chunk_size = len(entries) // world_size
                entries = entries[rank * chunk_size: (rank + 1) * chunk_size]
            else:
                # For local sharding, e.g., splitting into 8 shards per node.
                chunk_size = len(entries) // 8
                entries = entries[local_rank * chunk_size: (local_rank + 1) * chunk_size]

        # Synchronize the image paths across processes.
        self.sync_image_paths(entries)

    def sync_image_paths(self, entries):
        """
        Synchronize the number of images across processes to ensure consistency.
        """
        count = len(entries)
        gathered = [None] * get_world_size()
        torch.distributed.all_gather_object(gathered, count)

        # FIXME: it's wrong.
        if self.shard_data:
            if self.global_sharding:
                if len(set(gathered)) != 1:
                    raise ValueError(f"Inconsistent number of images across nodes: {gathered}")
            else:
                min_count = min(gathered)
                if min_count == 0:
                    raise ValueError(f"Empty data on some node: {gathered}")
                entries[:] = entries[:min_count]
        else:
            if len(set(gathered)) != 1:
                raise ValueError(f"Inconsistent number of images across nodes: {gathered}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        # Try multiple times in case of errors when reading an image.
        return self.get_data(idx)

    def get_data(self, idx):
        group = self.group_ids[idx]
        for _ in range(20):
            try:

                entry = self.entries[idx]
                image_path = entry['image']
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform[self.base_sizes[group]](image)
                # Return the sample along with its group id.

                return image
            except Exception as e:
                print(f"Error details: {e}")
                idx = np.random.choice(self.group_to_indices[group])
        raise RuntimeError("Too many bad data samples.")

class ResolutionBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True):
        """
        dataset: Can be either a single ResolutionDataset or a ConcatDataset whose
                 sub-datasets each have a 'group_ids' attribute.
        batch_size: Number of samples per batch.
        drop_last: If True, drop the last incomplete batch.

        This sampler groups indices so that each batch contains samples
        with the same resolution group (i.e. the same 'group' value).
        """
        self.batch_size = batch_size
        self.drop_last = drop_last

        # If dataset is a ConcatDataset, aggregate group_ids from each sub-dataset.
        if hasattr(dataset, "datasets"):
            all_group_ids = []
            for sub_ds in dataset.datasets:
                if not hasattr(sub_ds, "group_ids"):
                    raise ValueError("Sub-dataset does not have 'group_ids' attribute.")
                all_group_ids.extend(sub_ds.group_ids)
            self.group_ids = all_group_ids
        else:
            self.group_ids = dataset.group_ids

        # Build a mapping from group id to a list of global indices.
        self.group_to_indices = {}
        for i, group in enumerate(self.group_ids):
            self.group_to_indices.setdefault(group, []).append(i)

    def __iter__(self):
        batches = []
        # For each group, shuffle the indices and create batches.
        for group, indices in self.group_to_indices.items():
            indices = indices.copy()
            np.random.shuffle(indices)  # Shuffle indices each epoch
            num_batches = len(indices) // self.batch_size
            if self.drop_last:
                indices = indices[:num_batches * self.batch_size]
            else:
                num_batches = int(np.ceil(len(indices) / self.batch_size))
            for i in range(num_batches):
                batch = indices[i * self.batch_size: (i + 1) * self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        np.random.shuffle(batches)  # Shuffle the order of batches.
        for batch in batches:
            yield batch

    def __len__(self):
        count = 0
        for indices in self.group_to_indices.values():
            count += len(indices) // self.batch_size
        return count



def build_multi_resolution_dataset(args, transform):
    dataset_args = args
    if 'dataset' in dataset_args:
        dataset_args.pop('dataset')
    return ResolutionConcatDataset(**dataset_args, transform=transform)


# ----- Example usage -----
if __name__ == "__main__":
    # Assume images are stored under data_path, and a JSON file contains image info (path, size, caption).
    data_path = "/path/to/images"
    json_file = "annotations.json"  # If no JSON, the code scans data_path for images.
    # Define an optional transform (e.g., resizing); here, we leave it as None.
    transform = None  # You can use torchvision.transforms.Compose([...]) here.

    # Specify base sizes for grouping and internal batch size (each __getitem__ returns one complete batch).
    base_sizes = [(512, 512), (640, 480), (1024, 768)]
    batch_size = 4

    dataset = ResolutionConcatDataset(data_path, json_file=json_file, transform=transform,
                                      shard_data=True, global_sharding=True,
                                      base_sizes=base_sizes, batch_size=batch_size)
    print("Total number of batches in the final dataset:", len(dataset))

    # In the DataLoader, set batch_size=1 since each __getitem__ already returns a complete batch.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        # Because DataLoader batch_size=1, extract the internal batch
        images = batch['images'][0]  # images should have shape [batch_size, C, H, W] if transform returns tensors.
        captions = batch['captions'][0]
        print("Batch images shape:", images.shape if isinstance(images, torch.Tensor) else None)
        print("Captions:", captions)
        break
