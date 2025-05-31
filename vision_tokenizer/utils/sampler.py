import torch
from torch.utils.data import Sampler
from typing import List, Optional, Tuple


def get_resolution_grouped_indices_by_sorting(
        resolutions: List[Tuple[int, ...]],
        batch_size: int,
        world_size: int,
        generator=None
) -> List[int]:
    """
    Group image indices by sorting based on image resolution.

    Instead of using a single metric (like min(H, W)), this function sorts
    the indices lexicographically by (H, W) so that images with similar heights
    and widths (and thus similar aspect ratios) are adjacent. Then it partitions
    the sorted list into megabatches and shuffles the megabatches for randomness.

    Args:
        resolutions (List[Tuple[int, ...]]): A list of image shape tuples. Each tuple
            is either (H, W) or (C, H, W) (in which case the last two dimensions are used).
        batch_size (int): Per-device batch size.
        world_size (int): Number of devices.
        generator: Optional torch.Generator for random shuffling.

    Returns:
        List[int]: The final list of grouped indices.
    """
    # Create a list of indices.
    indices = list(range(len(resolutions)))

    # Define a sorting key: extract (H, W) from the resolution.
    def sort_key(i):
        shape = resolutions[i]
        # If the shape has three dimensions (e.g., (C, H, W)), use the last two.
        if len(shape) == 2:
            H, W = shape
        else:
            H, W = shape[-2:]
        return (H, W)

    # Sort indices based on (H, W) in ascending order.
    sorted_indices = sorted(indices, key=sort_key)

    # Partition sorted indices into megabatches of size world_size * batch_size.
    megabatch_size = world_size * batch_size
    megabatches = [sorted_indices[i: i + megabatch_size] for i in range(0, len(sorted_indices), megabatch_size)]

    # Optionally, shuffle the megabatch order for randomness.
    if generator is None:
        perm = torch.randperm(len(megabatches)).tolist()
    else:
        perm = torch.randperm(len(megabatches), generator=generator).tolist()
    megabatches = [megabatches[i] for i in perm]

    # Flatten the megabatches to get the final order.
    grouped_indices = [idx for mb in megabatches for idx in mb]
    return grouped_indices


class ImageResolutionGroupedSampler(Sampler):
    r"""
    Sampler that groups image indices such that images with similar resolutions and aspect ratios
    (as determined by sorting on height and width) are batched together.

    Args:
        batch_size (int): Per-device batch size.
        world_size (int): Number of devices.
        resolutions (Optional[List[Tuple[int, ...]]]): List of image shape tuples, e.g. (H, W) or (C, H, W).
        generator: Optional random generator.
    """

    def __init__(
            self,
            batch_size: int,
            world_size: int,
            resolutions: Optional[List[Tuple[int, ...]]] = None,
            generator=None,
    ):
        if resolutions is None:
            raise ValueError("Resolutions must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.resolutions = resolutions
        self.generator = generator

    def __len__(self):
        return len(self.resolutions)

    def __iter__(self):
        indices = get_resolution_grouped_indices_by_sorting(
            self.resolutions, self.batch_size, self.world_size, generator=self.generator
        )
        return iter(indices)


# ----------------------------
# Example to verify the sampler
# ----------------------------
if __name__ == '__main__':
    # Define a list of image resolutions.
    # Each resolution is a tuple. For simplicity, we use (H, W).
    resolutions = [
        (240, 320),  # Aspect ratio 0.75
        (256, 256),  # Aspect ratio 1.0
        (480, 640),  # Aspect ratio 0.75
        (300, 400),  # Aspect ratio 0.75
        (200, 200),  # Aspect ratio 1.0
        (360, 480),  # Aspect ratio 0.75
        (128, 128),  # Aspect ratio 1.0
        (512, 512),  # Aspect ratio 1.0
        (350, 300),  # Aspect ratio ~1.17
        (400, 600),  # Aspect ratio ~0.67
    ]

    batch_size = 2  # Per-device batch size.
    world_size = 1  # Single device for this example.

    sampler = ImageResolutionGroupedSampler(batch_size, world_size, resolutions)
    grouped_indices = list(iter(sampler))

    print("Grouped indices:", grouped_indices)
    print("\nIndex - Resolution (H, W)")
    for idx in grouped_indices:
        print(f"{idx:2d}  -  {resolutions[idx]}")
