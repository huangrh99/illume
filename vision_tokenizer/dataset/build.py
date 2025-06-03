import copy
import os
import random

from PIL import Image
import torch
from torch.utils.data import ConcatDataset

from dataset.imagenet import build_imagenet
from dataset.openimage import build_openimage
from dataset.folder import build_folder
from dataset.multi_ratio_dataset import build_multi_resolution_dataset

from torchvision import transforms
from torchvision.transforms.functional import crop

from tokenizer.movqgan.image_processing_movqgan import MoVQImageProcessor


class WrapDQVAEImageProcessor:
    def __init__(self, **kwargs):
        kwargs.update(factor=16)
        self._tf = MoVQImageProcessor(**kwargs)

    def __call__(self, image, return_tensors='pt', **kwargs):
        output = self._tf(image, return_tensors=return_tensors)
        return dict(
            image=output['pixel_values'],
        )

class centercrop_and_resize:
    def __init__(self,
                 size):
        self.size = size
    def __call__(self, img):
        # 获取原始尺寸
        target_h, target_w = self.size
        img_w, img_h = img.size


        scale_w, scale_h = img_w / target_w, img_h / target_h
        if scale_h > scale_w:
            new_w, new_h = target_w, int(target_w / img_w * img_h)
        else:
            new_w, new_h = int(target_h / img_h * img_w), target_h

        # Resize 图片，保持长宽比
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # 计算中心裁剪区域
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h

        # 裁剪多余部分
        img = img.crop((left, top, right, bottom))

        return img

class CustomDiffusionTrainTransform:
    def __init__(self, resolution, center_crop=True, random_flip=True):
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.train_resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_crop = transforms.CenterCrop(resolution) if center_crop else \
            transforms.RandomCrop(resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1)

        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __call__(self, image):
        image = image.convert("RGB")
        # image aug
        original_size = (image.height, image.width)
        image = self.train_resize(image)
        if self.random_flip and random.random() < 0.5:
            # flip
            image = self.train_flip(image)
        if self.center_crop:
            y1 = max(0, int(round((image.height - self.resolution[0]) / 2.0)))
            x1 = max(0, int(round((image.width - self.resolution[1]) / 2.0)))
            image = self.train_crop(image)
        else:
            y1, x1, h, w = self.train_crop.get_params(image, self.resolution)
            image = crop(image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        image = self.train_transforms(image)

        return dict(pixel_values=image,
                    original_sizes=original_size,
                    crop_top_lefts=crop_top_left)


def make_transform(n_px, augment=None):
    if augment:
        augment = copy.deepcopy(augment)
        if augment.type == "imagenet_train":
            transform = transforms.Compose([
                transforms.Resize(n_px),
                transforms.RandomCrop(n_px),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif augment.type == "imagenet_val":
            transform = transforms.Compose([
                transforms.Resize(n_px),
                transforms.CenterCrop(n_px),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        elif augment.type == "imagenet_clip_train":
            transform = transforms.Compose([
                transforms.Resize(n_px),
                transforms.RandomCrop(n_px),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
            ])
        elif augment.type == 'dualvitok_fix_resolution_train':
            augment.pop('type')
            transform = transforms.Compose([
                transforms.Resize(n_px),
                transforms.CenterCrop(n_px),
                transforms.Resize((n_px, n_px)),
                transforms.RandomHorizontalFlip(p=0.5),
                WrapDQVAEImageProcessor(**augment),
            ])
        elif augment.type == 'dualvitok_anyres_resolution_train':
            augment.pop('type')
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                WrapDQVAEImageProcessor(**augment),
            ])
        elif augment.type == 'dualvitok_fix_resolution_inference':
            augment.pop('type')
            h, w = n_px
            transform = transforms.Compose([
                centercrop_and_resize((h, w)),
                WrapDQVAEImageProcessor(**augment),
            ])
        elif augment.type == 'dualvitok_anyres_inference':
            augment.pop('type')
            transform = transforms.Compose([
                WrapDQVAEImageProcessor(**augment),
            ])
        elif augment.type == 'dualvitok_fix_resolution_val':
            augment.pop('type')
            transform = transforms.Compose([
                transforms.Resize(n_px),
                transforms.CenterCrop(n_px),
                WrapDQVAEImageProcessor(**augment),
            ])
        elif augment.type == 'multi_resolution_random_crop_flip':
            assert isinstance(n_px, list)
            transform = dict()
            for resolution in n_px:
                transform[resolution] = CustomDiffusionTrainTransform(resolution)
        else:
            transform = transforms.Compose([
                transforms.Resize(n_px),
                transforms.CenterCrop(n_px),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize(n_px),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    return transform


SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))
OVERFIT_TEST = bool(os.environ.get("OVERFIT_TEST", 0))


def get_one_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        dataset = build_imagenet(args, **kwargs)
    elif args.dataset == 'openimage':
        dataset = build_openimage(args, **kwargs)
    elif args.dataset == 'folder':
        dataset = build_folder(args, **kwargs)
    elif args.dataset == 'multires':
        dataset = build_multi_resolution_dataset(args, **kwargs)
    else:
        raise ValueError(f'dataset {args.dataset} is not supported')
    return dataset


def build_dataset(args, **kwargs):
    transform = make_transform(n_px=args.resolution,
                               augment=args.augment)
    kwargs['transform'] = transform

    if isinstance(args.dataset, list):
        datasets = []
        for dataset in args.dataset:
            datasets.append(get_one_dataset(dataset, **kwargs))
        dataset = ConcatDataset(datasets)
    else:
        dataset = get_one_dataset(args, **kwargs)

    if SMOKE_TEST:
        dataset_len = 256 + 10
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:dataset_len])

    return dataset


if __name__ == '__main__':
    tf = WrapDQVAEImageProcessor()
    print(tf._tf)