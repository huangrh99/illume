from torchvision import transforms
import numpy as np
import math
import torch
from illume.data.data_utils import center_crop_and_resize, resize_with_padding, unpad_and_resize_back

RATIOS = [
    (512, 512),
    (384, 512),
    (512, 384),
    (384, 768),
    (768, 384),
    (384, 576),
    (576, 384),
    (320, 960),
    (960, 320),
    (256, 1024),
    (1024, 256),
]

RATIO_TYPES = [
    'ratio_h512_w512',
    'ratio_h384_w512',
    'ratio_h512_w384',
    'ratio_h384_w768',
    'ratio_h768_w384',
    'ratio_h384_w576',
    'ratio_h576_w384',
    'ratio_h320_w960',
    'ratio_h960_w320',
    'ratio_h256_w1024',
    'ratio_h1024_w256',
]

def calculate_ratio():
    max_area = 512 * 512
    ratios = [(2, 2), (3, 4), (4, 3), (2, 4), (4, 2), (1, 4), (4, 1), (2, 3), (3, 2), (1, 3), (3, 1)]
    ratio_candicates = []
    for ratio in ratios:
        x = math.sqrt(max_area / ratio[0] / ratio[1])
        x = round(x / 64) * 64
        tmp = (x*ratio[0], x*ratio[1])
        # print(ratio, x, tmp)
        ratio_candicates.append(tmp)

    print("ratio_candicates", ratio_candicates)
    return ratio_candicates


class AspectRatioCrop(object):
    """
    Aspect Ratio Crop transform.
    For a given image, find the corresponding aspect ratio and
    resize / resize + crop to the corresponding base sizes

    Args:
        base_sizes: list[tuple], the base sizes of final output.
            For example, [(512, 512), (512, 768), (768, 512)]

        resize_and_crop: bool .If False, find the matched aspect ratio and resize to base size.
    """

    def __init__(self, base_sizes, crop_percent_thresh=0.2):
        self.base_sizes = [(math.floor(h), math.floor(w)) for (h, w) in base_sizes]
        self.aspect_ratios = [x[1] / x[0] for x in self.base_sizes]  # w / h
        self.crop_percent_thresh = crop_percent_thresh

    def _find_size(self, w, h):
        base_size_indexes = list(range(len(self.base_sizes)))
        aspect_ratios = [self.aspect_ratios[i] for i in base_size_indexes]
        aspect_ratio = w / h
        ratio_diff = [abs(ratio - aspect_ratio) for ratio in aspect_ratios]
        min_diff = np.min(ratio_diff)
        match_diff_indexes = [j for j in range(len(ratio_diff)) if ratio_diff[j] == min_diff]
        match_diff_indexes = sorted(match_diff_indexes, key=lambda x: (h-self.base_sizes[base_size_indexes[x]][0])**2
                                                                    + (w-self.base_sizes[base_size_indexes[x]][1])**2) # pick the area most match one
        corr_index = base_size_indexes[match_diff_indexes[0]]
        return corr_index

    def get_pred_target_w_h(self, w, h):
        aspect_ratio = w / h
        aspect_index = self._find_size(w, h)
        pred_h, pred_w = self.base_sizes[aspect_index]

        solutions = [
            (pred_w, int(pred_w / aspect_ratio)),
            (int(pred_h * aspect_ratio), pred_h),
        ]
        w_tar = None
        h_tar = None
        for solution in solutions:
            w_s, h_s = solution
            if w_s >= pred_w and h_s >= pred_h:
                w_tar = w_s
                h_tar = h_s

        return pred_w, pred_h, w_tar, h_tar, aspect_index

    def __call__(self, image, is_inference=False):
        ## step 1: find the cloest aspect ratios
        flag_matched = True
        w, h = image.size
        pred_w, pred_h, w_tar, h_tar, aspect_index = self.get_pred_target_w_h(w, h)

        crop_percent = 1 - pred_w * pred_h / (w_tar * h_tar)
        if self.crop_percent_thresh > 0 and crop_percent > self.crop_percent_thresh:
            flag_matched = False  # filter data

        if not is_inference:
            ## step 2: train: crop and resize
            image = center_crop_and_resize(image, output_size=(pred_h, pred_w))
        else:
            ## step 2: inference: resize and padding
            image = resize_with_padding(image, output_size=(pred_h, pred_w))

        original_size = [h, w]
        target_size = [pred_h, pred_w]

        return image, original_size, target_size, flag_matched


if __name__ == "__main__":
    # ratio_candicates = calculate_ratio()
    # v2: [(512, 512), (384, 512), (512, 384), (384, 768), (768, 384), (256, 1024), (1024, 256), (384, 576), (576, 384)] # h, w
    RATIOS = [(512, 1024)]
    arc = AspectRatioCrop(RATIOS, crop_percent_thresh=0.2)
    image_path = "../configs/data_examples/train_data_examples/examples_llava_format/images/0.png"
    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    image_size = image.size  # h, w
    print("image size", image.size)

    image, original_size, target_size, flag_matched = arc(image, is_inference=True)
    print("original_size, target_size, flag_matched", original_size, target_size, flag_matched)
    output_file = "/home/ma-user/work/wangchunwei/ILLUME_plus/debug_ratio.png"
    image.save(output_file)

    image2 = unpad_and_resize_back(image, image_size[0], image_size[1])
    output_file = "/home/ma-user/work/wangchunwei/ILLUME_plus/debug_ratio_2.png"
    image2.save(output_file)
    print("image2", image2.size)

