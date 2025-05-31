import copy
import json
import orjson
import os
import re
import random
from dataclasses import dataclass
import tarfile
from collections import defaultdict
import torch
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from illume.utils import rank0_print

from .preprocess import *
from ..constants import IGNORE_INDEX
from ..mm_utils import expand2square, process_anyres_image

from .data_utils import ROLE_TEMPLATES, read_data_file, encode_image_token_into_code, center_crop_and_resize, return_all_files_in_dir


class DefaultDataset(Dataset):
    def __init__(self, meta_info: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 ):
        super(DefaultDataset, self).__init__()

        self.cfg_text_dropout_probability = getattr(data_args, "cfg_text_dropout_probability", 0.)

        self.data_args = data_args
        list_data_dict, total_num = self.load_data(meta_info)

        assert total_num, f"Empty dataset in {self.data_args}"

        rank0_print(f"{meta_info['dataset_name']}, data length", len(list_data_dict), "total num", total_num)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def load_data(self, meta_info):
        self.image_folder = meta_info["image_dir"] if "image_dir" in meta_info else ""
        self.dataset_name = meta_info["dataset_name"]

        if meta_info["codebook_dir"] is not None and meta_info["tokenizer_inference"]:  # for dataset include image generation
            dir = os.path.join(meta_info["root"], meta_info["codebook_dir"])
            self.is_gen_task = True
        else:  # for text-only and understanding only dataset
            dir = meta_info["annotation_dir"]
            self.is_gen_task = False

        filelist = return_all_files_in_dir(dir)
        files = sorted([f for f in list(filter(lambda x: x.endswith(("jsonl", "json")), filelist))])
        assert len(files), f"No files in {dir}."
        sample_num = meta_info["sample_num"]
        infos, total_num = self._load_data(files, sample_num)
        return infos, total_num

    def _load_data(self, files, sample_num):
        total_num = 0
        infos = []
        for i, file in enumerate(files):
            cur_infos = read_data_file(file, load_with_bytes=True)
            total_num += len(cur_infos)
            infos.extend(cur_infos)

        if sample_num > 0:
            infos = infos[:sample_num]
            total_num = sample_num

        return infos, total_num

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        return [1] * len(self.list_data_dict)  # not support yet

    @property
    def modality_lengths(self):
        return [1] * len(self.list_data_dict)  # not support yet

    def get_ratio_tag_from_ratio(self, ratio):
        h, w = ratio
        ratio_tag = f"<height_{h}><width_{w}>"
        return ratio_tag

    def get_unconditional_sample(self, info):
        return info

    def replace_image_tag_with_vision_token(self, info, image_tag="<image>"):
        image_embed_inds = info["image_embed_inds"]
        matched_ratios = info["matched_ratios"]
        image_cnt = 0
        images = []
        for conv in info["conversations"]:
            text_list = conv["value"].split(image_tag)
            image_num = len(text_list) - 1
            text = conv["value"]

            # FIXME: may get wrong if multiple {resolution_tag} in prompt
            text = text.format(resolution_tag=self.get_ratio_tag_from_ratio(matched_ratios[image_cnt]))

            if conv["from"] == "human":
                for i in range(image_num):
                    matched_ratio = matched_ratios[image_cnt]
                    ratio_tag = self.get_ratio_tag_from_ratio(matched_ratio)
                    text = re.sub(image_tag, ratio_tag + image_tag, text)
                    images.append(info["images"][image_cnt])
                    image_cnt += 1
            else:
                for i in range(image_num):
                    matched_ratio = matched_ratios[image_cnt]
                    ratio_tag = self.get_ratio_tag_from_ratio(matched_ratio)
                    image_embed_string = encode_image_token_into_code(image_embed_inds[image_cnt])
                    text = re.sub(image_tag, ratio_tag + image_embed_string, text)
                    image_cnt += 1
            conv["value"] = text

        if len(images):
            return {
                "images": images,
                "conversations": info["conversations"]
            }
        else:
            return {
                "conversations": info["conversations"]
            }

    def get_one_sample_data(self, info):
        # replace <image> tag with vision token and ratio tag.
        if "image_embed_inds" in info:
            info = self.get_unconditional_sample(info)
            info = self.replace_image_tag_with_vision_token(info)

        return info

    def read_one_image(self, image_file, image_folder):
        if isinstance(image_folder, list):
            for one_image_folder in image_folder:
                image_path = os.path.join(one_image_folder, image_file)
                if os.path.exists(image_path):
                    break
        else:
            image_path = os.path.join(image_folder, image_file)

        image = Image.open(image_path).convert('RGB')
        return image

    def preprocess_one_image(self, image, processor):
        if self.data_args.image_aspect_ratio == 'pad':
            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image_size = image.size
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        elif self.data_args.image_aspect_ratio == "anyres":  # llava-next setting
            image_size = image.size
            image, best_resolution = process_anyres_image(
                image, processor, self.data_args.image_grid_pinpoints)
            image_size = (image_size, best_resolution)
        elif self.data_args.image_aspect_ratio == "anyres_qwen":  # qwen2vit setting
            inputs = processor(images=[image], return_tensors="pt")
            image = inputs['pixel_values']  # [grid_H * grid_W, 1176]
            image_size = inputs['image_grid_thw'][0]  # [1, grid_H, grid_W]
        elif "anyres_dualvitok" in self.data_args.image_aspect_ratio:  # dualvitok setting
            base_resolution = self.data_args.get("base_resolution", 256)
            if self.data_args.image_aspect_ratio == "anyres_dualvitok_fix_centercrop":
                # for fixed resolution with generation data
                image = center_crop_and_resize(image, (base_resolution, base_resolution))
            elif self.data_args.image_aspect_ratio == "anyres_dualvitok_fix_resize":
                # for fixed resolution with understanding data
                image = image.resize((base_resolution, base_resolution))
            elif self.data_args.image_aspect_ratio == "anyres_dualvitok_fix_anchors":
                # for multiple aspect ratio resolution with generation data
                from illume.data.aspect_ratio_utils import AspectRatioCrop, RATIOS
                arc = AspectRatioCrop(RATIOS)
                image, _, _, _ = arc(image)
            image_size = image.size
            image = processor.preprocess(image, return_tensors='pt')['pixel_values']
        elif self.data_args.image_aspect_ratio == "resize":
            image_size = image.size
            image = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image_size = image.size
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image, image_size

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            info = self.list_data_dict[i]
            if isinstance(info, bytes):
                info = orjson.loads(info)
            sources = self.get_one_sample_data(info)
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self) - 1))

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        has_image = 'images' in sources[0]
        if has_image:
            processor = self.data_args.image_processor
            image_folder = self.image_folder

            image_list, image_size_list = [], []
            for image_file in sources[0]['images']:
                try:
                    image = self.read_one_image(image_file, image_folder)
                    image, image_size = self.preprocess_one_image(image, processor)
                except Exception as e:
                    print(e)
                    return self.__getitem__(random.randint(0, len(self) - 1))

                image_list.append(image)
                image_size_list.append(image_size)

        sources = copy.deepcopy([e["conversations"] for e in sources])

        try:
            data_dict = preprocess(
                sources,
                self.tokenizer,
                has_image=has_image)
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self) - 1))

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if has_image:  # image exist in the data
            data_dict['image'] = image_list
            data_dict['image_size'] = image_size_list
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            if self.data_args.image_aspect_ratio == "anyres_qwen":  # 448
                data_dict['image'] = torch.zeros(1024, 1176)
                data_dict['image_size'] = torch.tensor([1, 32, 32])
            elif "anyres_dualvitok" in self.data_args.image_aspect_ratio:
                base_resolution = self.data_args.get("base_resolution", 256)
                data_dict['image'] = torch.zeros(1, 3, base_resolution, base_resolution)
                data_dict['image_size'] = (base_resolution, base_resolution)
            else:
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                data_dict['image_size'] = (crop_size['height'], crop_size['width'])

            # transfer to list
            data_dict['image'] = [data_dict['image']]
            data_dict['image_size'] = [data_dict['image_size']]

        data_dict.update(dataset_name=f"{self.dataset_name}")

        return data_dict


class Text2ImageDataset(DefaultDataset):
    def __init__(self, meta_info: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(Text2ImageDataset, self).__init__(meta_info, tokenizer, data_args)

    def get_unconditional_sample(self, info):
        if random.random() < self.cfg_text_dropout_probability:
            question = random.choice(ROLE_TEMPLATES["random2image"])
            info["conversations"][0]["value"] = question

        return info


class SingleTurnEditDataset(DefaultDataset):
    def __init__(self, meta_info: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(SingleTurnEditDataset, self).__init__(meta_info, tokenizer, data_args)

    def get_unconditional_sample(self, info):
        if random.random() < self.cfg_text_dropout_probability:
            question = random.choice(ROLE_TEMPLATES["image_reconstruction"])
            info["conversations"][0]["value"] = question
            info["images"] = [info["images"][0], info["images"][0]]
            info["image_embed_inds"] = [info["image_embed_inds"][0], info["image_embed_inds"][0]]
            info["matched_ratios"] = [info["matched_ratios"][0], info["matched_ratios"][0]]

        return info

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_aspect_ratio: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            dataset_names=[instance['dataset_name'] for instance in instances]
        )

        images, image_sizes = [], []
        has_image = sum([len(instance['image']) for instance in instances]) > 0
        if has_image:
            for instance in instances:
                images.extend(instance['image'])
                image_sizes.extend(instance['image_size'])

            if self.image_aspect_ratio == "anyres_qwen":
                batch['images'] = torch.concat(images, dim=0)
                batch['image_sizes'] = torch.stack(image_sizes)
            elif "anyres_dualvitok" in self.image_aspect_ratio:
                batch['images'] = images
                batch['image_sizes'] = image_sizes
            else:
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch['images'] = torch.stack(images)
                else:
                    batch['images'] = images
                batch['image_sizes'] = image_sizes
        else:
            batch['images'] = None
            batch['image_sizes'] = image_sizes

        return batch


class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.modality_lengths = self._compute_modality_lengths()
        self.lengths = self._compute_lengths()

    def _compute_modality_lengths(self):
        modality_lengths = []
        for dataset in self.datasets:
            modality_lengths.extend(dataset.modality_lengths)
        return modality_lengths

    def _compute_lengths(self):
        lengths = []
        for dataset in self.datasets:
            lengths.extend(dataset.lengths)
        return lengths


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = []
    meta_data_info_list = data_args.meta_data_info_list
    for meta_data_info in meta_data_info_list:
        cur_data_args = copy.deepcopy(data_args)
        if isinstance(meta_data_info, dict):
            meta_configs = meta_data_info["meta_configs"]
            meta_infos = {}
            for meta_config in meta_configs:
                with open(meta_config, 'r') as f:
                    meta_info = json.load(f)
                    meta_infos.update(meta_info)

            if "image_aspect_ratio" in meta_data_info:
                cur_data_args["image_aspect_ratio"] = meta_data_info["image_aspect_ratio"]
            if "cfg_text_dropout_probability" in meta_data_info:
                cur_data_args["cfg_text_dropout_probability"] = meta_data_info["cfg_text_dropout_probability"]

            data_infos = meta_data_info["data_infos"]
            for dataset, sample_num in data_infos.items():
                meta_info = meta_infos[dataset]
                meta_info["dataset_name"] = dataset
                meta_info["codebook_dir"] = meta_data_info["codebook_dir"] if "codebook_dir" in meta_data_info else None
                meta_info["sample_num"] = sample_num
                cur_dataset = eval(meta_data_info["dataset_dtype"])(tokenizer=tokenizer,
                                                                    meta_info=meta_info,
                                                                    data_args=copy.deepcopy(cur_data_args))
                if len(cur_dataset) > 0:
                    train_dataset.append(cur_dataset)

    train_dataset = CustomConcatDataset(train_dataset)
    rank0_print("------------------------------------------------")
    rank0_print(f"dataset length: {train_dataset.__len__()}")
    rank0_print("------------------------------------------------")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer,
                                                     image_aspect_ratio=data_args.image_aspect_ratio)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
