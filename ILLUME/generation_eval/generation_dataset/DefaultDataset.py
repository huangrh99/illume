import os
import json
import random
from torch.utils.data import Dataset
from PIL import Image

from .builder import GENERATION_EVAL_DATASET
from illume.data.data_utils import ROLE_TEMPLATES


@GENERATION_EVAL_DATASET.register_module()
class DefaultDataset(Dataset):
    def __init__(
        self,
        annotation_path="",
        role="",
        unconditional_role="",
        ratios=[(256, 256)],
        sample_num=-1,
        **kwargs
    ):

        self.role = role
        self.unconditional_role = unconditional_role
        self.ratios = ratios
        self.annotations = self.load_data(annotation_path, sample_num)

    def load_data(self, annotation_path, sample_num):
        with open(annotation_path, 'r') as f:
            infos = [json.loads(l.strip('\n')) for l in f.readlines()]
        annotations = []
        for id, anno in enumerate(infos):
            caption = anno["conversations"][0]["value"]
            out_image_path = f'{id}-{caption.replace("/", " ").replace(".", "")[:200]}.png'
            sys_prompt = random.choice(ROLE_TEMPLATES[self.role])
            prompt = sys_prompt.replace("<prompt>", caption)

            annotations.append({
                "id": id,
                "caption": caption,
                "prompt": prompt,
                "out_image_path": out_image_path
            })
        if sample_num > 0:
            annotations = annotations[:sample_num]
        return annotations

    def get_unconditional_prompt(self):
        return random.choice(ROLE_TEMPLATES[self.unconditional_role])

    def get_ratios(self):
        return self.ratios

    def get_ratio_name_from_ratio(self, ratio):
        return f"ratio_h{ratio[0]}_w{ratio[1]}"

    def get_role(self):
        return self.role

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        info = self.annotations[idx]

        if "input_image_path" in info:
            if isinstance(info["input_image_path"], str):
                image_path = os.path.join(self.image_dir, info["input_image_path"])
                image = Image.open(image_path).convert('RGB')
                info["images_data"] = [image]
            elif isinstance(info["input_image_path"], list):
                images_data = []
                for cur_image_path in info["input_image_path"]:
                    image_path = os.path.join(self.image_dir, cur_image_path)
                    image = Image.open(image_path).convert('RGB')
                    images_data.append(image)
                info["images_data"] = images_data
        return info

