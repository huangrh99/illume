import json
import os
import random
from torch.utils.data import Dataset
from PIL import Image

from .builder import GENERATION_EVAL_DATASET
from .DefaultDataset import DefaultDataset
from illume.data.data_utils import ROLE_TEMPLATES

@GENERATION_EVAL_DATASET.register_module()
class EditingSingleTurnDataset(DefaultDataset):
    def __init__(
            self,
            annotation_path="",
            role="",
            unconditional_role="",
            ratios=[(256, 256)],
            sample_num=-1,
            image_dir="",
            **kwargs
    ):
        self.image_dir = image_dir
        super(EditingSingleTurnDataset, self).__init__(annotation_path, role, unconditional_role, ratios, sample_num, **kwargs)

    def load_data(self, annotation_path, sample_num):
        with open(annotation_path, 'r') as f:
            infos = [json.loads(l.strip('\n')) for l in f.readlines()]

        annotations = []
        for id, anno in enumerate(infos):
            instruction = anno["conversations"][0]["value"].replace("<image>", "").rstrip('.')
            input_image_path = anno["images"][0]
            out_image_path = f'{id}-{instruction.replace("/", " ").replace(".", "")[:200]}.png'
            sys_prompt = random.choice(ROLE_TEMPLATES[self.role]) if self.role != 'default' else "@"
            prompt = sys_prompt.replace("<prompt>", instruction)

            annotations.append({
                "id": id,
                "instruction": instruction,
                "prompt": prompt,
                "input_image_path": input_image_path,
                "out_image_path": out_image_path,
            })

        if sample_num > 0:
            annotations = annotations[:sample_num]

        return annotations

