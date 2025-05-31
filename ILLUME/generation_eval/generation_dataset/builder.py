import io

from illume.registry_utils import Registry, build_from_cfg
GENERATION_EVAL_DATASET = Registry('generation_eval_dataset')

from .meta_dataset_configs import *


def build_eval_dataset(dataset_name, update_configs={}):
    dataset_cfg = eval(dataset_name)
    dataset_cfg.update(update_configs)
    dataset = build_from_cfg(dataset_cfg, GENERATION_EVAL_DATASET)
    return dataset