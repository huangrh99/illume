import os
import re

from transformers import AutoConfig

from illume.dist_utils import get_rank, synchronize


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer ILLUME code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'IllumeLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def is_zero3_model(model=None, params=None):
    if model:
        params = list(model.parameters())

    assert params
    for p in params:
        if any(hasattr(p, attr) for attr in ['ds_tensor', 'ds_id', 'ds_status', 'ds_shape', 'ds_numel']):
            return True
    else:
        return False


def load_state_dict_maybe_zero_3(model, state_dict, strict=False, ignore_status=False):
    import deepspeed
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    params = list(model.parameters())
    msg = None
    if is_zero3_model(params=params):
        with zero.GatheredParameters(params, modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                msg = model.load_state_dict(state_dict, strict=strict)
    else:
        msg = model.load_state_dict(state_dict, strict=strict)
    return msg
