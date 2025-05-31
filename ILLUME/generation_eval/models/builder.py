from illume.registry_utils import Registry, build_from_cfg

EVAL_MODELS = Registry('eval_models')

def build_eval_model(eval_model_cfg, **kwargs):
    eval_model_cfg.update(kwargs)
    eval_model = build_from_cfg(eval_model_cfg, EVAL_MODELS)
    return eval_model
