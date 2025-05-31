from utils.registry_utils import Registry, build_from_cfg

MODELS = DETAIL_ENCODERS = DECODERS = QUANTIZERS = VQLOSSES = Registry('vqvae')


def build_vq_model(model_cfg, default_args=None, **kwargs):
    if kwargs:
        default_args.update(kwargs)
    model = build_from_cfg(model_cfg, MODELS, default_args=default_args)
    return model


def build_detail_encoder(model_cfg, default_args=None, **kwargs):
    if kwargs:
        default_args.update(kwargs)

    model = build_from_cfg(model_cfg, DETAIL_ENCODERS, default_args=default_args)
    return model


def build_decoder(model_cfg, default_args=None, **kwargs):
    if kwargs:
        default_args.update(kwargs)

    model = build_from_cfg(model_cfg, DECODERS, default_args=default_args)
    return model


def build_quantizer(model_cfg, default_args=None, **kwargs):
    if kwargs:
        default_args.update(kwargs)

    model = build_from_cfg(model_cfg, DECODERS, default_args=default_args)
    return model


def build_vqloss(model_cfg, default_args=None, **kwargs):
    if kwargs:
        default_args.update(kwargs)

    model = build_from_cfg(model_cfg, VQLOSSES, default_args=default_args)
    return model


# VQ_models = {'TokenFlow': TokenFlowFunc,
#              'dualvitok_pixel2d': PixelVQ2DFunc}


