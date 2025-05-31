import torch
import os
import math
import torch.nn as nn
from pathlib import Path
from einops import rearrange

from illume.model.multimodal_encoder.builder import VISION_TOWER, BaseModalityEncoder, build_mm_encoder
from illume.model.utils import load_state_dict_maybe_zero_3

from tokenizer.builder import build_vq_model
from tokenizer.movqgan.image_processing_movqgan import MoVQImageProcessor
from tokenizer.dualvitok_model import ScalingLayerForQwen2ViT

from illume.utils import read_config


@VISION_TOWER.register_module()
class DualVisionTower(BaseModalityEncoder):
    IMAGEPROCSSOR_OBJ_CLS = MoVQImageProcessor

    def __init__(self,
                 vq_config,
                 vq_ckpt=None,
                 min_pixels=256 * 256,
                 max_pixels=256 * 256,
                 use_ema=False,
                 delay_load=False,
                 unfreeze_mm_vision_tower=False,
                 **kwargs,
                 ):
        super().__init__()

        self.is_loaded = False
        self.delay_load = delay_load

        current_path = Path(__file__).resolve()  # path of current file
        vq_config = os.path.join(current_path.parent.parent.parent.parent.parent, vq_config)
        print("vq_config", vq_config)
        self.vq_config = read_config(vq_config)
        self.vq_config.use_ema = use_ema
        self.vq_ckpt = vq_ckpt

        self.hparam = kwargs

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.scaling_layer = ScalingLayerForQwen2ViT()

        if not delay_load:
            self.load_model()
        elif unfreeze_mm_vision_tower:
            self.load_model()
            # self.enable_gradient_checkpointing()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('vision tower is already loaded, `load_model` called again, skipping.')
            return

        self.is_loaded = True

        self.image_processor = self.IMAGEPROCSSOR_OBJ_CLS(min_pixels=self.min_pixels, max_pixels=self.max_pixels, spatial_factor=16)
        self.image_processor.crop_size = {'height': 256, 'width':256}

        vq_model = build_vq_model(self.vq_config.vq_model).to(device_map)
        if self.vq_ckpt is not None:
            checkpoint = torch.load(self.vq_ckpt, map_location="cpu")
            if "ema" in checkpoint and self.vq_config.use_ema:  # ema
                model_weight = checkpoint["ema"]
                print("Using ema params for evaluation.")
            elif "model" in checkpoint:  # ddp
                model_weight = checkpoint["model"]
            elif "state_dict" in checkpoint:
                model_weight = checkpoint["state_dict"]
            else:
                model_weight = checkpoint
            # msg = vq_model.load_state_dict(model_weight, strict=False)
            msg = load_state_dict_maybe_zero_3(vq_model, model_weight, strict=False)
            print(f'Using vq tokenizer checkpoint from {self.vq_ckpt}. message: {msg}')
            del checkpoint

        self.pixel_encoder = vq_model.pixel_encoder
        self.semantic_encoder = vq_model.semantic_encoder.model

        class DotDict(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        self._config = DotDict()
        self._config.hidden_size = [3584, 32]
        self._config.image_size = [252, 256]
        self._config.patch_size = [28, 16]

    def forward(self, images):
        if isinstance(images, list) and all(x is not None and x.shape == images[0].shape for x in images):
            images = torch.concat(images, dim=0)
            images = images.to(device=self.device, dtype=self.dtype)
        else:
            images = [image.to(device=self.device, dtype=self.dtype) for image in images]

        image_feature_shape_pixels, image_feature_shape_semantics = [], []
        if isinstance(images, list):  # anyres setting
            h_pixels = []
            for image in images:
                if image.ndim==3:
                    image = image.unsqueeze(0)
                h_pixel = self.pixel_encoder(image)
                b, c, h, w = h_pixel.shape
                image_feature_shape_pixels.append((h, w))
                h_pixel = rearrange(h_pixel, 'b c h w -> b (h w) c')
                h_pixels.append(h_pixel)
            h_pixels = torch.cat(h_pixels, dim=1)

            h_semantics = []
            for image in images:
                if image.ndim==3:
                    image = image.unsqueeze(0)
                image = image.unsqueeze(dim=1)
                scale_output = self.scaling_layer(image.clone())
                image_2, image_grid_thw = scale_output['image'], scale_output['image_grid_thw']
                image_feature_shape_semantics.append((int(image_grid_thw[0][1]) // 2, int(image_grid_thw[0][2] // 2)))
                h_semantic = self.semantic_encoder(image_2, image_grid_thw)
                h_semantics.append(h_semantic)
            h_semantics = torch.cat(h_semantics, dim=0)
            h_semantics = h_semantics.unsqueeze(dim=0)

            image_feature_shapes = [[shape_semantic, shape_pixel] for shape_semantic, shape_pixel in zip(image_feature_shape_semantics, image_feature_shape_pixels)]

        else:  # fixed res setting
            assert images.ndim==4
            h_pixels = self.pixel_encoder(images)
            b, c, h, w = h_pixels.shape
            h_pixels = rearrange(h_pixels, 'b c h w -> (b h w) c')
            h_pixels = h_pixels.unsqueeze(dim=0)

            images = images.unsqueeze(dim=1)
            scale_output = self.scaling_layer(images.clone())
            images_2, images_grid_thw = scale_output['image'], scale_output['image_grid_thw']

            h_semantics = self.semantic_encoder(images_2, images_grid_thw)
            h_semantics = h_semantics.unsqueeze(dim=0)

            shape_semantic = (int(images_grid_thw[0][1]) // 2, int(images_grid_thw[0][2] // 2))
            shape_pixel = (h, w)
            image_feature_shapes = [[shape_semantic, shape_pixel] for i in range(b)]

        return [h_semantics, h_pixels], image_feature_shapes

    def tune(self):
        if self.hparam.get('tune_vit_from_layer', None):
            print(f"Tuning vit from layer {self.hparam.get('tune_vit_from_layer')}")
            # semantic encoder
            for n, p in self.semantic_encoder.named_parameters():
                if 'blocks.' in n:
                    layer_id = int(
                        n.split('blocks.')[-1].split('.')[0])
                    if layer_id >= self.hparam.get('tune_vit_from_layer'):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                elif 'merger' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            # pixel encoder
            for n, p in self.pixel_encoder.named_parameters():
                if 'down' in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            super(DualVisionTower, self).tune()

    def freeze(self):
        if self.is_loaded:
            self.pixel_encoder.requires_grad_(False)
            self.pixel_encoder.eval()

            self.semantic_encoder.requires_grad_(False)
            self.semantic_encoder.eval()

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.semantic_encoder.dtype

    @property
    def device(self):
        return self.semantic_encoder.device

    @property
    def config(self):
        return self._config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
