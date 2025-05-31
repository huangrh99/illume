from abc import ABC
from functools import partial

import torch
import torch.nn as nn
import re

from illume.registry_utils import build_from_cfg

from einops import rearrange, repeat
from . import MM_PROJECTOR


class BaseMMProjector(ABC, nn.Module):
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def tune(self):
        for p in self.parameters():
            p.requires_grad = True

    @property
    def downsample_rate(self):
        return 1

    @property
    def downsample_rate_per_side(self):
        return 1


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super(LambdaLayer, self).__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@MM_PROJECTOR.register_module()
class IdentityMap(BaseMMProjector):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


@MM_PROJECTOR.register_module()
class SimpleResBlock(BaseMMProjector):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


@MM_PROJECTOR.register_module()
class MLPProjector(nn.Sequential, BaseMMProjector):
    def __init__(self, mm_hidden_size, hidden_size, mlp_depth=2, pre_norm=False):
        modules = []
        if pre_norm:
            modules.append(nn.LayerNorm(mm_hidden_size))
        modules.append(nn.Linear(mm_hidden_size, hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        super(MLPProjector, self).__init__(*modules)


class ImagePosEmbeddingLayer(nn.Module):
    def __init__(self, num_input_token, mm_hidden_size):
        super(ImagePosEmbeddingLayer, self).__init__()
        self.pos_emb = nn.Parameter(
            torch.randn(num_input_token, mm_hidden_size) * 0.02
        )

    def forward(self, x):
        return x + self.pos_emb.unsqueeze(0)


class ImageNewLineEmbeddingLayer(nn.Module):
    def __init__(self, hidden_size):
        super(ImageNewLineEmbeddingLayer, self).__init__()
        embed_std = 1 / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
        # embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
        self.image_newline = nn.Parameter(
            torch.randn(hidden_size) * embed_std
        )

    def forward(self, image_feature):
        return torch.cat(
            (image_feature,
            repeat(self.image_newline[:, None, None],
                   'c 1 1-> c l 1', l=image_feature.shape[1]).to(image_feature.device)),
            dim=-1)


class BaseDownsamplerMMProjector(BaseMMProjector):
    def __init__(self, mm_hidden_size,
                 hidden_size,
                 mlp_depth=2,
                 whole_image_first=True,
                 has_sep_token=False,
                 window_resolution=2,
                 downsample_rate=4,
                 downsample_size=None,
                 num_input_token=576,
                 add_pos_embed=False,
                 add_pre_norm=False,
                 add_image_newline_embed=False,
                 enable_gradient_checkpointing=False,
                 **kwargs
                 ):
        super(BaseDownsamplerMMProjector, self).__init__()
        self.window_resolution = window_resolution
        self.whole_image_first = whole_image_first
        self.has_sep_token = has_sep_token
        self.enable_gradient_checkpointing = enable_gradient_checkpointing

        self._downsample_rate = downsample_rate
        self.downsample_size = (downsample_size, downsample_size) \
            if isinstance(downsample_size, int) else downsample_size
        assert num_input_token // self.downsample_rate == self.downsample_size[0] * self.downsample_size[1]

        if has_sep_token:
            self.global_img_sep = nn.Parameter(torch.randn([1, 1, mm_hidden_size]) * mm_hidden_size ** -0.5)
            self.sub_img_sep = nn.Parameter(torch.randn([1, 1, 1, mm_hidden_size]) * mm_hidden_size ** -0.5)
            self.sub_img_line_sep = nn.Parameter(torch.randn([1, 1, 1, mm_hidden_size]) * mm_hidden_size ** -0.5)

        self.downsampler = self.build_downsampler(mm_hidden_size, **kwargs)
        self.mlp = self.build_mlp(mm_hidden_size, hidden_size, mlp_depth)

        self.norm = nn.LayerNorm(mm_hidden_size) if add_pre_norm else None

        #  nn.Parameter should be included in a nn.Module. Otherwise, load state_dict in sft will be size(0)
        self.pos_emb = self.build_pos_embed(add_pos_embed, num_input_token, mm_hidden_size)
        self.image_newline_embed_layer = self.build_image_newline_embed(add_image_newline_embed, hidden_size)

    def build_pos_embed(self, add_pos_embed, num_input_token, mm_hidden_size):
        if add_pos_embed:
            pos_emb = ImagePosEmbeddingLayer(num_input_token, mm_hidden_size)
        else:
            pos_emb = None
        return pos_emb

    def build_image_newline_embed(self, add_image_newline_embed, hidden_size):
        if add_image_newline_embed:
            embedding_layer = ImageNewLineEmbeddingLayer(hidden_size)
        else:
            embedding_layer = None
        return embedding_layer

    def build_mlp(self, mm_hidden_size, hidden_size, mlp_depth):
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    def build_downsampler(self, mm_hidden_size, **kwargs):
        raise NotImplementedError

    def forward_downsampler(self, x):
        raise NotImplementedError

    def add_sep_token(self, x, windows_resolution):
        if self.whole_image_first:
            global_img, window_imgs = x.split([1, x.size(1) - 1], dim=1)
        else:
            window_imgs, global_img = x.split([x.size(1) - 1, 1], dim=1)

        # append sub images seperators.
        sub_img_seps = repeat(self.sub_img_sep, '1 1 1 c -> b m 1 c', b=window_imgs.size(0),
                              m=window_imgs.size(1))
        window_imgs = torch.cat([window_imgs, sub_img_seps], dim=2)

        # append line seperators.
        window_imgs = rearrange(window_imgs, 'b (p k) l c -> b p (k l) c', p=windows_resolution[0],
                                k=windows_resolution[1])
        sub_img_line_seps = repeat(self.sub_img_sep, '1 1 1 c -> b p 1 c',
                                   b=window_imgs.size(0),
                                   p=window_imgs.size(1))
        window_imgs = torch.cat([window_imgs, sub_img_line_seps], dim=2)

        # add global image seperator.
        window_imgs = rearrange(window_imgs, 'b p l c -> b (p l) c')
        global_img = rearrange(global_img, 'b p l c -> b (p l) c')
        global_seps = repeat(self.global_img_sep, '1 1 c -> b 1 c', b=global_img.size(0))

        if self.whole_image_first:
            x = torch.cat([global_img, global_seps, window_imgs], dim=1)
        else:
            x = torch.cat([window_imgs, global_seps, global_img], dim=1)
        return x

    def forward(self, x, windows_resolution=None):

        if self.norm is not None:
            x = self.norm(x)

        if self.pos_emb is not None:
            x = self.pos_emb(x)

        if x.ndim == 3:
            # [b, l, c]
            # if self.enable_gradient_checkpointing:
            #     x = torch.utils.checkpoint.checkpoint(self.forward_downsampler, x)
            # else:
            x = self.forward_downsampler(x)
        else:
            # [b, p, l, c]
            if windows_resolution is None:
                windows_resolution = (self.window_resolution, self.window_resolution)
            if self.whole_image_first:
                x = torch.cat([x[:, -1:], x[:, :-1]], dim=1)

            # if self.enable_gradient_checkpointing:
            #     x = torch.utils.checkpoint.checkpoint(self.forward_downsampler, x)
            # else:
            x = self.forward_downsampler(x)

            if self.add_sep_token:
                x = self.add_sep_token(x, windows_resolution)
            else:
                x = rearrange(x, 'b m l c -> b (m l) c')

        if self.enable_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self.mlp, x)
        else:
            return self.mlp(x)

    @property
    def downsample_rate(self):
        return self._downsample_rate

    @property
    def downsample_rate_per_side(self):
        res = self._downsample_rate ** 0.5
        if res.is_integer():
            return int(res)
        else:
            return res


@MM_PROJECTOR.register_module()
class AvgPoolMMProjector(BaseDownsamplerMMProjector):
    def build_downsampler(self, mm_hidden_size, ):
        return nn.AdaptiveAvgPool2d(self.downsample_size)

    def forward_downsampler(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        if x.ndim == 3:
            h = x.size(1) ** 0.5
            assert h.is_integer()
            x = rearrange(x, 'b (h w) c -> b c h w', h=int(h))
            x = self.downsampler(x)
            x = rearrange(x, 'b c h w -> b (h w) c ')
        else:
            m = x.size(1)
            h = x.size(2) ** 0.5
            assert h.is_integer()
            x = rearrange(x, 'b m (h w) c -> (b m) c h w', h=int(h))
            x = self.downsampler(x)
            x = rearrange(x, '(b m) c h w -> b m (h w) c ', m=m)
        x = x.to(dtype)
        return x


@MM_PROJECTOR.register_module()
class CAbstractorMMProjector(BaseDownsamplerMMProjector):
    def build_downsampler(self, mm_hidden_size,
                          conv_hidden_size=1024, conv_block_depth=3):
        from timm.models.regnet import RegStage
        try:
            from timm.layers import LayerNorm2d
        except:
            from timm.models.layers import LayerNorm2d

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            conv_block_depth,
            mm_hidden_size,
            conv_hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d(self.downsample_size)
        s2 = RegBlock(
            conv_block_depth,
            conv_hidden_size,
            conv_hidden_size,
        )
        self.conv_hidden_size = conv_hidden_size
        return nn.Sequential(
            s1,
            sampler,
            s2,
        )

    def build_mlp(self, mm_hidden_size, hidden_size, mlp_depth):
        modules = [nn.Linear(self.conv_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    def forward_downsampler(self, x, grid_thw=None):
        # print("!!! grid_thw: ", grid_thw)
        if x.ndim == 3:
            if grid_thw:
                h = grid_thw[0, 1]
            else:
                h = x.size(1) ** 0.5
            assert h.is_integer()

            x = rearrange(x, 'b (h w) c -> b c h w', h=int(h))

            if self.enable_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.downsampler, x)
            else:
                x = self.downsampler(x)

            x = rearrange(x, 'b c h w -> b (h w) c ')
        else:
            m = x.size(1)
            if grid_thw:
                h = grid_thw[0, 1]
            else:
                h = x.size(2) ** 0.5
            assert h.is_integer()

            x = rearrange(x, 'b m (h w) c -> (b m) c h w', h=int(h))
            x = self.downsampler(x)
            x = rearrange(x, '(b m) c h w -> b m (h w) c ', m=m)
        return x


# FIXME(HUI): Not test yet.
@MM_PROJECTOR.register_module()
class ConcatChannelMMProjector(BaseDownsamplerMMProjector):
    def build_downsampler(self, mm_hidden_size, **kwargs):
        return LambdaLayer(lambda x: rearrange(x, 'b (h p w l) c -> b (h w) (p l c)',
                                               h=int((x.size(1) // self.downsample_rate) ** 0.5),
                                               p=self.downsample_rate_per_side,
                                               l=self.downsample_rate_per_side))

    def build_mlp(self, mm_hidden_size, hidden_size, mlp_depth):
        modules = [nn.Linear(mm_hidden_size * self.downsample_rate, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    def forward_downsampler(self, x):
        if x.ndim == 3:
            x = self.downsampler(x)
        else:
            m = x.size(1)
            x = rearrange(x, 'b m l c -> (b m) l c')
            x = self.downsampler(x)
            x = rearrange(x, '(b m) l c -> b m l c', m=m)
        return x


@MM_PROJECTOR.register_module()
class MixedProjector(nn.Module):
    def __init__(self, projector_cfg1, projector_cfg2, mm_hidden_size, hidden_size, **kwargs):
        super(MixedProjector, self).__init__()
        projector_cfg1.update(mm_hidden_size=mm_hidden_size[0], hidden_size=hidden_size)
        projector_cfg2.update(mm_hidden_size=mm_hidden_size[1], hidden_size=hidden_size)
        self.projector_1 = build_from_cfg(projector_cfg1, MM_PROJECTOR)
        self.projector_2 = build_from_cfg(projector_cfg2, MM_PROJECTOR)
    def tune(self):
        self.projector_1.tune()
        self.projector_2.tune()
    def freeze(self):
        self.projector_1.freeze()
        self.projector_2.freeze()
    def downsample_rate(self):
        return self.projector_1.downsample_rate

    @property
    def downsample_rate_per_side(self):
        return self.projector_1.downsample_rate_per_side

    def forward(self, image_features):
        image_feature_1, image_feature_2 = image_features
        image_feature_1 = self.projector_1(image_feature_1)
        image_feature_2 = self.projector_2(image_feature_2)
        image_features = torch.concat([image_feature_1, image_feature_2], dim=1)
        return image_features


def build_mm_projector(mm_projector_cfg, **kwargs):
    mm_projector_cfg.update(kwargs)
    trainable = mm_projector_cfg.pop('trainable', True)
    model = build_from_cfg(mm_projector_cfg, MM_PROJECTOR)

    if trainable:
        model.tune()
    else:
        model.freeze()
    return model


def build_vision_projector(vision_projector_cfg, **kwargs):
    return build_mm_projector(vision_projector_cfg, **kwargs)
