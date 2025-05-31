""" MoVQ model """

import math
from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import PreTrainedModel

from .configuration_movqgan import MoVQConfig

try:
    import xformers.ops as xops

    is_xformers_available = True
except Exception as e:
    is_xformers_available = False

if torch.__version__ > "2.1.2":
    IS_SDPA_AVAILABLE = True
else:
    IS_SDPA_AVAILABLE = False


class MoVQActivation(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


class MoVQUpsample(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x.float(), scale_factor=2.0, mode="nearest").to(x.dtype)
        x = self.conv(x)
        return x


class DCDownBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, downsample: bool = True,
                 shortcut: bool = True) -> None:
        super().__init__()
        out_channels = out_channels if out_channels else in_channels

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = in_channels * self.factor ** 2 // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor ** 2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states  # x + y


class DCUpBlock2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int = None,
            interpolate: bool = False,
            shortcut: bool = True,
            interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()
        out_channels = out_channels if out_channels else in_channels

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor ** 2 // in_channels

        out_ratio = self.factor ** 2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class MoVQDownsample(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class MoVQSpatialNorm(nn.Module):

    def __init__(
            self,
            f_channels: int,
            zq_channels: int,
            norm_layer: nn.Module = nn.GroupNorm,
            add_conv: bool = False,
            num_groups: int = 32,
            eps: float = 1e-6,
            affine: bool = True,
    ):
        super().__init__()
        self.norm_layer = norm_layer(
            num_channels=f_channels,
            num_groups=num_groups,
            eps=eps,
            affine=affine,
        )

        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv2d(
                zq_channels,
                zq_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.conv_y = nn.Conv2d(
            zq_channels,
            f_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_b = nn.Conv2d(
            zq_channels,
            f_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor, zq: torch.Tensor):
        zq = F.interpolate(zq.float(), size=x.shape[-2:], mode="nearest").to(zq.dtype)

        if self.add_conv:
            zq = self.conv(zq)

        x = self.norm_layer(x)
        x = x * self.conv_y(zq) + self.conv_b(zq)
        return x


class MoVQResnetBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            conv_shortcut: bool = False,
            dropout: float = 0.0,
            zq_ch: Optional[int] = None,
            add_conv: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.zq_ch = zq_ch

        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, eps=1e-6, affine=True)
            self.norm1 = nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, **norm_kwargs)
        else:
            self.norm1 = MoVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)
            self.norm2 = MoVQSpatialNorm(out_channels, zq_ch, add_conv=add_conv)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.act = MoVQActivation()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x: torch.Tensor, zq: Optional[torch.Tensor] = None):
        norm_args = tuple() if self.zq_ch is None else (zq,)

        h = self.norm1(x, *norm_args)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h, *norm_args)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class MoVQAttnBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            zq_ch: Optional[int] = None,
            add_conv: bool = False,
            num_heads=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.zq_ch = zq_ch
        self.num_heads = num_heads

        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, eps=1e-6, affine=True)
            self.norm = nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
        else:
            self.norm = MoVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)

        self.q = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor, zq: Optional[torch.Tensor] = None):
        # x: [b, c1, h1, w1]
        # zq: [b, c2, h2, w2]
        # attention_mask: [b, 1, h3, w3]
        norm_args = tuple() if self.zq_ch is None else (zq,)

        # if context is not None:
        #     context = F.interpolate(context.float(), size=x.shape[-2:], mode="nearest").to(context.dtype)
        #     x = x + self.conv_context(context)

        nx = self.norm(x, *norm_args)
        q = self.q(nx)
        k = self.k(nx)
        v = self.v(nx)

        b, c, h, w = q.shape
        if is_xformers_available:
            # If xformers is available, create attn_bias for xops.memory_efficient_attention.
            attn_bias = None

            v = xops.memory_efficient_attention(
                rearrange(q, 'b (n c) h w -> b (h w) n c', n=self.num_heads).contiguous(),
                rearrange(k, 'b (n c) h w -> b (h w) n c', n=self.num_heads).contiguous(),
                rearrange(v, 'b (n c) h w -> b (h w) n c', n=self.num_heads).contiguous(),
                scale=1.0 / math.sqrt(c // self.num_heads),
                attn_bias=attn_bias,
            )
            v = rearrange(v, 'b (h w) n c -> b (n c) h w', h=h, w=w).contiguous()
        elif IS_SDPA_AVAILABLE:
            # compute attention
            q = rearrange(q, 'b (n c) h w -> b n (h w) c', n=self.num_heads).contiguous()
            k = rearrange(k, 'b (n c) h w -> b n (h w) c', n=self.num_heads).contiguous()
            v = rearrange(v, 'b (n c) h w -> b n (h w) c', n=self.num_heads).contiguous()

            attn_bias = None

            v = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
            v = v.transpose(1, 2)
            v = rearrange(v, 'b (h w) n c -> b (n c) h w', h=h, w=w)
        else:
            # compute attention
            q = rearrange(q, 'b (n c) h w -> b n c (h w)', n=self.num_heads).contiguous()
            k = rearrange(k, 'b (n c) h w -> b n c (h w)', n=self.num_heads).contiguous()
            v = rearrange(v, 'b (n c) h w -> b n c (h w)', n=self.num_heads).contiguous()

            # score = torch.bmm(q.permute(0, 2, 1), k)
            score = torch.einsum('b n c k, b n c l -> b n k l', q, k)
            score = score / math.sqrt(c // self.num_heads)

            score = F.softmax(score, dim=2)

            # attend to values
            # v = v.reshape(b, c, h * w)
            # v = torch.bmm(v, score.permute(0, 2, 1))
            v = torch.einsum('b n c l, b n k l -> b n c k', v, score)
            v = v.reshape(b, c, h, w)

        v = self.proj_out(v)

        return x + v


class MoVQVectorQuantizer(nn.Module):

    def __init__(self, config: MoVQConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.codebook_size, config.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / config.codebook_size, 1.0 / config.codebook_size)

    def forward(self, x: torch.Tensor):
        # b t c h w -> b t h w c
        b, t, c, h, w = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x_flattened = x.view(-1, c)

        codebook = self.embedding.weight

        d = torch.sum(x_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', x_flattened, codebook.permute(1, 0))

        indices = torch.argmin(d, dim=1)
        indices = indices.view(b, t, h, w)
        return indices


class MoVQPretrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MoVQConfig
    base_model_prefix = "movq"
    main_input_name = "pixel_values"
    _no_split_modules = ["MoVQResnetBlock", "MoVQAttnBlock"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # copied from the `reset_parameters` method of `class Linear(Module)` in `torch`.
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class MoVQEncoder(nn.Module):
    def __init__(self, config: MoVQConfig):
        super().__init__()
        self.config = config
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.in_channels = config.in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            self.in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            padding=1
        )

        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    MoVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(MoVQAttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if config.use_dc_up_down_blocks:
                    down.downsample = DCDownBlock2d(block_in)
                else:
                    down.downsample = MoVQDownsample(block_in)

            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = MoVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )
        self.mid.attn_1 = MoVQAttnBlock(block_in)
        self.mid.block_2 = MoVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )

        # end

        self.norm_out = nn.GroupNorm(num_channels=block_in, num_groups=32, eps=1e-6, affine=True)

        self.act = MoVQActivation()

        out_z_channels = 2 * config.z_channels if config.double_z else config.z_channels
        self.conv_out = nn.Conv2d(
            block_in,
            out_z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.out_shortcut_average_group_size = block_in // out_z_channels

    def forward(self, x: torch.Tensor):

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = self.act(h)

        if self.config.use_dc_up_down_blocks:
            x = h.unflatten(1, (-1, self.out_shortcut_average_group_size))
            x = x.mean(dim=2)
            h = self.conv_out(h) + x
        else:
            h = self.conv_out(h)
        return h


class MoVQDecoder(nn.Module):
    def __init__(self, config: MoVQConfig):
        super().__init__()
        self.config = config
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks

        in_ch_mult = (1,) + tuple(config.ch_mult)
        zq_ch = config.embed_dim

        block_in = config.ch * config.ch_mult[-1]

        self.in_shortcut_repeats = block_in // config.embed_dim

        self.conv_in = nn.Conv2d(
            config.z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = MoVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
            zq_ch=zq_ch,
        )
        self.mid.attn_1 = MoVQAttnBlock(block_in, zq_ch)
        self.mid.block_2 = MoVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
            zq_ch=zq_ch,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    MoVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                        zq_ch=zq_ch,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(MoVQAttnBlock(block_in, zq_ch))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if config.use_dc_up_down_blocks:
                    up.upsample = DCUpBlock2d(block_in)
                else:
                    up.upsample = MoVQUpsample(block_in)

            self.up.insert(0, up)

        self.act = MoVQActivation()

        self.norm_out = MoVQSpatialNorm(block_in, zq_ch)
        self.conv_out = nn.Conv2d(
            block_in,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z: torch.Tensor, zq: torch.Tensor):
        h = z

        if self.config.use_dc_up_down_blocks:
            h = h.repeat_interleave(self.in_shortcut_repeats, dim=1)
            h = self.conv_in(z) + h
        else:
            h = self.conv_in(h)

        # middle
        h = self.mid.block_1(h, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h, zq)
        h = self.act(h)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    def __init__(self, config: MoVQConfig):
        super().__init__()
        self.config = config
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks

        in_ch_mult = (1,) + tuple(config.ch_mult)

        block_in = config.ch * config.ch_mult[-1]

        self.conv_in = nn.Conv2d(
            config.z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = MoVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )
        self.mid.attn_1 = MoVQAttnBlock(block_in)
        self.mid.block_2 = MoVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    MoVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(MoVQAttnBlock(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = MoVQUpsample(block_in)

            self.up.insert(0, up)

        self.act = MoVQActivation()

        norm_kwargs = dict(num_groups=32, eps=1e-6, affine=True)
        self.norm_out = nn.GroupNorm(num_channels=block_in, **norm_kwargs)
        self.conv_out = nn.Conv2d(
            block_in,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z: torch.Tensor, zq: torch.Tensor):
        h = z
        h = self.conv_in(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)

        return h


class MoVQModel(MoVQPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = MoVQEncoder(config)
        self.decoder = MoVQDecoder(config)
        self.quantize = MoVQVectorQuantizer(config)

        self.quant_conv = nn.Conv2d(config.z_channels, config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.z_channels, 1)

        self.spatial_scale_factor = 2 ** (len(config.ch_mult) - 1)

        self.post_init()

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = self.quant_conv(h)
        codes = self.quantize(h)
        return codes

    def decode(self, x: torch.Tensor):
        quant = self.quantize.embedding(x.flatten())
        b, h, w, c = quant.shape
        quant = quant.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        quant2 = self.post_quant_conv(quant)
        image = self.decoder(quant2, quant)
        image = image.reshape(
            b,
            self.config.out_channels,
            h * self.spatial_scale_factor,
            w * self.spatial_scale_factor,
        )
        return image

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
