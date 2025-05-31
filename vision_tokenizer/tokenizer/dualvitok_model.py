# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/
#   VAR: https://github.com/FoundationVision/VAR
import math
import os
import sys
from functools import partial

from tokenizer.builder import MODELS, QUANTIZERS, build_quantizer
from tokenizer.movqgan.configuration_movqgan import MoVQConfig
from tokenizer.movqgan.modeling_movqgan import MoVQModel, MoVQEncoder, MoVQDecoder, Decoder
from tokenizer.qwen2vit.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel, \
    VisionRotaryEmbedding, Qwen2VLBatchVisionBlock

from tokenizer.qwen2vit.configuration_qwen2_vl import Qwen2VLVisionConfig

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import get_parameter_device, get_parameter_dtype

from timm.models.layers import trunc_normal_

from einops import rearrange, repeat


def init_weights(m, dim=None):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        if dim:
            std = math.sqrt(2 / (5 * dim))
            m.weight.data.normal_(mean=0.0, std=std)
            print(f'Init nn.Linear with Scale Embed method. std: {std}')
        else:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        w = m.weight.data
        nn.init.xavier_uniform_(w)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        if dim:
            std = math.sqrt(2 / (5 * dim))
            m.weight.data.normal_(mean=0.0, std=std)
            print(f'Init nn.Embedding with Scale Embed method. std: {std}')
        else:
            nn.init.normal_(m.weight, mean=0, std=1)

RESOLUTION_MAPPING = {
        (256, 256): (252, 252),
        (512, 512): (504, 504),
        (512, 384): (448, 336),
        (384, 512): (336, 448),
        (768, 384): (728, 364),
        (384, 768): (364, 728),
        (1024, 256): (1008, 252),
        (256, 1024): (252, 1008),
        (384, 576): (336, 504),
        (576, 384): (504, 336),
        (960, 320): (924, 308),
        (320, 960): (308, 924),
    }

class ScalingLayerForQwen2ViT:
    def __init__(
            self,
            min_pixels: int = 56 * 56,
            max_pixels: int = 28 * 28 * 1280,
            patch_size: int = 14,
            temporal_patch_size: int = 2,
            merge_size: int = 2,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        OPENAI_CLIP_MEAN = torch.as_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
        OPENAI_CLIP_STD = torch.as_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]

        self.image_mean = OPENAI_CLIP_MEAN
        self.image_std = OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}

    def __call__(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(1)
        batch_size, temporal, channel, height, width = images.shape

        factor = self.patch_size * self.merge_size

        if (height, width) in RESOLUTION_MAPPING:
            resized_height, resized_width = RESOLUTION_MAPPING[(height, width)]
        else:
            resized_height, resized_width = max(factor, height // factor * factor), max(factor, width // factor * factor)

        images = (images + 1) / 2  # rescale to [0, 1.]

        images = torch.nn.functional.interpolate(
            images.flatten(0, 1).float(),
            size=(resized_height, resized_width),
            mode='bicubic',
            align_corners=False,
            antialias=True
        ).to(images.dtype)

        images = images.clamp(0, 1)  # rescale to [0, 1.]
        images = ((images - self.image_mean.to(images)) / self.image_std.to(images))

        images = rearrange(images, '(b t) c h w -> b t c h w', b=batch_size, t=temporal)
        if temporal == 1:
            images = images.repeat_interleave(self.temporal_patch_size, dim=1)
            temporal = self.temporal_patch_size

        grid_t = temporal // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size

        images = images.reshape(
            batch_size * grid_t,
            self.temporal_patch_size,
            channel,
            -1
        )

        images = rearrange(images, 'b p c n -> b n (c p)')
        images = images.reshape(
            batch_size * grid_t,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
            -1
        )
        images = rearrange(images, 'b h k s1 w l s2 n -> (b h w k l) (n s1 s2)')

        return dict(image=images, image_grid_thw=torch.as_tensor([[grid_t, grid_h, grid_w] for _ in range(batch_size)]))


class SemanticEncoder(nn.Module):
    def __init__(self,
                 semantic_encoder,
                 z_channels=4,
                 double_z=False,
                 num_blocks=2,
                 embed_dim=1280,
                 proj_layer='linear',
                 attn_implementation='xformers',
                 target_mlp='norm',
                 ):
        super().__init__()
        self.embed_dim = embed_dim

        self.model = Qwen2VisionTransformerPretrainedModel.from_pretrained(
            semantic_encoder,
            attn_implementation=attn_implementation
        )
        input_channels = self.model.config.hidden_size

        for p in self.model.parameters():
            p.requires_grad = False

        self.proj_in = nn.Conv2d(input_channels, embed_dim, 1, 1) if input_channels != embed_dim else nn.Identity()

        config = Qwen2VLVisionConfig(depth=num_blocks, embed_dim=embed_dim)
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLBatchVisionBlock(config, attn_implementation) for _ in range(num_blocks)]
        )

        out_z_channels = 2 * z_channels if double_z else z_channels

        if proj_layer == 'norm_linear':
            self.proj_out = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(
                    embed_dim,
                    out_z_channels,
                )
            )
        elif proj_layer == 'linear':
            self.proj_out = nn.Sequential(
                nn.Linear(
                    embed_dim,
                    out_z_channels,
                )
            )
        elif proj_layer == 'mlp':
            self.proj_out = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, out_z_channels),
            )
        else:
            raise RuntimeError(f"Wrong proj layer. Got {proj_layer}")

        if target_mlp == 'identity':
            self.target_mlp = nn.Sequential(
                nn.Identity(),
            )
        elif target_mlp == 'norm':
            self.target_mlp = nn.Sequential(
                nn.LayerNorm(input_channels, eps=1e-6, elementwise_affine=False),
            )
        self.init_weight()

    def init_weight(self):
        self.proj_in.apply(init_weights)
        self.blocks.apply(init_weights)
        self.proj_out.apply(init_weights)
        self.target_mlp.apply(init_weights)

    def rot_pos_emb(self, grid_thw, max_seq_len):
        pos_ids = torch.zeros((len(grid_thw), max_seq_len, 2), dtype=torch.long)
        for idx, (t, h, w) in enumerate(grid_thw):
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.flatten()

            current_pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            pos_ids[idx, :current_pos_ids.shape[0]] = current_pos_ids
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(2)
        return rotary_pos_emb

    def forward(self, x, grid_thw):
        x = self.model(x, grid_thw=grid_thw)

        x = x_target = self.target_mlp(x)

        x = F.linear(x,
                     self.proj_in.weight.view(self.proj_in.weight.shape[0], -1),
                     self.proj_in.bias)

        new_grid_thw = torch.as_tensor([[t, h // 2, w // 2] for t, h, w in grid_thw])

        seq_lens = [t_i * h_i * w_i for t_i, h_i, w_i in new_grid_thw]
        max_seq_len = max(seq_lens)

        x = rearrange(x, '(b h w) c -> b (h w) c', h=new_grid_thw[0, 1], w=new_grid_thw[0, 2])

        rotary_pos_emb = self.rot_pos_emb(new_grid_thw, max_seq_len)

        for blk in self.blocks:
            x = blk(x, rotary_pos_emb=rotary_pos_emb)

        x = self.proj_out(x)  # [b, max_length, d]

        t, h, w = new_grid_thw[0]
        b = len(grid_thw)
        x = rearrange(x, 'b (h w) c ->b c h w', b=b, h=h, w=w)
        x_target = rearrange(x_target, '(b h w) c ->b c h w', b=b, h=h, w=w)
        return x, x_target


class SemanticDecoder(nn.Module):
    def __init__(self,
                 z_channels=4,
                 embed_dim=1280,
                 num_blocks=2,
                 output_channels=1280,
                 attn_implementation='xformers',
                 proj_layer='linear', ):
        super().__init__()
        self.proj_in = nn.Linear(z_channels, embed_dim)

        self.output_channels = output_channels
        config = Qwen2VLVisionConfig(depth=num_blocks, embed_dim=embed_dim)

        self.blocks = nn.ModuleList(
            [Qwen2VLBatchVisionBlock(config, attn_implementation) for _ in range(num_blocks)]
        )
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        if proj_layer == 'norm_linear':
            self.proj_out = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, output_channels),
            )
        elif proj_layer == 'linear':
            self.proj_out = nn.Sequential(
                nn.Linear(embed_dim, output_channels)
            )
        elif proj_layer == 'mlp':
            self.proj_out = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Tanh(),
                nn.Linear(embed_dim, output_channels),
            )
        elif proj_layer == 'linear_norm':
            self.proj_out = nn.Sequential(
                nn.Linear(embed_dim, output_channels),
                nn.LayerNorm(output_channels),
            )

        self.apply(init_weights)

    @property
    def last_layer(self):
        return self.proj_out[-1].weight

    def rot_pos_emb(self, grid_thw, max_seq_len):
        pos_ids = torch.zeros((len(grid_thw), max_seq_len, 2), dtype=torch.long)
        for idx, (t, h, w) in enumerate(grid_thw):
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.flatten()

            current_pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            pos_ids[idx, :current_pos_ids.shape[0]] = current_pos_ids
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(2)
        return rotary_pos_emb

    def forward(self, z: torch.Tensor):
        x = z

        b, c, h, w = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')

        grid_thw = torch.as_tensor([[1, h, w] for _ in range(b)])
        seq_lens = [t * h * w for t, h, w in grid_thw]
        max_seq_len = max(seq_lens)

        x = self.proj_in(x)

        rotary_pos_emb = self.rot_pos_emb(grid_thw, max_seq_len)

        for blk in self.blocks:
            x = blk(x, rotary_pos_emb=rotary_pos_emb)

        x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def get_movqgan_config(**kwargs):
    default_kwargs = {
        "attn_resolutions": [4],
        "ch": 256,
        "ch_mult": [1, 1, 2, 2, 4],
        "codebook_size": 32768,
        "double_z": False,
        "dropout": 0.0,
        "embed_dim": 4,
        "in_channels": 3,
        "model_type": "MoVQ",
        "num_res_blocks": 2,
        "out_channels": 3,
        "z_channels": 32
    }
    default_kwargs.update(kwargs)
    return MoVQConfig(**default_kwargs)


@MODELS.register_module()
class SingleVQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.semantic_encoder = SemanticEncoder(**config.semantic_encoder)
        self.semantic_quantizer = self.pixel_quantizer = build_quantizer(config.semantic_quantizer)
        self.semantic_decoder = SemanticDecoder(**config.semantic_decoder)

        self.pixel_post_quant_conv = nn.Conv2d(config.embed_dim, config.z_channels, 1)

        decoder_config = get_movqgan_config(config.pixel_decoder)
        self.decoder = MoVQDecoder(decoder_config)

        self.scaling_layer = ScalingLayerForQwen2ViT()

        print(f'Current model is: {__class__.__name__}, initialization finished.')
        if 'vq_ckpt' in self.config:
            checkpoint = torch.load(self.config.vq_ckpt,
                                    map_location='cuda')
            if self.config.get('use_vq_ckpt_ema', False):
                model_state = checkpoint["ema"]
            else:
                model_state = checkpoint["model"]
            msg = self.load_state_dict(model_state, strict=False)
            print(f"Loaded model from checkpoint: {self.config.vq_ckpt} MSG: {msg}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        return get_parameter_device(self)

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    @property
    def pixel_channel(self):
        return self._pixel_channel

    def encode(self, image: torch.FloatTensor):
        scale_output = self.scaling_layer(image)
        image, image_grid_thw = scale_output['image'], scale_output['image_grid_thw']

        h_semantic, target_semantic, before_proj_x = self.semantic_encoder(image, image_grid_thw,
                                                                           return_before_proj=True)
        quant_semantic, emb_loss_semantic, info_semantic = self.semantic_quantizer(h_semantic)

        return (quant_semantic, emb_loss_semantic, info_semantic, target_semantic), \
               (quant_semantic, emb_loss_semantic, info_semantic)

    def encode_semantic(self, image: torch.FloatTensor):
        scale_output = self.scaling_layer(image)
        image, image_grid_thw = scale_output['image'], scale_output['image_grid_thw']

        h_semantic, target_semantic = self.semantic_encoder(image, image_grid_thw)

        quant_semantic, emb_loss_semantic, info_semantic = self.semantic_quantizer(h_semantic.float())

        return quant_semantic, emb_loss_semantic, info_semantic, target_semantic

    def apply_noise(self, quant_semantic, quant_pixel, noise_type):
        batch_mask = (torch.rand(quant_pixel.shape[0], 1, 1, 1) > self.config['pixel_drop_batch_rate']).to(
            quant_pixel)

        if noise_type == "zero":
            mask = (torch.rand(quant_pixel.shape[0], 1, 1, 1) > self.config['pixel_drop_rate']).to(quant_pixel)
            quant_pixel = quant_pixel * mask
        elif noise_type == "random_simple":
            mask = (torch.rand(quant_pixel.shape[0], 1, quant_pixel.shape[2], quant_pixel.shape[3]) > self.config[
                'pixel_drop_rate']).to(quant_pixel)
            mask = batch_mask + (1 - batch_mask) * mask

            shuffled_quant = torch.cat([quant_pixel[-1:], quant_pixel[:-1]], dim=0).detach()
            quant_pixel = quant_pixel * mask + (1 - mask) * shuffled_quant

        return quant_semantic, quant_pixel

    def merge_quants(self, quant_semantic: torch.Tensor, quant_pixel = None, ):
        quant_semantic = F.interpolate(
            quant_semantic.float(), (16, 16), mode='bicubic').to(quant_semantic.dtype)
        return quant_semantic

    def decode(self, quant_semantic: torch.Tensor, quant_pixel: torch.Tensor = None):
        quant = self.merge_quants(quant_semantic, quant_pixel)
        quant2 = self.pixel_post_quant_conv(quant)

        x = self.decoder(quant2, quant)
        return x

    def decode_semantic(self, x: List[torch.Tensor]):
        x = self.semantic_decoder(x)
        return x

    def decode_code(self, pixel_indices):
        quant_pixel = self.pixel_quantizer.indices_to_codes(pixel_indices)
        return self.decode(quant_pixel)

    def forward(self, image: torch.FloatTensor):
        (quant_semantic, diff_semantic, _, target_semantic), \
        (quant_pixel, diff_pixel, _) = self.encode(image)
        dec = self.decode(quant_semantic, quant_pixel)
        dec_semantic = self.decode_semantic(quant_semantic)
        return (dec_semantic, diff_semantic, target_semantic), \
               (dec, diff_pixel)

    def get_input(self, batch):
        if isinstance(batch, list):
            batch = batch[0]

        image = batch['image']

        if image.dtype == torch.double:
            image = image.float()

        return dict(
            image=image.contiguous().cuda(non_blocking=True),
        )


@MODELS.register_module()
class DualViTok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self._semantic_channel = config.semantic_encoder.z_channels
        self._pixel_channel = config.pixel_encoder.z_channels

        # semantic
        self.semantic_encoder = SemanticEncoder(**config.semantic_encoder)
        self.semantic_quantizer = build_quantizer(config.semantic_quantizer)
        self.semantic_decoder = SemanticDecoder(**config.semantic_decoder)

        # quantizer
        encoder_config = get_movqgan_config(**config.pixel_encoder)
        self.pixel_encoder = MoVQEncoder(encoder_config)
        self.pixel_quant_conv = nn.Conv2d(config.pixel_encoder.z_channels, config.pixel_encoder.embed_dim, 1)

        self.pixel_quantizer = build_quantizer(config.pixel_quantizer)

        self.pixel_post_quant_conv = nn.Conv2d(config.pixel_decoder.embed_dim,
                                               config.pixel_decoder.z_channels, 1)

        decoder_config = get_movqgan_config(**config.pixel_decoder)
        self.pixel_decoder = MoVQDecoder(decoder_config)

        self.scaling_layer = ScalingLayerForQwen2ViT()

        print(f'Current model is: {__class__.__name__}, initialization finished.')
        if 'vq_ckpt' in self.config:
            checkpoint = torch.load(self.config.vq_ckpt,
                                    map_location='cuda')
            if self.config.get('use_vq_ckpt_ema', False):
                model_state = checkpoint["ema"]
            else:
                model_state = checkpoint["model"]
            msg = self.load_state_dict(model_state, strict=False)
            print(f"Loaded model from checkpoint: {self.config.vq_ckpt}. MSG: {msg}")

    @property
    def device(self):
        return get_parameter_device(self)

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    @property
    def pixel_channel(self):
        return self._pixel_channel

    @property
    def semantic_channel(self):
        return self._semantic_channel

    def encode(self, image: torch.FloatTensor, ):
        scale_output = self.scaling_layer(image)
        image, image_grid_thw, image_gen = scale_output['image'], scale_output['image_grid_thw'], image

        h_semantic, target_semantic = self.semantic_encoder(image, image_grid_thw)
        quant_semantic, emb_loss_semantic, info_semantic = self.semantic_quantizer(h_semantic.float())

        h_pixel = self.pixel_encoder(image_gen)
        h_pixel = self.pixel_quant_conv(h_pixel)

        quant_pixel, emb_loss_pixel, info_pixel = self.pixel_quantizer(h_pixel.float())

        return (quant_semantic, emb_loss_semantic, info_semantic, target_semantic), \
               (quant_pixel, emb_loss_pixel, info_pixel)

    def encode_code(self, *args, **kwargs):
        (_, _, semantic_indices, _), \
        (_, _, pixel_indices) = self.encode(*args, **kwargs)
        return semantic_indices, pixel_indices

    def indices_to_codes(self, semantic_indices, pixel_indices):
        quant_semantic = self.semantic_quantizer.indices_to_codes(semantic_indices)
        quant_pixel = self.pixel_quantizer.indices_to_codes(pixel_indices)
        return quant_semantic, quant_pixel

    def encode_semantic(self, image: torch.FloatTensor):
        scale_output = self.scaling_layer(image)
        image, image_grid_thw, image_gen = scale_output['image'], scale_output['image_grid_thw'], image

        h_semantic, target_semantic = self.semantic_encoder(image, image_grid_thw)
        quant_semantic, emb_loss_semantic, info_semantic = self.semantic_quantizer(h_semantic.float())
        return quant_semantic, emb_loss_semantic, info_semantic, target_semantic

    def apply_noise(self, quant_semantic, quant_pixel, noise_type):
        batch_mask = (torch.rand(quant_pixel.shape[0], 1, 1, 1) > self.config['pixel_drop_batch_rate']).to(
            quant_pixel)

        if noise_type == "zero":
            mask = (torch.rand(quant_pixel.shape[0], 1, quant_pixel.shape[2], quant_pixel.shape[3]) > self.config[
                'pixel_drop_rate']).to(quant_pixel)
            mask = batch_mask + (1 - batch_mask) * mask

            quant_pixel = quant_pixel * mask

        # elif noise_type == "random":
        #     mask = (torch.rand(quant_pixel.shape[0], 1, quant_pixel.shape[0], quant_pixel.shape[0]) > self.config[
        #         'pixel_drop_rate']).to(quant_pixel)
        #
        #     B, C, H, W = quant_pixel.shape
        #     flat_quant = quant_pixel.view(B, C, -1)
        #     rand_idx = torch.randint(0, H * W, (B, H, W), device=quant_pixel.device)
        #     rand_idx_flat = rand_idx.view(B, 1, -1).expand(B, C, -1)
        #     random_tokens = torch.gather(flat_quant, 2, rand_idx_flat).detach()
        #     random_tokens = random_tokens.view(B, C, H, W)
        #     quant_pixel = quant_pixel * mask + (1 - mask) * random_tokens

        elif noise_type == "random":
            mask = (torch.rand(quant_pixel.shape[0], 1, quant_pixel.shape[2], quant_pixel.shape[3]) > self.config[
                'pixel_drop_rate']).to(quant_pixel)
            mask = batch_mask + (1 - batch_mask) * mask

            # print(mask)
            shuffled_quant = torch.cat([quant_pixel[-1:], quant_pixel[:-1]], dim=0).detach()
            quant_pixel = quant_pixel * mask + (1 - mask) * shuffled_quant

        return quant_semantic, quant_pixel

    def merge_quants(self, quant_semantic: torch.Tensor, quant_pixel: torch.Tensor):
        """
        quant_semantic: [b, c, h, w],
        quant_pixel: [b, c, h, w],
        """
        if self.config.get('semantic_detail_merge_type', 'cat') == 'cat':
            quant_semantic_resized = F.interpolate(
                quant_semantic.float(), quant_pixel.shape[-2:], mode='bicubic'
            ).to(quant_semantic.dtype)
            quant_semantic = quant_semantic_resized

            if self.training and self.config.get('pixel_drop_rate', None):
                quant_semantic, quant_pixel = self.apply_noise(quant_semantic, quant_pixel,
                                                               self.config.get('pixel_noise_type', 'random'))

            quant = torch.cat([quant_semantic, quant_pixel], dim=1)
        else:
            if self.training and self.config.get('pixel_drop_rate', None):
                quant_semantic, quant_pixel = self.apply_noise(quant_semantic, quant_pixel,
                                                               self.config.get('pixel_noise_type', 'random'))

            quant = quant_pixel

        return quant

    def decode(self, quant_semantic: torch.Tensor, quant_pixel: torch.Tensor, ):
        quant = self.merge_quants(quant_semantic, quant_pixel)
        quant2 = self.pixel_post_quant_conv(quant)
        x = self.pixel_decoder(quant2, quant)
        return x

    def decode_code(self, semantic_indices, pixel_indices):
        quant_semantic = self.semantic_quantizer.indices_to_codes(semantic_indices)
        quant_pixel = self.pixel_quantizer.indices_to_codes(pixel_indices)
        return self.decode(quant_semantic, quant_pixel)

    def decode_semantic(self, x: List[torch.Tensor]):
        return self.semantic_decoder(x)

    def forward(self, image: torch.FloatTensor):
        (quant_semantic, diff_semantic, _, target_semantic), \
        (quant_pixel, diff_pixel, _) = self.encode(image)
        dec = self.decode(quant_semantic, quant_pixel)
        dec_semantic = self.decode_semantic(quant_semantic)
        return (dec_semantic, diff_semantic, target_semantic), (dec, diff_pixel)

    def get_input(self, batch):
        if isinstance(batch, list):
            batch = batch[0]

        image = batch['image']

        if image.dtype == torch.double:
            image = image.float()

        if image.ndim == 5:
            image = image[:, 0]

        return dict(
            image=image.contiguous().cuda(non_blocking=True),
        )


@MODELS.register_module()
class SemanticVQ(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.semantic_encoder = SemanticEncoder(**config.semantic_encoder)
        self.semantic_quantizer = build_quantizer(config.semantic_quantizer)
        self.decoder = SemanticDecoder(**config.semantic_decoder)

        self.scaling_layer = ScalingLayerForQwen2ViT()

        print(f'Current model is: {__class__.__name__}, initialization finished.')
        if 'vq_ckpt' in self.config:
            checkpoint = torch.load(self.config.vq_ckpt,
                                    map_location='cuda')
            if self.config.get('use_vq_ckpt_ema', False):
                model_state = checkpoint["ema"]
            else:
                model_state = checkpoint["model"]
            msg = self.load_state_dict(model_state, strict=False)
            print(f"Loaded model from checkpoint. MSG: {msg}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        return get_parameter_device(self)

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def encode(self, image: torch.FloatTensor):
        scale_output = self.scaling_layer(image)
        image, image_grid_thw = scale_output['image'], scale_output['image_grid_thw']

        h_semantic, target_semantic = self.semantic_encoder(image, image_grid_thw)

        quant_semantic, emb_loss_semantic, info_semantic = self.semantic_quantizer(h_semantic.float())

        return quant_semantic, emb_loss_semantic, info_semantic, target_semantic

    def encode_semantic(self, image: torch.FloatTensor):
        scale_output = self.scaling_layer(image)
        image, image_grid_thw = scale_output['image'], scale_output['image_grid_thw']

        h_semantic, target_semantic = self.semantic_encoder(image, image_grid_thw)

        quant_semantic, emb_loss_semantic, info_semantic = self.semantic_quantizer(h_semantic.float())

        return quant_semantic, emb_loss_semantic, info_semantic, target_semantic

    def decode(self, x: List[torch.Tensor]):
        quant = x
        quant2 = self.semantic_post_quant_conv(quant)
        x = self.decoder(quant2, quant)
        return x

    def decode_semantic(self, x: List[torch.Tensor]):
        quant = x
        quant2 = self.semantic_post_quant_conv(quant)
        x = self.decoder(quant2, quant)
        return x

    def decode_code(self, semantic_indices):
        quant_semantic = self.semantic_quantizer.indices_to_codes(semantic_indices)
        return self.decode(quant_semantic)

    def forward(self, image: torch.FloatTensor):
        quant_semantic, emb_loss_semantic, info_semantic, target_semantic = self.encode(image)
        dec = self.decode(quant_semantic)
        return (dec, emb_loss_semantic, target_semantic), \
               (torch.zeros_like(image), torch.zeros_like(emb_loss_semantic))

    def get_input(self, batch):
        if isinstance(batch, list):
            batch = batch[0]

        image = batch['image']

        if image.dtype == torch.double:
            image = image.float()

        return dict(
            image=image.contiguous().cuda(non_blocking=True),
        )

