# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.diffaug import DiffAug
from tokenizer.lpips import LPIPS
from tokenizer.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from tokenizer.discriminator_stylegan import Discriminator as StyleGANDiscriminator
from tokenizer.discriminator_dino import DinoDisc as DINODiscriminator

from tokenizer.builder import VQLOSSES
from einops import rearrange


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real), logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake), logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
          torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg


@VQLOSSES.register_module()
class DualViTokLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight=False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0,
                 codebook_weight=1.0, perceptual_weight=1.0,
                 codebook_enlarge_ratio=0, codebook_enlarge_steps=2000,
                 gen_loss_weight=1.0,

                 aug_prob=1,
                 norm_type='bn',

                 semantic_weight=1.0,
                 semantic_codebook_weight=1.0,
                 semantic_reconstruction_weight=1.0,
                 semantic_reconstruction='cos',
                 lecam_loss_weight=None):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan", 'dinodisc']
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        self.daug = None
        self.disc_type = disc_type
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels,
                image_size=image_size,
            )
        elif disc_type == "dinodisc":
            self.discriminator = DINODiscriminator(
                norm_type=norm_type,
                # input_nc=disc_in_channels,
                # image_size=image_size,
            )
            self.daug = DiffAug(prob=aug_prob, cutout=0.2)

        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight

        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        self.gen_loss_weight = gen_loss_weight
        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight is not None:
            self.lecam_ema = LeCAM_EMA()

        # codebook loss
        self.codebook_weight = codebook_weight
        self.codebook_enlarge_ratio = codebook_enlarge_ratio
        self.codebook_enlarge_steps = codebook_enlarge_steps

        self.semantic_weight = semantic_weight
        self.semantic_codebook_weight = semantic_codebook_weight
        self.semantic_reconstruction_weight = semantic_reconstruction_weight
        self.semantic_reconstruction = semantic_reconstruction

    def compute_semantic_loss(self, codebook_loss, x, xrec_semantic, **kwargs):
        if self.semantic_reconstruction == 'l1':
            loss_dict = self.compute_l1_loss(x, xrec_semantic, **kwargs)
        elif self.semantic_reconstruction == 'cos':
            loss_dict = self.compute_cosine_loss(x, xrec_semantic, **kwargs)
        elif self.semantic_reconstruction == 'cos_l1':
            loss_dict = self.compute_cos_l1_loss(x, xrec_semantic, **kwargs)
        elif self.semantic_reconstruction == 'cos_l2':
            loss_dict = self.compute_cos_l2_loss(x, xrec_semantic, **kwargs)
        else:
            raise RuntimeError(f"Unrecognized semantic reconstruction loss. Got: {self.semantic_reconstruction}")

        loss = self.semantic_codebook_weight * codebook_loss + \
               self.semantic_reconstruction_weight * sum(loss_dict.values())
        loss_dict['semantic_loss'] = loss
        loss_dict['semantic_codebook_loss'] = codebook_loss
        return self.semantic_weight * loss, loss_dict

    def compute_cos_l1_loss(self, x_semantic, xrec_semantic, compute_negative=False):
        loss = dict()
        loss.update(self.compute_cosine_loss(x_semantic, xrec_semantic, compute_negative=compute_negative))
        loss.update(self.compute_l1_loss(x_semantic, xrec_semantic))
        return loss

    def compute_cos_l2_loss(self, x_semantic, xrec_semantic, compute_negative=False):
        loss = dict()
        loss.update(self.compute_cosine_loss(x_semantic, xrec_semantic, compute_negative=compute_negative))
        loss.update(self.compute_l2_loss(x_semantic, xrec_semantic))
        return loss

    def compute_l1_loss(self, x_semantic, xrec_semantic):
        loss = torch.nn.functional.smooth_l1_loss(xrec_semantic, x_semantic.detach())
        return dict(semantic_l1_loss=loss)

    def compute_l2_loss(self, x_semantic, xrec_semantic):
        loss = torch.nn.functional.mse_loss(xrec_semantic, x_semantic.detach())
        return dict(semantic_l2_loss=loss)

    def compute_cosine_loss(self, x_semantic, xrec_semantic):
        if x_semantic.ndim == 4:
            x_semantic_flat = rearrange(x_semantic, 'b c h w -> (b h w) c')
        elif x_semantic.ndim == 2:
            x_semantic_flat = x_semantic
        else:
            raise RuntimeError(f"Unrecognized x_semantic dimension. Got {x_semantic.ndim} dim.")

        if xrec_semantic.ndim == 4:
            xrec_semantic_flat = rearrange(xrec_semantic, 'b c h w -> (b h w) c')
        elif xrec_semantic.ndim == 2:
            xrec_semantic_flat = xrec_semantic
        else:
            raise RuntimeError(f"Unrecognized xrec_semantic dimension. Got {xrec_semantic.ndim} dim.")

        def l2_norm(x):
            return x / (x.norm(dim=-1, keepdim=True))

        x_semantic_flat = l2_norm(x_semantic_flat)
        xrec_semantic_flat = l2_norm(xrec_semantic_flat)

        bhw = x_semantic_flat.size(0)

        target_positive = torch.ones(bhw, ).to(x_semantic.device)

        cosine_loss = F.cosine_embedding_loss(
            x_semantic_flat,
            xrec_semantic_flat,
            target_positive,
            margin=1e-4,
        )

        return dict(semantic_cos_loss=cosine_loss)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self,
                codebook_loss,
                inputs,
                reconstructions,
                optimizer_idx, global_step, last_layer=None,
                logger=None, log_every=100, fade_blur_schedule=0):
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = torch.mean(p_loss)

            # discriminator loss
            if self.disc_type == "dinodisc":
                if fade_blur_schedule < 1e-6:
                    fade_blur_schedule = 0
                logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous(), fade_blur_schedule))
            else:
                logits_fake = self.discriminator(reconstructions.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)

            if self.gen_loss_weight is None:
                disc_adaptive_weight = torch.tensor(self.gen_loss_weight)
            else:
                if self.disc_adaptive_weight:
                    null_loss = self.rec_weight * rec_loss + self.perceptual_weight * p_loss
                    disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss,
                                                                          last_layer=last_layer)
                else:
                    disc_adaptive_weight = 1

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)

            scale_codebook_loss = self.codebook_weight * codebook_loss  # entropy_loss
            if self.codebook_enlarge_ratio > 0:
                scale_codebook_loss = self.codebook_enlarge_ratio * (
                    max(0, 1 - global_step / self.codebook_enlarge_steps)) * scale_codebook_loss + scale_codebook_loss

            loss = self.rec_weight * rec_loss + \
                   self.perceptual_weight * p_loss + \
                   disc_adaptive_weight * disc_weight * generator_adv_loss + \
                   scale_codebook_loss

            if global_step % log_every == 0:
                rec_loss = self.rec_weight * rec_loss
                p_loss = self.perceptual_weight * p_loss
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
                            f"vq_loss: {codebook_loss:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, "
                            # f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}"
                            )
            loss_dict = {
                'rec_loss': rec_loss.detach().float(),
                'p_loss': p_loss.detach().float(),
                'vq_loss': codebook_loss.detach().float(),
                'generator_adv_loss': generator_adv_loss.detach().float(),
                # 'disc_adaptive_weight': disc_adaptive_weight,
                # 'disc_weight': disc_weight,
            }
            return loss, loss_dict

        # discriminator update
        if optimizer_idx == 1:
            if self.disc_type == "dinodisc":
                if fade_blur_schedule < 1e-6:
                    fade_blur_schedule = 0
                # add blur since disc is too strong
                logits_fake = self.discriminator(
                    self.daug.aug(reconstructions.contiguous().detach(), fade_blur_schedule))
                logits_real = self.discriminator(self.daug.aug(inputs.contiguous().detach(), fade_blur_schedule))
            else:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)

            if self.lecam_loss_weight is not None and disc_weight > 0:  ## update only when disc_factor > 0
                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
                non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
                d_adversarial_loss = disc_weight * (lecam_loss * self.lecam_loss_weight + non_saturate_d_loss)
            else:
                d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)

            if global_step % log_every == 0:
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                logger.info(f"(Discriminator) "
                            f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}")

            loss_dict = {
                'discriminator_adv_loss': d_adversarial_loss.detach().float(),
                'logits_real': logits_real.detach().float(),
                'logits_fake': logits_fake.detach().float(),
                # 'disc_weight': disc_weight,
            }
            return d_adversarial_loss, loss_dict
