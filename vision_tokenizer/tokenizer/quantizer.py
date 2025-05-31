import torch

from .vector_quantize_pytorch import ResidualVQ as _ResidualVQ
from .vector_quantize_pytorch import VectorQuantize as _VectorQuantize
from .vector_quantize_pytorch import SimVQ as _SimVQ
from .vector_quantize_pytorch import ResidualSimVQ as _ResidualSimVQ
from .vector_quantize_pytorch import LFQ as _LFQ
from .vector_quantize_pytorch import FSQ as _FSQ

from tokenizer.builder import QUANTIZERS


@QUANTIZERS.register_module()
class ResidualVQ(_ResidualVQ):
    def forward(self, *args, **kwargs):
        quantized, indices, commit_loss = super(ResidualVQ, self).forward(*args, **kwargs)
        return quantized, commit_loss.mean(), indices


@QUANTIZERS.register_module()
class VectorQuantize(_VectorQuantize):
    def forward(self, *args, **kwargs):
        quantized, indices, commit_loss = super(VectorQuantize, self).forward(*args, **kwargs)
        return quantized, commit_loss.mean(), indices


@QUANTIZERS.register_module()
class ResidualSimVQ(_ResidualSimVQ):
    def forward(self, *args, **kwargs):
        quantized, indices, commit_loss = super(ResidualSimVQ, self).forward(*args, **kwargs)
        return quantized, commit_loss.mean(), indices


@QUANTIZERS.register_module()
class SimVQ(_SimVQ):
    def forward(self, *args, **kwargs):
        quantized, indices, commit_loss = super(SimVQ, self).forward(*args, **kwargs)
        return quantized, commit_loss.mean(), indices


@QUANTIZERS.register_module()
class LFQ(_LFQ):
    def forward(self, *args, **kwargs):
        quantized, indices, commit_loss = super(LFQ, self).forward(*args, **kwargs)
        return quantized, commit_loss.mean(), indices


@QUANTIZERS.register_module()
class FSQ(_FSQ):
    def forward(self, *args, **kwargs):
        xhat, indices = super(FSQ, self).forward(*args, **kwargs)
        return xhat, torch.as_tensor(0), indices

