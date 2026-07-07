"""Quantization-aware layers for the spiking SDCAE.

Learned Step-Size Quantization (LSQ) 1D conv / transposed-conv and the
integer ``MultiSpike`` activation. Extracted once here and shared by the
SDCAE model instead of being duplicated in every model file.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._quant_base import Qmodes, _ActQ, _Conv1dQ, _ConvTranspose1dQ


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class ActLSQ(_ActQ):
    """Activation placeholder kept for checkpoint/state-dict compatibility."""

    def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise, **kwargs):
        super().__init__(in_features=in_features, nbits=nbits_a, mode=mode)

    def forward(self, x):
        return x


class Conv1dLSQ(_Conv1dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4,
                 mode=Qmodes.kernel_wise, **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        self.act = ActLSQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.conv1d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        alpha = grad_scale(self.alpha, g).unsqueeze(1).unsqueeze(2)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        x = self.act(x)
        return F.conv1d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvTranspose1dLSQ(_ConvTranspose1dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 nbits_w=4, mode=Qmodes.kernel_wise, **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias, dilation=dilation, nbits=nbits_w, mode=mode)
        self.act = ActLSQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x, output_size=None):
        if self.alpha is None:
            return F.conv_transpose1d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilation)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        alpha = grad_scale(self.alpha, g).unsqueeze(0).unsqueeze(2)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        x = self.act(x)
        return F.conv_transpose1d(
            x, w_q, self.bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)


class _Quant(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        (i,) = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class MultiSpike(nn.Module):
    """Integer multi-level spike activation in ``[min_value, max_value]``."""

    def __init__(self, min_value=0, max_value=4, Norm=None):
        super().__init__()
        self.Norm = max_value if Norm is None else Norm
        self.min_value = min_value
        self.max_value = max_value

    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"

    def forward(self, x):
        return _Quant.apply(x, self.min_value, self.max_value) / self.Norm
