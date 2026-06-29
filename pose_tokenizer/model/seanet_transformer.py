"""Self-contained SEANet (conv) + causal-transformer backend for the pose tokenizer.

This is a Mimi-style encoder/decoder: a SEANet convolutional front-end, a causal
Transformer bottleneck operating at the input frame rate, and a separate causal
resample (down on the encode side / up on the decode side) that sets the token
rate. It is a clean re-implementation built on this repo's existing causal-conv
primitives (``pose_tokenizer.model.layers.make_conv1d``) plus PyTorch's
``scaled_dot_product_attention(is_causal=True)`` and interleaved RoPE -- rather
than a vendor of moshi's streaming framework -- so causality is easy to prove
(see ``tests``) and there are no audio-only dependencies.

Tensor contract (matches the conv ``Encoder``/``Decoder`` so the rest of
``PoseTokenizerModel`` is untouched):
    encoder: (B, input_features, T)          -> (B, latent_dim, T // downsample)
    decoder: (B, latent_dim, T // downsample) -> (B, input_features, T)

Everything is strictly causal: SEANet convs left-pad only, the transformer uses
a causal attention mask, the downsample is a causal strided conv, and the
upsample is a transposed conv whose right (future) tail is trimmed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from .layers import make_conv1d


# ---------------------------------------------------------------------------
# Causal resample primitives
# ---------------------------------------------------------------------------

class CausalConvTranspose1d(nn.Module):
    """Transposed conv upsampler made causal by trimming the right ``K - S`` samples.

    ``ConvTranspose1d`` with padding=0 produces ``(T-1)*stride + kernel`` outputs;
    dropping the trailing ``kernel - stride`` removes the only samples that depend
    on future inputs, leaving exactly ``T * stride`` causal outputs (matches the
    EnCodec/moshi ``unpad1d(0, K-S)`` convention).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 use_weight_norm=False):
        super().__init__()
        convtr = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.convtr = weight_norm(convtr) if use_weight_norm else convtr
        self.trim = kernel_size - stride

    def forward(self, x):
        y = self.convtr(x)
        if self.trim > 0:
            y = y[..., : -self.trim]
        return y


# ---------------------------------------------------------------------------
# SEANet conv front/back end (causal)
# ---------------------------------------------------------------------------

class SEANetResnetBlock(nn.Module):
    """SEANet residual block: ELU + (kernel, 1) causal convs with a true skip."""

    def __init__(self, dim, kernel_sizes=(3, 1), dilations=(1, 1),
                 compress=2, causal=True, use_weight_norm=False):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        hidden = max(1, dim // compress)
        layers: list[nn.Module] = []
        for i, (k, d) in enumerate(zip(kernel_sizes, dilations)):
            in_c = dim if i == 0 else hidden
            out_c = dim if i == len(kernel_sizes) - 1 else hidden
            layers += [
                nn.ELU(),
                make_conv1d(in_c, out_c, kernel_size=k, dilation=d,
                            causal=causal, use_weight_norm=use_weight_norm),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class SEANetEncoder(nn.Module):
    """Causal SEANet encoder. ``ratios`` are temporal strides (1 = no downsample)."""

    def __init__(self, channels, dimension, n_filters=64, n_residual_layers=1,
                 ratios=(1,), kernel_size=7, residual_kernel=3, dilation_base=2,
                 compress=2, causal=True, use_weight_norm=False):
        super().__init__()
        ratios = list(reversed(list(ratios)))
        mult = 1
        model: list[nn.Module] = [
            make_conv1d(channels, n_filters, kernel_size=kernel_size,
                        causal=causal, use_weight_norm=use_weight_norm)
        ]
        for ratio in ratios:
            for j in range(n_residual_layers):
                model.append(
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=(residual_kernel, 1),
                        dilations=(dilation_base ** j, 1),
                        compress=compress, causal=causal,
                        use_weight_norm=use_weight_norm,
                    )
                )
            model += [
                nn.ELU(),
                make_conv1d(mult * n_filters, mult * n_filters * 2,
                            kernel_size=ratio * 2, stride=ratio,
                            causal=causal, use_weight_norm=use_weight_norm),
            ]
            mult *= 2
        model += [
            nn.ELU(),
            make_conv1d(mult * n_filters, dimension, kernel_size=kernel_size,
                        causal=causal, use_weight_norm=use_weight_norm),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """Causal SEANet decoder mirroring :class:`SEANetEncoder`."""

    def __init__(self, channels, dimension, n_filters=64, n_residual_layers=1,
                 ratios=(1,), kernel_size=7, residual_kernel=3, dilation_base=2,
                 compress=2, causal=True, use_weight_norm=False):
        super().__init__()
        ratios = list(ratios)
        mult = int(2 ** len(ratios))
        model: list[nn.Module] = [
            make_conv1d(dimension, mult * n_filters, kernel_size=kernel_size,
                        causal=causal, use_weight_norm=use_weight_norm)
        ]
        for ratio in ratios:
            model += [
                nn.ELU(),
                CausalConvTranspose1d(mult * n_filters, mult * n_filters // 2,
                                      kernel_size=ratio * 2, stride=ratio,
                                      use_weight_norm=use_weight_norm),
            ]
            for j in range(n_residual_layers):
                model.append(
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=(residual_kernel, 1),
                        dilations=(dilation_base ** j, 1),
                        compress=compress, causal=causal,
                        use_weight_norm=use_weight_norm,
                    )
                )
            mult //= 2
        model += [
            nn.ELU(),
            make_conv1d(n_filters, channels, kernel_size=kernel_size,
                        causal=causal, use_weight_norm=use_weight_norm),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        return self.model(z)


# ---------------------------------------------------------------------------
# Causal transformer bottleneck (RoPE + SDPA causal mask)
# ---------------------------------------------------------------------------

def _apply_rope(q: torch.Tensor, k: torch.Tensor, max_period: float = 10000.0):
    """Interleaved RoPE on q,k of shape (B, H, T, D) with D even.

    Positions are absolute 0..T-1 (whole-sequence training). Causality comes
    from the attention mask, not from RoPE.
    """
    B, H, T, D = q.shape
    assert D % 2 == 0
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2.0 / D))      # (D/2,)
    t = torch.arange(T, device=q.device, dtype=torch.float32).view(1, 1, T, 1)
    ang = t * freqs.view(1, 1, 1, -1)                              # (1,1,T,D/2)
    cos, sin = torch.cos(ang), torch.sin(ang)

    def rot(x):
        x = x.float().view(B, H, T, D // 2, 2)
        xr, xi = x[..., 0], x[..., 1]
        out = torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1)
        return out.view(B, H, T, D)

    return rot(q).to(q.dtype), rot(k).to(k.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope_max_period=10000.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_max_period = rope_max_period
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # (3, B, H, T, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _apply_rope(q, k, self.rope_max_period)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward,
                 layer_scale=0.01, rope_max_period=10000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, rope_max_period)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_feedforward)
        if layer_scale and layer_scale > 0:
            self.ls1 = nn.Parameter(layer_scale * torch.ones(d_model))
            self.ls2 = nn.Parameter(layer_scale * torch.ones(d_model))
        else:
            self.ls1 = self.ls2 = None

    def forward(self, x):
        a = self.attn(self.norm1(x))
        x = x + (a * self.ls1 if self.ls1 is not None else a)
        f = self.ff(self.norm2(x))
        x = x + (f * self.ls2 if self.ls2 is not None else f)
        return x


class CausalTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward,
                 layer_scale=0.01, rope_max_period=10000.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward,
                             layer_scale, rope_max_period)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):          # (B, T, d_model)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder wrappers (the public contract)
# ---------------------------------------------------------------------------

class SeanetTransformerEncoder(nn.Module):
    """(B, F, T) -> (B, latent_dim, T // downsample), strictly causal."""

    def __init__(self, input_features, latent_dim, *, seanet_dim, n_filters,
                 n_residual_layers, ratios, seanet_kernel, residual_kernel,
                 compress, d_model, num_heads, num_layers, dim_feedforward,
                 layer_scale, rope_max_period, downsample, causal=True,
                 use_weight_norm=True):
        super().__init__()
        self.downsample = downsample
        self.seanet = SEANetEncoder(
            channels=input_features, dimension=seanet_dim, n_filters=n_filters,
            n_residual_layers=n_residual_layers, ratios=ratios,
            kernel_size=seanet_kernel, residual_kernel=residual_kernel,
            compress=compress, causal=causal, use_weight_norm=use_weight_norm,
        )
        self.in_proj = nn.Conv1d(seanet_dim, d_model, kernel_size=1)
        self.transformer = CausalTransformer(
            d_model, num_heads, num_layers, dim_feedforward,
            layer_scale=layer_scale, rope_max_period=rope_max_period,
        )
        # Causal strided conv: kernel = 2*stride, left-padded -> length ceil(T/stride).
        self.down = make_conv1d(
            d_model, d_model, kernel_size=2 * downsample, stride=downsample,
            causal=causal, use_weight_norm=use_weight_norm,
        ) if downsample > 1 else nn.Identity()
        self.out_proj = nn.Conv1d(d_model, latent_dim, kernel_size=1)

    def forward(self, x):
        z = self.seanet(x)                       # (B, seanet_dim, T)
        z = self.in_proj(z)                      # (B, d_model, T)
        z = self.transformer(z.transpose(1, 2)).transpose(1, 2)
        z = self.down(z)                         # (B, d_model, T//ds)
        return self.out_proj(z)                  # (B, latent_dim, T//ds)


class SeanetTransformerDecoder(nn.Module):
    """(B, latent_dim, T // upsample) -> (B, F, T), strictly causal."""

    def __init__(self, output_features, latent_dim, *, seanet_dim, n_filters,
                 n_residual_layers, ratios, seanet_kernel, residual_kernel,
                 compress, d_model, num_heads, num_layers, dim_feedforward,
                 layer_scale, rope_max_period, upsample, causal=True,
                 use_weight_norm=True):
        super().__init__()
        self.upsample = upsample
        self.in_proj = nn.Conv1d(latent_dim, d_model, kernel_size=1)
        self.up = CausalConvTranspose1d(
            d_model, d_model, kernel_size=2 * upsample, stride=upsample,
            use_weight_norm=use_weight_norm,
        ) if upsample > 1 else nn.Identity()
        self.transformer = CausalTransformer(
            d_model, num_heads, num_layers, dim_feedforward,
            layer_scale=layer_scale, rope_max_period=rope_max_period,
        )
        self.out_proj = nn.Conv1d(d_model, seanet_dim, kernel_size=1)
        self.seanet = SEANetDecoder(
            channels=output_features, dimension=seanet_dim, n_filters=n_filters,
            n_residual_layers=n_residual_layers, ratios=ratios,
            kernel_size=seanet_kernel, residual_kernel=residual_kernel,
            compress=compress, causal=causal, use_weight_norm=use_weight_norm,
        )

    def forward(self, z_q):
        z = self.in_proj(z_q)                    # (B, d_model, T//ds)
        z = self.up(z)                           # (B, d_model, T)
        z = self.transformer(z.transpose(1, 2)).transpose(1, 2)
        z = self.out_proj(z)                     # (B, seanet_dim, T)
        return self.seanet(z)                    # (B, F, T)
