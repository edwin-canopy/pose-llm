import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    """Conv1d with left-only temporal padding."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        use_weight_norm=False,
    ):
        super().__init__()
        kernel_width = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        self.pad_left = (kernel_width - 1) * dilation
        conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad_left, 0)))


def make_conv1d(
    *args,
    use_weight_norm: bool = False,
    causal: bool = False,
    **kwargs,
) -> nn.Module:
    """Create a Conv1d with optional weight norm and causal padding.

    The non-causal path returns a bare Conv1d, preserving legacy state_dict
    names. Causal mode wraps only kernel>1 convolutions because 1x1 convs are
    already causal.
    """
    kernel_size = kwargs.get("kernel_size", args[2] if len(args) > 2 else 1)
    kernel_width = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
    if causal and kernel_width > 1:
        kwargs.pop("padding", None)
        return CausalConv1d(*args, use_weight_norm=use_weight_norm, **kwargs)

    conv = nn.Conv1d(*args, **kwargs)
    return weight_norm(conv) if use_weight_norm else conv


def WNConv1d(*args, **kwargs):
    return make_conv1d(*args, use_weight_norm=True, **kwargs)


class PermutedLayerNorm(nn.Module):
    """LayerNorm on the channel dim of (N, C, L) tensors."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )

    def forward(self, x):
        return self.layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)


class ResidualUnit(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=1,
        groups=1,
        use_weight_norm=False,
        causal=False,
    ):
        super().__init__()

        self.conv1 = make_conv1d(
            channels, channels, kernel_size=kernel_size,
            padding="same", dilation=dilation, groups=groups,
            use_weight_norm=use_weight_norm, causal=causal,
        )
        self.norm1 = PermutedLayerNorm(normalized_shape=channels)
        self.act1 = nn.SiLU()

        self.conv2 = make_conv1d(
            channels, channels, kernel_size=kernel_size,
            padding="same", dilation=1, groups=groups,
            use_weight_norm=use_weight_norm, causal=causal,
        )
        self.norm2 = PermutedLayerNorm(normalized_shape=channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class ResidualBlock(nn.Module):
    """Stack of ResidualUnits with a skip connection."""

    def __init__(
        self,
        channels,
        kernel_size=3,
        num_units=2,
        dilation=1,
        groups=1,
        use_weight_norm=False,
        causal=False,
    ):
        super().__init__()
        self.units = nn.ModuleList(
            [
                ResidualUnit(
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    groups=groups,
                    use_weight_norm=use_weight_norm,
                    causal=causal,
                )
                for _ in range(num_units)
            ]
        )
        self.act_final = nn.SiLU()

    def forward(self, x):
        residual = x
        for unit in self.units:
            x = unit(x)
        return self.act_final(x + residual)
