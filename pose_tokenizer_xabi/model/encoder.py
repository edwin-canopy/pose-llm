import torch.nn as nn

from .layers import PermutedLayerNorm, ResidualBlock, make_conv1d


class EncoderBlock(nn.Module):
    """Residual blocks + optional strided downsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_residual_blocks=2,
        residual_units_per_block=2,
        stride=1,
        use_dilation=True,
        kernel_size=3,
        groups=1,
        use_weight_norm=False,
        causal=False,
    ):
        super().__init__()

        if in_channels != out_channels:
            self.channel_proj_conv = make_conv1d(
                in_channels, out_channels, kernel_size=1,
                use_weight_norm=use_weight_norm,
            )
            self.channel_proj_norm = PermutedLayerNorm(normalized_shape=out_channels)
            self.channel_proj_act = nn.SiLU()

        self.residual_blocks = nn.ModuleList()
        for i in range(num_residual_blocks):
            dilation = 2**i if use_dilation else 1
            self.residual_blocks.append(
                ResidualBlock(
                    out_channels,
                    kernel_size=kernel_size,
                    num_units=residual_units_per_block,
                    dilation=dilation,
                    groups=groups,
                    use_weight_norm=use_weight_norm,
                    causal=causal,
                )
            )

        if stride > 1:
            self.downsample_conv = make_conv1d(
                out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                use_weight_norm=use_weight_norm, causal=causal,
            )
            self.downsample_norm = PermutedLayerNorm(normalized_shape=out_channels)
            self.downsample_act = nn.SiLU()

    def forward(self, x):
        if hasattr(self, "channel_proj_conv"):
            x = self.channel_proj_act(self.channel_proj_norm(self.channel_proj_conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        if hasattr(self, "downsample_conv"):
            x = self.downsample_act(self.downsample_norm(self.downsample_conv(x)))

        return x


def compute_channel_progression(
    first_channel_size: int,
    d_model: int,
    downsampling_rates,
) -> list[int]:
    """Channel size BEFORE each encoder block + final size AFTER the last block.

    DAC/SNAC-style: channels double at every downsampling stage (stride > 1),
    capped at ``d_model``. Stride-1 blocks keep the channel count unchanged.
    Returns a list of length ``len(downsampling_rates) + 1``.
    """
    channels = [first_channel_size]
    cur = first_channel_size
    for stride in downsampling_rates:
        if stride > 1:
            cur = min(cur * 2, d_model)
        channels.append(cur)
    return channels


class Encoder(nn.Module):
    """TCN encoder with DAC/SNAC-style progressive downsampling."""

    def __init__(
        self,
        input_features=100,
        first_channel_size=256,
        latent_dim=None,
        num_encoder_blocks=3,
        downsampling_rates=(1, 2, 2),
        kernel_size=(3, 3, 3),
        num_residual_blocks=2,
        residual_units_per_block=2,
        d_model=None,
        num_groups=False,
        use_weight_norm=False,
        causal=False,
        dilation_mode="shallow",
    ):
        super().__init__()
        assert len(downsampling_rates) == num_encoder_blocks
        assert dilation_mode in ("none", "shallow", "deep"), dilation_mode
        self.dilation_mode = dilation_mode

        self.init_conv = make_conv1d(
            input_features, first_channel_size, kernel_size=1,
            use_weight_norm=use_weight_norm,
        )
        self.init_norm = PermutedLayerNorm(normalized_shape=first_channel_size)
        self.init_act = nn.SiLU()

        self.downsampling_rates = list(downsampling_rates)

        if d_model is None:
            d_model = first_channel_size * 2

        channel_progression = compute_channel_progression(
            first_channel_size, d_model, self.downsampling_rates,
        )
        if latent_dim is None:
            latent_dim = channel_progression[-1]

        self.encoder_blocks = nn.ModuleList()
        for i in range(num_encoder_blocks):
            in_ch = channel_progression[i]
            out_ch = channel_progression[i + 1]
            # Dilation placement controls the temporal receptive field:
            #   "shallow" (default): dilate early blocks (i < 2) — legacy.
            #   "deep": dilate the deepest blocks (mirrors the decoder); largest RF.
            #   "none": no dilation anywhere -> narrow RF (<< segment length), so
            #           most frames see only local context (intentional for causal).
            if self.dilation_mode == "none":
                use_dilation = False
            elif self.dilation_mode == "deep":
                use_dilation = i >= (num_encoder_blocks - 2)
            else:  # "shallow"
                use_dilation = i < 2

            groups = 1
            if num_groups and i in [1, 2]:
                groups = out_ch  # depthwise groups must divide out_ch

            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_residual_blocks=num_residual_blocks,
                    residual_units_per_block=residual_units_per_block,
                    stride=downsampling_rates[i],
                    kernel_size=kernel_size[i],
                    use_dilation=use_dilation,
                    groups=groups,
                    use_weight_norm=use_weight_norm,
                    causal=causal,
                )
            )

        self.bottleneck_conv = make_conv1d(
            channel_progression[-1], latent_dim, kernel_size=1,
            use_weight_norm=use_weight_norm,
        )

    def forward(self, x):
        x = self.init_act(self.init_norm(self.init_conv(x)))

        for block in self.encoder_blocks:
            x = block(x)

        return self.bottleneck_conv(x)
