import torch.nn as nn

from .encoder import compute_channel_progression
from .layers import PermutedLayerNorm, ResidualBlock, make_conv1d


class DecoderBlock(nn.Module):
    """Optional upsampling + residual blocks + optional channel projection."""

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
        self.stride = stride

        if self.stride > 1:
            self.upsample = nn.Upsample(scale_factor=self.stride, mode="nearest")
            self.upsample_conv = make_conv1d(
                in_channels, in_channels, kernel_size=3, padding="same",
                use_weight_norm=use_weight_norm, causal=causal,
            )
            self.upsample_norm = PermutedLayerNorm(normalized_shape=in_channels)
            self.upsample_act = nn.SiLU()

        self.residual_blocks = nn.ModuleList()
        for i in range(num_residual_blocks):
            dilation = 2**i if use_dilation else 1
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels,
                    kernel_size=kernel_size,
                    num_units=residual_units_per_block,
                    dilation=dilation,
                    groups=groups,
                    use_weight_norm=use_weight_norm,
                    causal=causal,
                )
            )

        if in_channels != out_channels:
            self.channel_proj_conv = make_conv1d(
                in_channels, out_channels, kernel_size=1,
                use_weight_norm=use_weight_norm,
            )
            self.channel_proj_norm = PermutedLayerNorm(normalized_shape=out_channels)
            self.channel_proj_act = nn.SiLU()

    def forward(self, x):
        if hasattr(self, "upsample"):
            x = self.upsample(x)
            x = self.upsample_act(self.upsample_norm(self.upsample_conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        if hasattr(self, "channel_proj_conv"):
            x = self.channel_proj_act(
                self.channel_proj_norm(self.channel_proj_conv(x))
            )

        return x


class Decoder(nn.Module):
    """TCN decoder with DAC/SNAC-style progressive upsampling."""

    def __init__(
        self,
        output_features=100,
        first_channel_size=256,
        latent_dim=None,
        num_decoder_blocks=3,
        upsampling_rates=(2, 2, 1),
        kernel_size=(3, 3, 3),
        num_residual_blocks=2,
        residual_units_per_block=2,
        d_model=None,
        num_groups=False,
        use_weight_norm=False,
        causal=False,
    ):
        super().__init__()
        assert len(upsampling_rates) == num_decoder_blocks

        self.kernel_size = list(kernel_size)[::-1]
        self.upsampling_rates = list(upsampling_rates)

        if d_model is None:
            d_model = first_channel_size * 2

        # Mirror the encoder: compute its channel progression for the original
        # downsampling order, then walk it in reverse for the decoder blocks.
        downsampling_rates = list(upsampling_rates)[::-1]
        enc_channels = compute_channel_progression(
            first_channel_size, d_model, downsampling_rates,
        )
        # Decoder operates from the encoder's bottleneck back to first_channel_size:
        # block i has in_ch = enc_channels[-(i+1)], out_ch = enc_channels[-(i+2)].
        dec_channels = list(enc_channels[::-1])
        encoder_final_channels = dec_channels[0]
        if latent_dim is None:
            latent_dim = encoder_final_channels

        self.bottleneck_conv = make_conv1d(
            latent_dim, encoder_final_channels, kernel_size=1,
            use_weight_norm=use_weight_norm,
        )
        self.bottleneck_norm = PermutedLayerNorm(normalized_shape=encoder_final_channels)
        self.bottleneck_act = nn.SiLU()

        self.decoder_blocks = nn.ModuleList()
        for i in range(num_decoder_blocks):
            in_ch = dec_channels[i]
            out_ch = dec_channels[i + 1]
            use_dilation = i >= (num_decoder_blocks - 2)

            groups = 1
            if num_groups and i in [0, 1]:
                groups = in_ch  # depthwise on in_ch (residuals operate at in_ch)

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_residual_blocks=num_residual_blocks,
                    residual_units_per_block=residual_units_per_block,
                    stride=upsampling_rates[i],
                    kernel_size=self.kernel_size[i],
                    use_dilation=use_dilation,
                    groups=groups,
                    use_weight_norm=use_weight_norm,
                    causal=causal,
                )
            )

        self.final_conv = make_conv1d(
            dec_channels[-1], output_features, kernel_size=1,
            use_weight_norm=use_weight_norm,
        )

    def forward(self, encoded):
        x = self.bottleneck_act(self.bottleneck_norm(self.bottleneck_conv(encoded)))

        for block in self.decoder_blocks:
            x = block(x)

        return self.final_conv(x)
