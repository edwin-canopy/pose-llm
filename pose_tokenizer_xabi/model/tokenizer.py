from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from ..config import PoseTokenizerConfig
from .encoder import Encoder, compute_channel_progression
from .decoder import Decoder
from .vq import ResidualVectorQuantize


def _init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        # NOTE: zero-bias init on Conv1d turns out to act as an *implicit*
        # regulariser via the PermutedLayerNorm backward path on padded
        # frames. Removing it (using default PyTorch bias init) collapsed
        # codebook utilisation to <5% in early experiments, so we keep the
        # explicit zero-init even though it looks redundant.
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PoseTokenizerModel(
    nn.Module,
    PyTorchModelHubMixin,
    tags=["pose-tokenizer", "keypoints", "vqvae"],
    repo_url="https://github.com/canopy-labs-internal/pose-tokenizer.git",
    license="mit",
):
    """
    VQ-VAE that tokenises body-pose keypoints into discrete codes.

    Architecture ported from FaceTokenizer (TCN encoder/decoder + RVQ).

    HuggingFace integration via PyTorchModelHubMixin gives you:
        model = PoseTokenizerModel.from_pretrained("your-hf-repo")
        model.save_pretrained("./local-dir")
        model.push_to_hub("your-hf-repo")
    """

    def __init__(self, config: dict | PoseTokenizerConfig | None = None, **kwargs):
        super().__init__()

        if config is None:
            config = PoseTokenizerConfig(**kwargs)
        elif isinstance(config, dict):
            config = PoseTokenizerConfig(**{
                k: v for k, v in config.items()
                if k in PoseTokenizerConfig.__dataclass_fields__
            })

        self.config = config
        c = config

        self.input_features = c.input_features
        self.enable_quantization = c.enable_quantization

        self.downsampling_factor = 1
        for rate in c.downsampling_rates:
            self.downsampling_factor *= rate
        d_model = c.d_model if c.d_model is not None else c.first_channel_size * 2
        encoder_final_dim = compute_channel_progression(
            c.first_channel_size, d_model, c.downsampling_rates,
        )[-1]
        self.latent_dim = c.latent_dim if c.latent_dim is not None else encoder_final_dim

        self.encoder = Encoder(
            input_features=c.input_features,
            first_channel_size=c.first_channel_size,
            latent_dim=self.latent_dim,
            num_encoder_blocks=c.num_blocks,
            kernel_size=c.kernel_size,
            downsampling_rates=c.downsampling_rates,
            num_residual_blocks=c.num_residual_blocks,
            residual_units_per_block=c.residual_units_per_block,
            d_model=c.d_model,
            num_groups=c.num_groups,
            use_weight_norm=c.conv_weight_norm,
            causal=c.causal,
            dilation_mode="deep" if c.encoder_deep_dilation else c.encoder_dilation,
        )

        upsampling_rates = list(c.downsampling_rates)[::-1]

        self.decoder = Decoder(
            output_features=c.input_features,
            first_channel_size=c.first_channel_size,
            latent_dim=self.latent_dim,
            num_decoder_blocks=c.num_blocks,
            upsampling_rates=upsampling_rates,
            num_residual_blocks=c.num_residual_blocks,
            residual_units_per_block=c.residual_units_per_block,
            kernel_size=c.kernel_size,
            d_model=c.d_model,
            num_groups=c.num_groups,
            use_weight_norm=c.conv_weight_norm,
            causal=c.causal,
        )

        if c.enable_quantization:
            self.quantizer = ResidualVectorQuantize(
                input_dim=self.latent_dim,
                n_codebooks=c.n_codebooks,
                codebook_size=c.codebook_size,
                codebook_dim=c.codebook_dim,
                vq_strides=c.vq_strides,
                apply_rotation_trick=c.apply_rotation_trick,
                quantizer_dropout=c.quantizer_dropout,
                normalize_codes=c.normalize_codes,
                dead_code_threshold=c.dead_code_threshold,
                dead_code_ema_decay=c.dead_code_ema_decay,
            )
        else:
            self.quantizer = None

        self._active_n_codebooks = c.n_codebooks

        self.apply(_init_weights)

    # ------------------------------------------------------------------
    # Preprocessing: (B, T, F) <-> (B, F, T)
    # ------------------------------------------------------------------

    def preprocess(self, x):
        return x.permute(0, 2, 1)

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Encode / Decode / Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        """(B, T, F) -> quantised latent + codes + losses."""
        x = self.preprocess(x)
        z = self.encoder(x)

        if self.quantizer is not None:
            z_q, codes, commitment_loss, codebook_loss = self.quantizer(z)
            return z_q, codes, commitment_loss, codebook_loss

        zero = torch.tensor(0.0, device=z.device)
        return z, None, zero, zero

    def decode(self, codes: list[torch.Tensor], n_codebooks: int | None = None):
        """Reconstruct from discrete codes."""
        if n_codebooks is None:
            n_codebooks = self._active_n_codebooks
        z_q = self.quantizer.from_codes(codes, n_quantizers=n_codebooks)
        return self.postprocess(self.decoder(z_q))

    def forward(self, keypoints: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full encode -> (quantize) -> decode pass.

        Args:
            keypoints: (B, T, F) raw keypoint features.

        Returns:
            Dict with ``reconstruction``, ``codes``, ``commitment_loss``,
            ``codebook_loss``.
        """
        x = self.preprocess(keypoints)
        z = self.encoder(x)

        if self.quantizer is not None:
            z_q, codes, commitment_loss, codebook_loss = self.quantizer(z)
        else:
            z_q = z
            codes = None
            commitment_loss = torch.tensor(0.0, device=z.device)
            codebook_loss = torch.tensor(0.0, device=z.device)

        reconstruction = self.postprocess(self.decoder(z_q))

        return {
            "reconstruction": reconstruction,
            "codes": codes,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    def set_active_codebooks(self, n: int):
        if n > self.config.n_codebooks:
            raise ValueError(
                f"Cannot set {n} active codebooks, model only has {self.config.n_codebooks}"
            )
        self._active_n_codebooks = n

    def tokenize(self, keypoints: torch.Tensor) -> list[torch.Tensor]:
        """keypoints -> discrete codes."""
        _, codes, _, _ = self.encode(keypoints)
        return codes

    def detokenize(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """discrete codes -> reconstructed keypoints."""
        return self.decode(codes)
