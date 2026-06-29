from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import WNConv1d


def _ema_inplace(moving_avg: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    moving_avg.mul_(decay).add_(new, alpha=(1.0 - decay))


def _sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    """Pick ``num`` rows from ``samples`` (N, D), with replacement if N < num."""
    n = samples.shape[0]
    if n == 0:
        return samples.new_zeros(num, samples.shape[1])
    if n >= num:
        idx = torch.randperm(n, device=samples.device)[:num]
    else:
        idx = torch.randint(0, n, (num,), device=samples.device)
    return samples[idx]


class VectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        stride: int = 1,
        apply_rotation_trick: bool = True,
        normalize_codes: bool = False,
        threshold_ema_dead_code: float = 0.0,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.stride = stride
        self.apply_rotation_trick = apply_rotation_trick
        self.normalize_codes = normalize_codes
        # SoundStream/EnCodec-style dead-code revival: EMA per-code usage, and
        # reset codes below the threshold to random encoder vectors. 0 disables.
        self.threshold_ema_dead_code = float(threshold_ema_dead_code)
        self.ema_decay = float(ema_decay)

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)
        # Non-persistent: not saved in checkpoints (pure training-time stat).
        # Initialise at the dead-code threshold so NO code is considered dead at
        # step 0 (avoids a startup storm that would overwrite even heavily-used
        # codes and diverge the encoder). Genuinely unused codes then decay
        # below threshold within a few steps and get revived; used codes
        # accumulate counts and stay alive.
        self.register_buffer(
            "cluster_size",
            torch.full((codebook_size,), float(threshold_ema_dead_code)),
            persistent=False,
        )

    @torch.no_grad()
    def _expire_dead_codes(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        """Update EMA usage from ``indices`` and reinit codes below threshold.

        ``z_e`` is the (un-normalised) projected encoder output in codebook
        space, shape (B, codebook_dim, T); ``indices`` is (B, T).
        """
        # Per-rank LOCAL usage only — deliberately no collectives here. A
        # collective (all_reduce / broadcast) inside forward requires every rank
        # to call it in lockstep; that breaks whenever rank 0 does main-process
        # -only work after validation (checkpoint save + render), leaving the
        # other ranks to block on a collective rank 0 never reaches -> NCCL
        # timeout. Local revival is an approximation (codebooks drift slightly
        # across ranks for just-revived/unused codes, which carry no output
        # signal and re-converge once selected via DDP-averaged gradients), but
        # it is deadlock-free and robust to rank desync.
        flat_idx = indices.reshape(-1)
        counts = torch.bincount(flat_idx, minlength=self.codebook_size).to(
            self.cluster_size.dtype
        )
        _ema_inplace(self.cluster_size, counts, self.ema_decay)

        dead = self.cluster_size < self.threshold_ema_dead_code
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        z_e_flat = rearrange(z_e.detach(), "b d t -> (b t) d")
        samples = _sample_vectors(z_e_flat, n_dead)
        # Renormalise revived samples to the typical (alive) codebook magnitude.
        # With normalize_codes=True the lookup is magnitude-invariant, so storing
        # raw encoder outputs lets revival inherit (and, via the commitment loss,
        # amplify) any encoder-magnitude drift — a runaway that explodes
        # commit/codebook loss. Matching the existing scale breaks that loop.
        alive = ~dead
        ref = self.codebook.weight.data[alive] if alive.any() else self.codebook.weight.data
        target_norm = ref.norm(dim=1).mean().clamp(min=1e-6)
        samples = F.normalize(samples, dim=1) * target_norm
        self.codebook.weight.data[dead] = samples.to(self.codebook.weight.dtype)
        # Generous grace (~hundreds of steps at decay 0.99) so a revived code is
        # trained by gradient and given the chance to be selected before it can
        # be culled again — prevents per-step churn.
        self.cluster_size[dead] = self.threshold_ema_dead_code * 10.0

    def forward(self, z):
        target_len = z.shape[-1]
        if self.stride > 1:
            z = F.avg_pool1d(z, self.stride, stride=self.stride)

        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)

        if self.training and self.threshold_ema_dead_code > 0:
            self._expire_dead_codes(z_e, indices)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        if self.apply_rotation_trick:
            z_e_rot = rearrange(z_e, "b d t -> b t d")
            z_q_rot = rearrange(z_q, "b d t -> b t d")

            with torch.no_grad():
                e_norm = F.normalize(z_e_rot.detach(), dim=-1)
                q_norm = F.normalize(z_q_rot.detach(), dim=-1)
                r = F.normalize(e_norm + q_norm, dim=-1)

                B, T, D = z_e_rot.shape
                I = torch.eye(D, device=z_e_rot.device).expand(B, T, D, D)
                rrt = torch.einsum("bti,btj->btij", r, r)
                qet = torch.einsum("bti,btj->btij", q_norm, e_norm)
                R = I - 2 * rrt + 2 * qet

                scaling = (
                    z_q_rot.norm(dim=-1) / z_e_rot.norm(dim=-1).clamp(min=1e-8)
                ).unsqueeze(-1)

            z_q = rearrange(
                scaling * torch.einsum("btij,btj->bti", R, z_e_rot), "b t d -> b d t"
            )
        else:
            z_q = z_e + (z_q - z_e).detach()

        z_q = self.out_proj(z_q)
        if self.stride > 1:
            z_q = F.interpolate(z_q, size=target_len, mode="linear")

        return z_q, indices, commitment_loss, codebook_loss

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight
        if self.normalize_codes:
            # ViT-VQGAN / DAC cosine-similarity NN: unit-norm both sides so
            # ‖u − v‖² = 2 − 2·u·v and argmin distance ⇔ argmax cos-sim. We
            # use the normalised vectors ONLY for index selection; z_q below
            # is then read from `self.codebook.weight` (un-normalised), and
            # the encoder output `z_e` upstream is also un-normalised, so the
            # MSE commit/code losses operate in the original magnitude space.
            encodings = F.normalize(encodings, dim=-1)
            codebook = F.normalize(codebook, dim=-1)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        vq_strides: List[int] = (1, 1, 1, 1),
        apply_rotation_trick: bool = True,
        quantizer_dropout: float = 0.0,
        normalize_codes: bool = False,
        dead_code_threshold: float = 0.0,
        dead_code_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.vq_strides = list(vq_strides[:n_codebooks])
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        # DAC-style quantizer dropout: fraction of each batch for which we use a
        # random subset of quantizers (1..N) instead of all N. Teaches the
        # decoder to reconstruct from variable bitrate so we can drop late
        # codebooks at inference. 0.0 = always use all N (no dropout). DAC paper
        # uses 0.5.
        self.quantizer_dropout = float(quantizer_dropout)

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(
                    input_dim, codebook_size, codebook_dim, self.vq_strides[i],
                    apply_rotation_trick=apply_rotation_trick,
                    normalize_codes=normalize_codes,
                    threshold_ema_dead_code=dead_code_threshold,
                    ema_decay=dead_code_ema_decay,
                )
                for i in range(self.n_codebooks)
            ]
        )

    def forward(self, z, n_quantizers: int = None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        codes = []

        B = z.shape[0]
        device = z.device

        if self.training and self.quantizer_dropout > 0.0:
            # Per-sample number of active quantizers. Default: N+1 (>= every i,
            # so all codebooks always pass the i < n_q[b] mask). Replace the
            # first n_dropout entries with a uniform-random count in [1, N].
            # Relies on the batch being shuffled (it is via DataLoader).
            n_q = torch.full((B,), self.n_codebooks + 1, device=device, dtype=torch.long)
            n_dropout = int(B * self.quantizer_dropout)
            if n_dropout > 0:
                n_q[:n_dropout] = torch.randint(
                    1, self.n_codebooks + 1, (n_dropout,), device=device
                )
            iter_n = self.n_codebooks
        else:
            # Eval / no-dropout path: honour the explicit n_quantizers arg if
            # given, else use all N codebooks for every sample.
            if n_quantizers is None:
                n_quantizers = self.n_codebooks
            n_quantizers = min(self.n_codebooks, n_quantizers)
            n_q = torch.full((B,), n_quantizers, device=device, dtype=torch.long)
            iter_n = n_quantizers

        for i in range(iter_n):
            z_q_i, indices_i, commitment_loss_i, codebook_loss_i = self.quantizers[i](residual)
            # mask[b]=True if sample b uses quantizer i (i.e., i < n_q[b]).
            mask = (i < n_q).to(z.dtype)
            z_q = z_q + z_q_i * mask[:, None, None]
            # Always update residual with the full quantized contribution so
            # deeper quantizers train on realistic residuals (matches DAC).
            residual = residual - z_q_i
            # Mask per-sample losses. .mean() over the full batch matches DAC's
            # convention; dropped samples contribute 0 to deep-quantizer losses.
            commitment_loss = commitment_loss + (commitment_loss_i * mask).mean()
            codebook_loss = codebook_loss + (codebook_loss_i * mask).mean()
            codes.append(indices_i)

        return z_q, codes, commitment_loss, codebook_loss

    def from_codes(self, codes: list[torch.Tensor], n_quantizers: int = None):
        """Reconstruct continuous representation from discrete codes."""
        z_q = 0.0
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        n_quantizers = min(n_quantizers, len(codes))
        target_len = codes[0].shape[-1] * self.quantizers[0].stride

        for i in range(n_quantizers):
            z_p_i = self.quantizers[i].decode_code(codes[i])
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            if self.quantizers[i].stride > 1:
                z_q_i = F.interpolate(z_q_i, size=target_len, mode="linear")
            z_q += z_q_i
        return z_q

    def get_cb_usage(self, codes):
        usage_stats = {}
        for i, code in enumerate(codes):
            unique_codes = torch.unique(code)
            utilization = len(unique_codes) / self.codebook_size
            usage_stats[f"codebook_{i}"] = {
                "unique_codes": len(unique_codes),
                "utilization": utilization,
                "total_codes": self.codebook_size,
            }
        return usage_stats
