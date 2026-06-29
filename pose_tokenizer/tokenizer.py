from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

from .config import PoseTokenizerConfig
from .model import PoseTokenizerModel

_NORM_STATS_FILE = "norm_stats.npz"


class PoseTokenizer:
    """High-level tokenizer interface that wraps the underlying model.

    Usage:
        tokenizer = PoseTokenizer.from_pretrained("your-hf-repo")
        tokens = tokenizer.encode(keypoints)
        keypoints_hat = tokenizer.decode(tokens)
    """

    def __init__(
        self,
        model: PoseTokenizerModel,
        device: str | torch.device = "cpu",
        norm_center: np.ndarray | None = None,
        norm_scale: np.ndarray | None = None,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.config = model.config
        self._norm_center = (
            torch.from_numpy(norm_center).float().to(device)
            if norm_center is not None else None
        )
        self._norm_scale = (
            torch.from_numpy(norm_scale).float().to(device)
            if norm_scale is not None else None
        )

    @property
    def has_norm_stats(self) -> bool:
        return self._norm_center is not None and self._norm_scale is not None

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> "PoseTokenizer":
        model = PoseTokenizerModel.from_pretrained(str(path_or_repo), **kwargs)
        norm_center, norm_scale = None, None
        ns_path = Path(path_or_repo) / _NORM_STATS_FILE
        if ns_path.is_file():
            st = np.load(ns_path)
            norm_center = st["center"].astype(np.float32)
            norm_scale = st["scale"].astype(np.float32)
        return cls(model=model, device=device,
                   norm_center=norm_center, norm_scale=norm_scale)

    def save_pretrained(self, path: str | Path) -> None:
        self.model.save_pretrained(str(path))
        if self.has_norm_stats:
            np.savez(
                Path(path) / _NORM_STATS_FILE,
                center=self._norm_center.cpu().numpy(),
                scale=self._norm_scale.cpu().numpy(),
            )

    def push_to_hub(self, repo_id: str, **kwargs) -> None:
        self.model.push_to_hub(repo_id, **kwargs)

    def _standardise(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_norm_stats:
            return (x - self._norm_center) / self._norm_scale
        return x

    def _destandardise(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_norm_stats:
            return x * self._norm_scale + self._norm_center
        return x

    @torch.no_grad()
    def encode(self, keypoints: torch.Tensor) -> list[torch.Tensor]:
        """Keypoints -> discrete token codes (one tensor per codebook).

        If norm stats are loaded, the input is standardised before encoding.
        Pass raw (shoulder-width-normalised) keypoints -- standardisation is
        handled internally.

        Args:
            keypoints: (B, T, F) or (T, F) raw keypoint features.

        Returns:
            List of code tensors, one per codebook.
        """
        was_unbatched = keypoints.ndim == 2
        if was_unbatched:
            keypoints = keypoints.unsqueeze(0)

        keypoints = keypoints.to(self.device)
        expected = self.config.input_features
        got = keypoints.shape[-1]
        if got != expected:
            raise ValueError(
                f"Input has {got} features but this model expects {expected}. "
                f"This typically means the data was prepared with a different "
                f"joint layout (e.g. 55 joints / 110 features vs 50 joints / 100 features)."
            )
        keypoints = self._standardise(keypoints)
        codes = self.model.tokenize(keypoints)

        if was_unbatched:
            codes = [c.squeeze(0) for c in codes]
        return codes

    @torch.no_grad()
    def decode(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """Discrete token codes -> reconstructed keypoints.

        If norm stats are loaded, the output is de-standardised back to
        shoulder-width-normalised offsets.

        Args:
            codes: List of code tensors from encode().

        Returns:
            Reconstructed keypoint features (shoulder-width-normalised).
        """
        was_unbatched = codes[0].ndim == 1
        if was_unbatched:
            codes = [c.unsqueeze(0) for c in codes]

        codes = [c.to(self.device) for c in codes]
        keypoints = self.model.detokenize(codes)
        keypoints = self._destandardise(keypoints)

        if was_unbatched:
            keypoints = keypoints.squeeze(0)
        return keypoints
