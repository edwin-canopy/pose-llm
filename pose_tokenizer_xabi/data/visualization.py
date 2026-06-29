from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

from .kinematic import (
    NUM_JOINTS,
    offsets_to_positions,
    FACE_CONNECTIONS,
    BODY_CONNECTIONS,
    LEFT_HAND_CONNECTIONS,
    RIGHT_HAND_CONNECTIONS,
    SKELETON_COLORS,
)

_CONF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "conf", ["#ff0000", "#ffaa00", "#00ff00"]
)


def _to_numpy(t):
    return t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def render_pose_frame(
    sample: dict[str, torch.Tensor | np.ndarray],
    frame_idx: int = 0,
    resolution: tuple[int, int] = (368, 240),
    point_size: int = 20,
    show_confidence: bool = True,
    confidence_threshold: float = 0.0,
) -> Figure:
    """Render a single frame from a dataset sample.

    Accepts the flattened kinematic-chain format (keypoints (T,100) +
    global_state (T,3)) and reconstructs absolute positions for rendering.

    Args:
        sample: dict as returned by PoseDataset.__getitem__.
        frame_idx: which frame to render.
        resolution: (height, width) in pixels for the output figure.
        point_size: scatter point size.
        show_confidence: colour joints by chain-propagated confidence.
        confidence_threshold: hide joints below this chain confidence.
    """
    kp_flat = _to_numpy(sample["keypoints"])          # (T, 100)
    global_state = _to_numpy(sample["global_state"])   # (T, 3)

    # Unflatten and de-normalise
    kc_offsets = kp_flat.reshape(-1, NUM_JOINTS, 2)    # (T, 50, 2)
    shoulder_width = global_state[:, 2:3, np.newaxis]  # (T, 1, 1)
    kc_offsets = kc_offsets * shoulder_width

    root = global_state[:, :2]  # (T, 2)
    positions = offsets_to_positions(root, kc_offsets)  # (T, 50, 2)
    kpts = positions[frame_idx]  # (50, 2)

    # Confidence for colouring / thresholding
    conf = None
    if show_confidence or confidence_threshold > 0:
        if "confidence" in sample:
            raw = _to_numpy(sample["confidence"])
            conf = raw[frame_idx]

    visible = np.ones(NUM_JOINTS, dtype=bool)
    if conf is not None and confidence_threshold > 0:
        visible = conf >= confidence_threshold

    h, w = resolution
    dpi = 50
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    y = kpts[:, 0] * h
    x = kpts[:, 1] * w

    def _draw(connections, fallback_color):
        for i, j in connections:
            if not (visible[i] and visible[j]):
                continue
            if conf is not None and show_confidence:
                color = _CONF_CMAP(min(conf[i], conf[j]))
            else:
                color = fallback_color
            ax.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=1.5)

        indices = sorted(
            {i for pair in connections for i in pair} & set(np.where(visible)[0])
        )
        if not indices:
            return
        if conf is not None and show_confidence:
            colors = [_CONF_CMAP(conf[i]) for i in indices]
            ax.scatter(x[indices], y[indices], c=colors, s=point_size, zorder=5)
        else:
            ax.scatter(
                x[indices], y[indices], c=fallback_color, s=point_size, zorder=5
            )

    _draw(FACE_CONNECTIONS, SKELETON_COLORS["face"])
    _draw(BODY_CONNECTIONS, SKELETON_COLORS["body"])
    _draw(LEFT_HAND_CONNECTIONS, SKELETON_COLORS["left_hand"])
    _draw(RIGHT_HAND_CONNECTIONS, SKELETON_COLORS["right_hand"])

    if conf is not None and show_confidence:
        sm = plt.cm.ScalarMappable(cmap=_CONF_CMAP, norm=plt.Normalize(0, 1))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=6, colors="white")
        cbar.set_label("chain confidence", fontsize=6, color="white")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    return fig


def render_test_comparison(
    original: dict,
    reconstructed_kp: np.ndarray | torch.Tensor,
    frame_idx: int = 0,
    title: str = "",
) -> Figure:
    """Side-by-side original vs reconstructed pose for a single sample."""
    recon_kp = _to_numpy(reconstructed_kp)
    global_state = _to_numpy(original["global_state"])

    # The model's strided convolutions can shorten the temporal axis;
    # truncate all signals to the shorter length so shapes match.
    t = min(recon_kp.shape[0], global_state.shape[0])
    recon = {
        "keypoints": recon_kp[:t],
        "global_state": global_state[:t],
    }
    orig = {
        "keypoints": _to_numpy(original["keypoints"])[:t],
        "global_state": global_state[:t],
        "confidence": _to_numpy(original["confidence"])[:t],
    }

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8, 6))
    fig.patch.set_facecolor("black")

    for src, ax, label in [(orig, ax_l, "Original"), (recon, ax_r, "Reconstructed")]:
        sub = render_pose_frame(src, frame_idx=frame_idx, show_confidence=False)
        sub.canvas.draw()
        buf = np.asarray(sub.canvas.buffer_rgba())
        ax.imshow(buf)
        ax.set_title(label, color="white", fontsize=10)
        ax.axis("off")
        plt.close(sub)

    if title:
        fig.suptitle(title, color="white", fontsize=12)
    fig.tight_layout()
    return fig
