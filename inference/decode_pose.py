"""Decode generated pose code tensors back to absolute xy joint positions.

Reads every `pose_*.pt` file produced by audio_pose_interleaved_inference.py
(shape (POSE_DEPTH, T), raw codebook indices), runs them through the trained
xabi PoseTokenizer, unfolds the resulting parent-relative offsets into joint
positions (with root pinned to the origin and shoulder width = 1, since the
generated tokens do not carry global_state), and writes one `keypoints_<i>.pt`
of shape (T, 55, 2) in (y, x) plus a sanity-check `pose_<i>.gif` per input.

Other inference scripts (eval_pose.py, comparisons.py) import the rendering
helpers from this module so all pose rendering goes through the same code path.
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from pose_tokenizer_xabi import PoseTokenizer
from pose_tokenizer_xabi.data.kinematic import (
    NUM_JOINTS,
    offsets_to_positions,
    FACE_CONNECTIONS,
    BODY_CONNECTIONS,
    LEFT_HAND_CONNECTIONS,
    RIGHT_HAND_CONNECTIONS,
)


_CONNECTIONS = (
    FACE_CONNECTIONS
    + BODY_CONNECTIONS
    + LEFT_HAND_CONNECTIONS
    + RIGHT_HAND_CONNECTIONS
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tokenizer_config.yaml")
CFG = yaml.safe_load(open(CONFIG_PATH))

GIF_FPS = CFG["gif"]["fps"]
GIF_CANVAS = CFG["gif"]["canvas"]
GIF_DOT_RADIUS = CFG["gif"]["dot_radius"]
GIF_PADDING = CFG["gif"]["padding"]


def decoded_features_to_positions(recon_features) -> np.ndarray:
    """xabi-decoded (T, F) shoulder-width offsets -> (T, 55, 2) absolute joint
    positions in (y, x), with root pinned to the origin and shoulder width = 1
    (no global_state applied). F may be 55*2 = 110 (y, x only) or 55*3 = 165
    (y, x, confidence); the confidence channel is dropped if present. Accepts
    torch tensor or numpy array."""
    if isinstance(recon_features, torch.Tensor):
        recon_features = recon_features.float().cpu().numpy()
    feat_dim = recon_features.shape[-1]
    assert feat_dim % NUM_JOINTS == 0, (
        f"feature dim {feat_dim} not divisible by NUM_JOINTS={NUM_JOINTS}"
    )
    joint_dim = feat_dim // NUM_JOINTS
    assert joint_dim in (2, 3), (
        f"unexpected joint dim {joint_dim}; expected 2 (y, x) or 3 (y, x, conf)"
    )
    offsets = recon_features.reshape(-1, NUM_JOINTS, joint_dim)[..., :2]
    root = np.zeros((offsets.shape[0], 2), dtype=offsets.dtype)
    return offsets_to_positions(root, offsets)


def render_pose_frames(positions: np.ndarray) -> list[Image.Image]:
    """(T, 55, 2) positions in (y, x) -> list of PIL skeleton frames (lines +
    dots) on a white canvas. Min/max-normalised across the whole clip so the
    figure fits the canvas."""
    pts = positions[..., ::-1]  # (y, x) -> (x, y) for image-space drawing
    x_min, y_min = pts[..., 0].min(), pts[..., 1].min()
    x_max, y_max = pts[..., 0].max(), pts[..., 1].max()
    span = max(x_max - x_min, y_max - y_min, 1e-6)
    scale = (GIF_CANVAS - 2 * GIF_PADDING) / span
    x_off = GIF_PADDING + ((GIF_CANVAS - 2 * GIF_PADDING) - (x_max - x_min) * scale) / 2
    y_off = GIF_PADDING + ((GIF_CANVAS - 2 * GIF_PADDING) - (y_max - y_min) * scale) / 2

    frames = []
    for frame_pts in pts:
        img = Image.new("RGB", (GIF_CANVAS, GIF_CANVAS), "white")
        draw = ImageDraw.Draw(img)
        px = (frame_pts[:, 0] - x_min) * scale + x_off
        py = (frame_pts[:, 1] - y_min) * scale + y_off
        for i, j in _CONNECTIONS:
            draw.line((px[i], py[i], px[j], py[j]), fill="black", width=1)
        for k in range(NUM_JOINTS):
            draw.ellipse(
                (px[k] - GIF_DOT_RADIUS, py[k] - GIF_DOT_RADIUS,
                 px[k] + GIF_DOT_RADIUS, py[k] + GIF_DOT_RADIUS),
                fill="black",
            )
        frames.append(img)
    return frames


def render_pose_gif(positions: np.ndarray, out_path: str) -> None:
    """Render skeleton frames from (T, 55, 2) positions and save as a GIF."""
    frames = render_pose_frames(positions)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / GIF_FPS),
        loop=0,
    )


def _main() -> None:
    weights_path = CFG["xabi_path"]
    n_codebooks_override = CFG.get("n_codebooks")
    inference_outputs_dir = CFG["inference_outputs_dir"]

    _device_cfg = CFG.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if _device_cfg == "auto" else _device_cfg

    tokenizer = PoseTokenizer.from_pretrained(weights_path, device=device)
    print(f"loaded PoseTokenizer (xabi) from {weights_path} on {device}")
    print(f"  checkpoint n_codebooks={tokenizer.config.n_codebooks}, "
          f"codebook_size={tokenizer.config.codebook_size}, "
          f"input_features={tokenizer.config.input_features}")

    assert tokenizer.config.input_features in (NUM_JOINTS * 2, NUM_JOINTS * 3), (
        f"expected (T, {NUM_JOINTS * 2}) (y, x) or (T, {NUM_JOINTS * 3}) "
        f"(y, x, conf) features; got input_features={tokenizer.config.input_features}"
    )

    if n_codebooks_override is not None:
        tokenizer.model.set_active_codebooks(n_codebooks_override)
        print(f"  active n_codebooks overridden to {n_codebooks_override}")

    pose_files = sorted(
        glob.glob(os.path.join(inference_outputs_dir, "pose_*.pt")),
        key=lambda p: int(re.search(r"pose_(\d+)\.pt$", p).group(1)),
    )
    if not pose_files:
        raise FileNotFoundError(f"no pose_*.pt files in {inference_outputs_dir}")

    print(f"found {len(pose_files)} pose tensors to decode")

    expected_n_codebooks = (
        n_codebooks_override if n_codebooks_override is not None
        else tokenizer.config.n_codebooks
    )

    for path in pose_files:
        idx = int(re.search(r"pose_(\d+)\.pt$", path).group(1))
        pose_codes = torch.load(path, map_location=device)  # (POSE_DEPTH, T)
        assert pose_codes.shape[0] == expected_n_codebooks, (
            f"{path} has {pose_codes.shape[0]} codebooks but config selects "
            f"{expected_n_codebooks} (n_codebooks={n_codebooks_override!r}, "
            f"checkpoint n_codebooks={tokenizer.config.n_codebooks})"
        )
        codes = list(pose_codes.long().unbind(0))           # list[(T,)] per codebook
        recon_offsets = tokenizer.decode(codes)             # (T, 110)
        positions = decoded_features_to_positions(recon_offsets)  # (T, 55, 2)

        out_path = os.path.join(inference_outputs_dir, f"keypoints_{idx}.pt")
        torch.save(torch.from_numpy(positions), out_path)

        gif_path = os.path.join(inference_outputs_dir, f"pose_{idx}.gif")
        render_pose_gif(positions, gif_path)
        print(f"  {path} -> {out_path}  shape={positions.shape}  gif={gif_path}")


if __name__ == "__main__":
    _main()
