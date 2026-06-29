"""Decode generated pose code tensors back to keypoint features.

Reads every `pose_*.pt` file produced by audio_pose_interleaved_inference.py
(shape (POSE_DEPTH, T), raw codebook indices), runs them through the trained
PoseTokenizer specified in inference/tokenizer_config.yaml, and writes one
`keypoints_<i>.pt` plus a sanity-check `pose_<i>.gif` per input.

Output tensor shape: (T, F) where F = config.input_features (e.g. 110 for
55 joints x 2, or 165 for 55 joints x 3 with confidence). These are
shoulder-width-normalised kinematic offsets; convert to (T, 55, 2) via
<pkg>.data.rendering.kp_flat_to_positions if you also have a global_state.
"""

import glob
import importlib
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tokenizer_config.yaml")
CFG = yaml.safe_load(open(CONFIG_PATH))

PACKAGE = CFG["package"]
if PACKAGE not in ("james", "xabi"):
    raise ValueError(f"package must be 'james' or 'xabi', got {PACKAGE!r}")
WEIGHTS_PATH = CFG[f"{PACKAGE}_path"]
N_CODEBOOKS = CFG.get("n_codebooks")
INFERENCE_OUTPUTS_DIR = CFG["inference_outputs_dir"]

_device_cfg = CFG.get("device", "auto")
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if _device_cfg == "auto" else _device_cfg

GIF_FPS = CFG["gif"]["fps"]
GIF_CANVAS = CFG["gif"]["canvas"]
GIF_DOT_RADIUS = CFG["gif"]["dot_radius"]
GIF_PADDING = CFG["gif"]["padding"]


pose_tokenizer_pkg = importlib.import_module(f"pose_tokenizer_{PACKAGE}")
PoseTokenizer = pose_tokenizer_pkg.PoseTokenizer

tokenizer = PoseTokenizer.from_pretrained(WEIGHTS_PATH, device=DEVICE)
print(f"loaded PoseTokenizer ({PACKAGE}) from {WEIGHTS_PATH} on {DEVICE}")
print(f"  checkpoint n_codebooks={tokenizer.config.n_codebooks}, "
      f"codebook_size={tokenizer.config.codebook_size}, "
      f"input_features={tokenizer.config.input_features}")

if N_CODEBOOKS is not None:
    tokenizer.model.set_active_codebooks(N_CODEBOOKS)
    print(f"  active n_codebooks overridden to {N_CODEBOOKS}")

pose_files = sorted(
    glob.glob(os.path.join(INFERENCE_OUTPUTS_DIR, "pose_*.pt")),
    key=lambda p: int(re.search(r"pose_(\d+)\.pt$", p).group(1)),
)
if not pose_files:
    raise FileNotFoundError(f"no pose_*.pt files in {INFERENCE_OUTPUTS_DIR}")

print(f"found {len(pose_files)} pose tensors to decode")

NUM_JOINTS = tokenizer.config.num_keypoints
JOINT_DIM = tokenizer.config.input_features // NUM_JOINTS


def _render_gif(keypoints: np.ndarray, out_path: str) -> None:
    """keypoints: (T, F=NUM_JOINTS*JOINT_DIM) per-joint [y, x, (conf)]; plot xy
    offsets as black dots on a white canvas (no kinematic chain — these are
    parent-relative offsets, not absolute positions)."""
    per_joint = keypoints.reshape(keypoints.shape[0], NUM_JOINTS, JOINT_DIM)
    # Layout is (y, x, [conf]); swap to (x, y) for image-space drawing.
    pts = per_joint[..., :2][..., ::-1]
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
        for x, y in frame_pts:
            px = (x - x_min) * scale + x_off
            py = (y - y_min) * scale + y_off
            draw.ellipse(
                (px - GIF_DOT_RADIUS, py - GIF_DOT_RADIUS,
                 px + GIF_DOT_RADIUS, py + GIF_DOT_RADIUS),
                fill="black",
            )
        frames.append(img)

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / GIF_FPS),
        loop=0,
    )


EXPECTED_N_CODEBOOKS = (
    N_CODEBOOKS if N_CODEBOOKS is not None else tokenizer.config.n_codebooks
)


for path in pose_files:
    idx = int(re.search(r"pose_(\d+)\.pt$", path).group(1))
    pose_codes = torch.load(path, map_location=DEVICE)  # (POSE_DEPTH, T)
    assert pose_codes.shape[0] == EXPECTED_N_CODEBOOKS, (
        f"{path} has {pose_codes.shape[0]} codebooks but config selects "
        f"{EXPECTED_N_CODEBOOKS} (n_codebooks={N_CODEBOOKS!r}, "
        f"checkpoint n_codebooks={tokenizer.config.n_codebooks})"
    )
    codes = list(pose_codes.long().unbind(0))           # list[(T,)] per codebook
    keypoints = tokenizer.decode(codes)                 # (T, F)

    out_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"keypoints_{idx}.pt")
    torch.save(keypoints.cpu(), out_path)

    gif_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"pose_{idx}.gif")
    _render_gif(keypoints.float().cpu().numpy(), gif_path)
    print(f"  {path} -> {out_path}  shape={tuple(keypoints.shape)}  gif={gif_path}")
