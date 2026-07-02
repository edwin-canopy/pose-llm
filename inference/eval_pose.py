"""TEMP — delete after use.

Reads the first N rows of the same eval dataset used by
audio_pose_interleaved_inference.py, runs each row's GROUND-TRUTH pose_tokens
through the PoseTokenizer, and writes inference_outputs/eval_pose_<i>.gif.
Lets us eyeball how the ground-truth tokens look when decoded — a baseline to
compare model-generated gifs against. All pose rendering delegates to
decode_pose so there's exactly one rendering implementation.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import yaml
from datasets import load_from_disk

from paths import MERGED_DATASET_DIR, POSE_NPZ_DIR, XABI_TOKENIZER_PATH
from pose_tokenizer_xabi import PoseTokenizer
from pose_tokenizer_xabi.data.kinematic import SELECTED_INDICES
from decode_pose import decoded_features_to_positions, render_pose_gif


DATASET_DIR = f"{MERGED_DATASET_DIR}/hf_pose_dataset_filtered"
INFERENCE_OUTPUTS_DIR = "/home/edwin/pose-llm/inference_outputs"
TOKENIZER_PATH = XABI_TOKENIZER_PATH

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
CONFIG = yaml.safe_load(open(CONFIG_PATH))
POSE_DEPTH = CONFIG["pose_depth_model"]["residual_depth"]

TOKENIZER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tokenizer_config.yaml")
TOKENIZER_CFG = yaml.safe_load(open(TOKENIZER_CONFIG_PATH))
GIF_FPS = TOKENIZER_CFG["gif"]["fps"]

N_SAMPLES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = PoseTokenizer.from_pretrained(TOKENIZER_PATH, device=DEVICE)
tokenizer.model.set_active_codebooks(POSE_DEPTH)
print(f"loaded PoseTokenizer from {TOKENIZER_PATH} on {DEVICE} "
      f"(active_codebooks={POSE_DEPTH})")

raw = load_from_disk(DATASET_DIR)
eval_dataset = raw.select(range(N_SAMPLES))
print(f"loaded {len(eval_dataset)} eval rows from {DATASET_DIR}")

os.makedirs(INFERENCE_OUTPUTS_DIR, exist_ok=True)

for i in range(N_SAMPLES):
    sample = eval_dataset[i]
    pose_flat = torch.tensor(sample["pose_tokens"], dtype=torch.long)
    pose_codes = pose_flat.view(-1, POSE_DEPTH).T.contiguous().to(DEVICE)  # (POSE_DEPTH, T)

    codes = list(pose_codes.long().unbind(0))
    recon = tokenizer.decode(codes)                              # (T, 110)
    positions = decoded_features_to_positions(recon)             # (T, 55, 2)

    gif_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"eval_pose_{i}.gif")
    render_pose_gif(positions, gif_path, fps=GIF_FPS)
    print(
        f"  sample {i}: pose_codes={tuple(pose_codes.shape)} "
        f"positions={positions.shape} -> {gif_path}"
    )

    clip_key = re.sub(r"_chunk_\d+$", "", sample["row_id"])
    npz_path = os.path.join(POSE_NPZ_DIR, f"{clip_key}_poses.npz")
    raw_poses = np.load(npz_path)["poses"]                 # (T, 133, 3) as (x, y, conf)
    raw_xy = raw_poses[:, SELECTED_INDICES, :2]            # (T, 55, 2) in (x, y)
    # Rotate 90° clockwise in image space: (x, y) -> (-y, x); render expects (y, x).
    raw_yx = np.stack([raw_xy[..., 0], -raw_xy[..., 1]], axis=-1).copy()

    original_gif_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"eval_pose_{i}_original.gif")
    render_pose_gif(raw_yx, original_gif_path, fps=GIF_FPS)
    print(f"    original: raw_poses={raw_poses.shape} -> {original_gif_path}")
