"""TEMP — delete after use.

Reads the first 3 rows of the same eval dataset used by
audio_pose_interleaved_inference.py, runs each row's GROUND-TRUTH pose_tokens
through the same PoseTokenizer used in decode.py, and writes
inference_outputs/eval_pose_<i>.gif. Lets us eyeball how the ground-truth
tokens look when decoded — a baseline to compare model-generated gifs against.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from PIL import Image, ImageDraw

from pose_tokenizer_james import PoseTokenizer


DATASET_DIR = "/mnt/somfs/pose_cond/merged_pose_audio_dataset/hf_pose_dataset_filtered"
INFERENCE_OUTPUTS_DIR = "/home/edwin/pose-llm/inference_outputs"
TOKENIZER_PATH = "/mnt/somfs/james-checkpoints/pose-tokenizer-james/cb_size_2048/base-causal-2x-wide-16cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048/step_200000"

CONFIG = yaml.safe_load(open("config.yaml"))
POSE_DEPTH = CONFIG["pose_depth_model"]["residual_depth"]

N_SAMPLES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GIF_FPS = 2
GIF_CANVAS = 512
GIF_DOT_RADIUS = 3
GIF_PADDING = 16


tokenizer = PoseTokenizer.from_pretrained(TOKENIZER_PATH, device=DEVICE)
print(f"loaded PoseTokenizer from {TOKENIZER_PATH} on {DEVICE}")

NUM_JOINTS = tokenizer.config.num_keypoints
JOINT_DIM = tokenizer.config.input_features // NUM_JOINTS


def _render_gif(keypoints: np.ndarray, out_path: str) -> None:
    per_joint = keypoints.reshape(keypoints.shape[0], NUM_JOINTS, JOINT_DIM)
    pts = per_joint[..., :2][..., ::-1]  # (y,x) -> (x,y)
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


raw = load_from_disk(DATASET_DIR)
eval_dataset = raw.select(range(N_SAMPLES))
print(f"loaded {len(eval_dataset)} eval rows from {DATASET_DIR}")

os.makedirs(INFERENCE_OUTPUTS_DIR, exist_ok=True)

for i in range(N_SAMPLES):
    sample = eval_dataset[i]
    pose_flat = torch.tensor(sample["pose_tokens"], dtype=torch.long)
    pose_codes = pose_flat.view(-1, POSE_DEPTH).T.contiguous().to(DEVICE)  # (POSE_DEPTH, T)

    codes = list(pose_codes.long().unbind(0))
    keypoints = tokenizer.decode(codes)  # (T, F)

    gif_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"eval_pose_{i}.gif")
    _render_gif(keypoints.float().cpu().numpy(), gif_path)
    print(
        f"  sample {i}: pose_codes={tuple(pose_codes.shape)} "
        f"keypoints={tuple(keypoints.shape)} -> {gif_path}"
    )
