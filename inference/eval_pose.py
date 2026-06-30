"""TEMP — delete after use.

Reads the first N rows of the same eval dataset used by
audio_pose_interleaved_inference.py, runs each row's GROUND-TRUTH pose_tokens
through the PoseTokenizer, and writes inference_outputs/eval_pose_<i>.gif.
Lets us eyeball how the ground-truth tokens look when decoded — a baseline to
compare model-generated gifs against. All pose rendering delegates to
decode_pose so there's exactly one rendering implementation.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from datasets import load_from_disk

from pose_tokenizer_james import PoseTokenizer
from decode_pose import decoded_features_to_positions, render_pose_gif


DATASET_DIR = "/mnt/somfs/pose_cond/merged_pose_audio_dataset/hf_pose_dataset_filtered"
INFERENCE_OUTPUTS_DIR = "/home/edwin/pose-llm/inference_outputs"
TOKENIZER_PATH = "/mnt/somfs/james-checkpoints/pose-tokenizer-james/cb_size_2048/base-causal-2x-wide-16cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048/step_200000"

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
CONFIG = yaml.safe_load(open(CONFIG_PATH))
POSE_DEPTH = CONFIG["pose_depth_model"]["residual_depth"]

N_SAMPLES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = PoseTokenizer.from_pretrained(TOKENIZER_PATH, device=DEVICE)
print(f"loaded PoseTokenizer from {TOKENIZER_PATH} on {DEVICE}")

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
    render_pose_gif(positions, gif_path)
    print(
        f"  sample {i}: pose_codes={tuple(pose_codes.shape)} "
        f"positions={positions.shape} -> {gif_path}"
    )
