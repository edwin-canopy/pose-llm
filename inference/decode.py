"""Decode generated pose code tensors back to keypoint features.

Reads every `pose_*.pt` file produced by audio_pose_interleaved_inference.py
(shape (POSE_DEPTH, T), raw codebook indices), runs them through the trained
PoseTokenizer, and writes one `keypoints_<i>.pt` per input next to it.

Output tensor shape: (T, F) where F = config.input_features (e.g. 110 for
55 joints x 2). These are shoulder-width-normalised kinematic offsets;
convert to (T, 55, 2) via pose_tokenizer.data.rendering.kp_flat_to_positions
if you also have a global_state.
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from pose_tokenizer import PoseTokenizer


INFERENCE_OUTPUTS_DIR = "/home/edwin/pose-llm/inference_outputs"
TOKENIZER_PATH = "/path/to/pose_tokenizer_checkpoint"  # local dir or HF repo id

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = PoseTokenizer.from_pretrained(TOKENIZER_PATH, device=DEVICE)
print(f"loaded PoseTokenizer from {TOKENIZER_PATH} on {DEVICE}")

pose_files = sorted(
    glob.glob(os.path.join(INFERENCE_OUTPUTS_DIR, "pose_*.pt")),
    key=lambda p: int(re.search(r"pose_(\d+)\.pt$", p).group(1)),
)
if not pose_files:
    raise FileNotFoundError(f"no pose_*.pt files in {INFERENCE_OUTPUTS_DIR}")

print(f"found {len(pose_files)} pose tensors to decode")

for path in pose_files:
    idx = int(re.search(r"pose_(\d+)\.pt$", path).group(1))
    pose_codes = torch.load(path, map_location=DEVICE)  # (POSE_DEPTH, T)
    codes = list(pose_codes.long().unbind(0))           # list[(T,)] per codebook
    keypoints = tokenizer.decode(codes)                 # (T, F)

    out_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"keypoints_{idx}.pt")
    torch.save(keypoints.cpu(), out_path)
    print(f"  {path} -> {out_path}  shape={tuple(keypoints.shape)}")
