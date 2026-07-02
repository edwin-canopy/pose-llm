"""TEMP — delete after use.

Reads the first 4 rows of the same eval dataset used by
audio_pose_interleaved_inference.py, runs each row's GROUND-TRUTH audio_tokens
through kyutai/mimi, and writes inference_outputs/eval_audio_<i>.mp3.
Lets us A/B the model-generated audio_<i>.mp3 against the ground-truth audio.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from datasets import load_from_disk
from transformers import MimiModel

from paths import MERGED_DATASET_DIR


DATASET_DIR = f"{MERGED_DATASET_DIR}/hf_pose_dataset_filtered"
INFERENCE_OUTPUTS_DIR = "/home/edwin/pose-llm/inference_outputs"

N_SAMPLES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIMI_MODEL_ID = "kyutai/mimi"
SAMPLE_RATE = 24000
MP3_BITRATE = "128k"


mimi = MimiModel.from_pretrained(MIMI_MODEL_ID).to(DEVICE).eval()
print(f"loaded MimiModel from {MIMI_MODEL_ID} on {DEVICE}")


def _write_mp3(samples: np.ndarray, out_path: str) -> None:
    pcm = samples.astype(np.float32, copy=False).clip(-1.0, 1.0).tobytes()
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE,
        out_path,
    ]
    subprocess.run(cmd, input=pcm, check=True, capture_output=True)


raw = load_from_disk(DATASET_DIR)
eval_dataset = raw.select(range(N_SAMPLES))
print(f"loaded {len(eval_dataset)} eval rows from {DATASET_DIR}")

os.makedirs(INFERENCE_OUTPUTS_DIR, exist_ok=True)

for i in range(N_SAMPLES):
    sample = eval_dataset[i]
    audio_codes = torch.tensor(sample["audio_tokens"], dtype=torch.long, device=DEVICE)  # (K, T)
    audio_codes = audio_codes.unsqueeze(0)                                                # (1, K, T)

    with torch.inference_mode():
        out = mimi.decode(audio_codes)
    waveform = out.audio_values if hasattr(out, "audio_values") else out[0]
    samples = waveform[0, 0].float().cpu().numpy()

    mp3_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"eval_audio_{i}.mp3")
    _write_mp3(samples, mp3_path)
    print(
        f"  sample {i}: codes={tuple(audio_codes.shape[1:])} "
        f"samples={samples.shape[0]} ({samples.shape[0] / SAMPLE_RATE:.2f}s) -> {mp3_path}"
    )
