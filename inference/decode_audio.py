"""Decode generated Mimi audio code tensors back to mp3.

Reads every `audio_*.pt` file produced by audio_pose_interleaved_inference.py
(shape (NUM_AUDIO_CODEBOOKS, T), raw codebook indices), runs them through the
kyutai/mimi neural codec, and writes `audio_<i>.mp3` per input next to it.

Mimi contract (from MimiModel.decode):
  audio_codes: (batch, num_quantizers, codes_length) long tensor, raw indices in [0, codebook_size).
  output:      (batch, 1, num_audio_samples) float, 24 kHz mono, num_audio_samples ≈ codes_length * 1920.

Mimi has 32 total quantizers (1 semantic + 31 acoustic, codebook_size=2048).
Our LM was trained on the first 8 (NUM_AUDIO_CODEBOOKS in preprocess_data.py),
which we pass through as-is — RVQ prefix decoding works at any depth ≤ 32.
"""

import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import yaml
from transformers import MimiModel


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tokenizer_config.yaml")
CFG = yaml.safe_load(open(CONFIG_PATH))

INFERENCE_OUTPUTS_DIR = CFG["inference_outputs_dir"]

_device_cfg = CFG.get("device", "auto")
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if _device_cfg == "auto" else _device_cfg

MIMI_MODEL_ID = "kyutai/mimi"
SAMPLE_RATE = 24000
MP3_BITRATE = "128k"


mimi = MimiModel.from_pretrained(MIMI_MODEL_ID).to(DEVICE).eval()
print(f"loaded MimiModel from {MIMI_MODEL_ID} on {DEVICE} "
      f"(num_quantizers={mimi.config.num_quantizers}, codebook_size={mimi.config.codebook_size}, "
      f"sample_rate={mimi.config.sampling_rate})")

audio_files = sorted(
    glob.glob(os.path.join(INFERENCE_OUTPUTS_DIR, "audio_*.pt")),
    key=lambda p: int(re.search(r"audio_(\d+)\.pt$", p).group(1)),
)
if not audio_files:
    raise FileNotFoundError(f"no audio_*.pt files in {INFERENCE_OUTPUTS_DIR}")
print(f"found {len(audio_files)} audio tensors to decode")


def _write_mp3(samples: np.ndarray, out_path: str) -> None:
    """Pipe float32 PCM mono into ffmpeg → libmp3lame mp3."""
    pcm = samples.astype(np.float32, copy=False).clip(-1.0, 1.0).tobytes()
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE,
        out_path,
    ]
    proc = subprocess.run(cmd, input=pcm, check=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))


for path in audio_files:
    idx = int(re.search(r"audio_(\d+)\.pt$", path).group(1))
    audio_codes = torch.load(path, map_location=DEVICE).long()  # (K, T)
    audio_codes = audio_codes.unsqueeze(0)                       # (1, K, T)

    with torch.inference_mode():
        out = mimi.decode(audio_codes)
    waveform = out.audio_values if hasattr(out, "audio_values") else out[0]
    samples = waveform[0, 0].float().cpu().numpy()               # (T_audio,)

    mp3_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"audio_{idx}.mp3")
    _write_mp3(samples, mp3_path)
    print(
        f"  {path} -> {mp3_path}  codes={tuple(audio_codes.shape[1:])} "
        f"samples={samples.shape[0]} ({samples.shape[0] / SAMPLE_RATE:.2f}s)"
    )
