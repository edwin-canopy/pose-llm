"""
One-time filter pass over hf_pose_dataset → hf_pose_dataset_filtered.

Drops:
  * rows with empty text (no word-level alignment)
  * rows where |audio_frames - pose_frames| > 1 (collator cannot align them)

Saves the result with save_to_disk so training can load_from_disk and start
the first step immediately, without paying the filter scan at launch time.
"""

import os
import sys
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import MERGED_DATASET_DIR


DATA_DIR = MERGED_DATASET_DIR
IN_DIR = f"{DATA_DIR}/hf_pose_dataset"
OUT_DIR = f"{DATA_DIR}/hf_pose_dataset_filtered"
POSE_CODEBOOKS = 8
NUM_PROC = int(os.environ.get("NUM_PROC", "1"))


def main():
    ds = load_from_disk(IN_DIR)
    n_in = len(ds)
    print(f"input:  {n_in:,} rows", flush=True)

    ds = ds.filter(
        lambda text: len(text) > 0,
        input_columns=["text"],
        num_proc=NUM_PROC,
        desc="drop empty text",
    )
    n_after_text = len(ds)
    print(
        f"  after empty-text filter: {n_after_text:,}  "
        f"(dropped {n_in - n_after_text:,})",
        flush=True,
    )

    ds = ds.filter(
        lambda audio, pose: abs(len(audio[0]) - len(pose) // POSE_CODEBOOKS) <= 1,
        input_columns=["audio_tokens", "pose_tokens"],
        num_proc=NUM_PROC,
        desc="drop audio/pose frame mismatch > 1",
    )
    n_after_align = len(ds)
    print(
        f"  after frame-mismatch filter: {n_after_align:,}  "
        f"(dropped {n_after_text - n_after_align:,})",
        flush=True,
    )

    # Materialise the filtered view onto a fresh on-disk arrow so training does
    # not have to walk the indices remap each row.
    os.makedirs(OUT_DIR, exist_ok=True)
    ds.save_to_disk(OUT_DIR)
    print(f"\nsaved {n_after_align:,} rows ({100*n_after_align/n_in:.2f}% kept) to {OUT_DIR}")


if __name__ == "__main__":
    main()
