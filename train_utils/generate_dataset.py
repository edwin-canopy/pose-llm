"""
One-time merge of the part-*.parquet shards with questions.parquet, joined on
row_id, saved as an HF Dataset to disk for fast load_from_disk() at train time.

Run once whenever the source parquet files change. Subsequent training launches
should hit OUT_DIR directly.
"""

import os
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import MERGED_DATASET_DIR


DATA_DIR = MERGED_DATASET_DIR
OUT_DIR = f"{DATA_DIR}/hf_pose_dataset"
NUM_PROC = int(os.environ.get("NUM_PROC", "1"))


def main():
    base = load_dataset(
        "parquet", data_files=f"{DATA_DIR}/part-*.parquet", split="train"
    )
    print(f"base:      {len(base):,} rows, cols={base.column_names}")

    questions = load_dataset(
        "parquet", data_files=f"{DATA_DIR}/questions.parquet", split="train"
    )
    print(f"questions: {len(questions):,} rows, cols={questions.column_names}")

    print("building row_id → question lookup...")
    question_map = dict(zip(questions["row_id"], questions["question"]))
    if len(question_map) != len(questions):
        raise ValueError(
            f"questions.parquet has duplicate row_ids: "
            f"{len(questions) - len(question_map)} duplicates"
        )
    print(f"  {len(question_map):,} unique row_ids")

    def attach(batch):
        # KeyError here means a base row_id has no matching question — fail loud.
        return {"question": [question_map[rid] for rid in batch["row_id"]]}

    merged = base.map(
        attach,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        desc="joining questions",
    )
    print(f"merged:    {len(merged):,} rows, cols={merged.column_names}")

    sample = merged[0]
    print(f"sample row[0]: row_id={sample['row_id']!r}  question={sample['question']!r}")

    os.makedirs(OUT_DIR, exist_ok=True)
    merged.save_to_disk(OUT_DIR)
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
