"""
Scaffolding for annotating the merged pose+audio dataset with Gemma 3 4B.

Loads the instruction-tuned Gemma 3 4B from HuggingFace and streams rows from
the merged parquet dataset at /mnt/somfs/pose_cond/merged_pose_audio_dataset.
The actual annotation task lives in `build_prompt` — replace the placeholder
prompt with whatever question-generation logic we land on.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_DIR = "/mnt/somfs/pose_cond/merged_pose_audio_dataset"
OUTPUT_PATH = "/mnt/somfs/pose_cond/question_annotations.parquet"
MODEL_ID = "google/gemma-3-4b-it"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 128
MAX_PROMPT_TOKENS = 4096
MAX_ROWS = 0  # 0 = process the full dataset; set small for smoke-testing


def load_model(model_id: str = MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched causal-LM generation

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    model.to("cuda")
    model.eval()
    return model, tokenizer


def load_streaming_dataset(dataset_dir: str = DATASET_DIR):
    files = sorted(glob.glob(os.path.join(dataset_dir, "part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet shards found in {dataset_dir}")
    return load_dataset(
        "parquet",
        data_files=files,
        split="train",
        streaming=True,
        columns=["row_id", "text"],
    )


def build_prompt(row: dict, tokenizer) -> str:
    # TODO: replace with the actual annotation prompt for the task.
    transcript = " ".join(w["word"] for w in row["text"])
    messages = [
        {
            "role": "user",
            "content": (
                "Read the following transcript and write one short question "
                "that a listener might ask the speaker.\n\n"
                f"Transcript:\n{transcript}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def generate(model, tokenizer, prompts: list[str]) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def main():
    model, tokenizer = load_model()
    ds = load_streaming_dataset()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("row_id", pa.string()), ("annotation", pa.string())])
    writer = pq.ParquetWriter(OUTPUT_PATH, schema)

    buf_ids: list[str] = []
    buf_prompts: list[str] = []
    rows_done = 0

    def flush():
        nonlocal rows_done
        if not buf_prompts:
            return
        outputs = generate(model, tokenizer, buf_prompts)
        writer.write_table(
            pa.table({"row_id": buf_ids, "annotation": outputs}, schema=schema)
        )
        rows_done += len(outputs)
        buf_ids.clear()
        buf_prompts.clear()
        if rows_done % (BATCH_SIZE * 25) == 0:
            print(f"wrote {rows_done:,} rows", flush=True)

    try:
        for row in ds:
            buf_ids.append(row["row_id"])
            buf_prompts.append(build_prompt(row, tokenizer))

            if len(buf_prompts) >= BATCH_SIZE:
                flush()

            if MAX_ROWS and rows_done >= MAX_ROWS:
                break

        flush()
    finally:
        writer.close()

    print(f"done. wrote {rows_done:,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
