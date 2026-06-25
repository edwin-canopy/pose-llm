"""
Scaffolding for annotating the merged pose+audio dataset with Gemma 3 4B.

Loads the instruction-tuned Gemma 3 4B from HuggingFace and streams rows from
the merged parquet dataset at /mnt/somfs/pose_cond/merged_pose_audio_dataset.
The actual annotation task lives in `build_prompt` — replace the placeholder
prompt with whatever question-generation logic we land on.
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_DIR = "/mnt/somfs/pose_cond/merged_pose_audio_dataset"
OUTPUT_PATH = "/mnt/somfs/pose_cond/merged_pose_audio_dataset/questions.parquet"
MODEL_ID = "google/gemma-3-4b-it"

NUM_PROCESSES = 5

BATCH_SIZE = 16
MAX_NEW_TOKENS = 128
MAX_PROMPT_TOKENS = 4096


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


def load_streaming_dataset(files: list[str]):
    return load_dataset(
        "parquet",
        data_files=files,
        split="train",
        streaming=True,
        columns=["row_id", "text"],
    )


def build_prompt(row: dict, tokenizer) -> str:
    transcript = " ".join(w["word"] for w in row["text"])
    messages = [
        {
            "role": "user",
            "content": (
                "Read the following transcript and write one short question "
                "that might have motivated this particular transcript. "
                "The question should come before the provided transcript in logical order "
                "and it should be exactly one sentence. Return this question only "
                "and nothing else.\n\n"
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


def extract_text(row: dict) -> str:
    return " ".join(w["word"] for w in row["text"])


def worker_fn(worker_idx: int, shard_files: list[str], out_path: str) -> None:
    model, tokenizer = load_model()
    ds = load_streaming_dataset(shard_files)

    schema = pa.schema([("row_id", pa.string()), ("question", pa.string())])
    writer = pq.ParquetWriter(out_path, schema)

    buf_ids: list[str] = []
    buf_prompts: list[str] = []
    rows_done = 0

    def flush():
        nonlocal rows_done
        if not buf_prompts:
            return
        outputs = generate(model, tokenizer, buf_prompts)
        writer.write_table(
            pa.table({"row_id": buf_ids, "question": outputs}, schema=schema)
        )
        rows_done += len(outputs)
        buf_ids.clear()
        buf_prompts.clear()
        if rows_done % (BATCH_SIZE * 25) == 0:
            print(f"[worker {worker_idx}] {rows_done:,} rows", flush=True)

    try:
        for row in ds:
            buf_ids.append(row["row_id"])
            buf_prompts.append(build_prompt(row, tokenizer))
            if len(buf_prompts) >= BATCH_SIZE:
                flush()
        flush()
    finally:
        writer.close()

    print(f"[worker {worker_idx}] done — {rows_done:,} rows → {out_path}", flush=True)


def shard_chunks(files: list[str], n: int) -> list[list[str]]:
    """Split files into n contiguous chunks; last chunk may be smaller."""
    size = (len(files) + n - 1) // n
    chunks = [files[i * size:(i + 1) * size] for i in range(n)]
    return [c for c in chunks if c]


def recombine(part_paths: list[str], output_path: str) -> None:
    """Concatenate per-worker parquet files in order then delete them."""
    schema = pa.schema([("row_id", pa.string()), ("question", pa.string())])
    writer = pq.ParquetWriter(output_path, schema)
    total = 0
    try:
        for path in part_paths:
            tbl = pq.read_table(path)
            writer.write_table(tbl)
            total += len(tbl)
    finally:
        writer.close()
    for path in part_paths:
        os.remove(path)
    print(f"recombined {len(part_paths)} parts → {output_path} ({total:,} rows total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run on 10 rows and print results; skip writing parquet")
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "part-*.parquet")))
    if not all_files:
        raise FileNotFoundError(f"No parquet shards found in {DATASET_DIR}")

    if args.test:
        model, tokenizer = load_model()
        ds = load_streaming_dataset(all_files)
        rows = list(ds.take(10))
        prompts = [build_prompt(r, tokenizer) for r in rows]
        questions = generate(model, tokenizer, prompts)
        for row, question in zip(rows, questions):
            print(f"[{row['row_id']}]")
            print(f"TEXT:     {extract_text(row)}")
            print(f"QUESTION: {question}")
            print()
        return

    chunks = shard_chunks(all_files, NUM_PROCESSES)
    actual = len(chunks)
    part_paths = [
        os.path.join(DATASET_DIR, f"questions_worker_{i:02d}.parquet")
        for i in range(actual)
    ]

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=worker_fn, args=(i, chunk, part_paths[i]), daemon=False)
        for i, chunk in enumerate(chunks)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Workers {failed} exited with non-zero status — aborting recombine")

    recombine(part_paths, OUTPUT_PATH)


if __name__ == "__main__":
    main()
