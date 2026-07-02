"""
Write one parquet shard per audio source file into OUT_DIR. Each row is one
chunk with:
  row_id:       string
  text:         list[{word: str, start: float, end: float}]  (chunk-relative seconds, start at 0)
  audio_tokens: list[list[int32]]   (mimi codes, [NUM_AUDIO_CODEBOOKS x num_frames])
  pose_tokens:  list[int32]         (pose input_ids, pose-only / rescaled)

Sources (from dir.toml): pose parquet (input_ids) inner-joined on row_id
with the audio parquets (mimi_codes + per-word transcription).

We iterate one audio parquet at a time and `pose.take(...)` the matching
subset of pose rows, so peak RSS stays at roughly one shard's worth instead
of materializing + duplicating the whole 1.87M-row table at once.

The source `word_clip_*_seconds` are measured from the start of the *clip*,
not the chunk. ~9% of clips are split into multiple chunks (chunk_NNN), so we
subtract `chunk_start_seconds` to make timestamps chunk-relative.

Output is a directory of parquet shards (one per audio file), loadable as a
HuggingFace dataset via `load_dataset("parquet", data_files=f"{OUT_DIR}/*.parquet")`.
OUT_DIR is assumed to already exist and be writable.
"""

import functools
import gc
import glob
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001  uv run pipes stdout; without this, late output is lost on crash

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from paths import AUDIO_DATA_DIR, MERGED_DATASET_DIR, POSE_DATA_DIR

# Settings:
EXTRACT_ONLY_POSE_TOKENS = True
NUM_AUDIO_CODEBOOKS = 8
NUM_POSE_CODEBOOKS = 8

# Source-data layout (not user-tunable). Pose codes in the interleaved
# input_ids stream span [POSE_TOKEN_MIN, POSE_TOKEN_MAX]: 12 codebooks of
# size 1024 starting at 160140. Each frame contributes 12 pose codes in
# fixed codebook order (cb0..cb11), so rank-within-row mod 12 identifies
# the codebook. The audio side has 32 mimi codebooks per row.
POSE_TOKEN_MIN = 160140
POSE_TOKEN_MAX = 172427
POSE_CODEBOOKS_PER_FRAME = 12
POSE_CODEBOOK_SIZE = 1024
AUDIO_CODEBOOKS_PER_ROW = 32

OUT_DIR = MERGED_DATASET_DIR


def build_text_column(words_col, starts_col, ends_col, chunk_start_col):
    """Zip three aligned list columns into one list<struct{word, start, end}>,
    subtracting chunk_start_seconds so each row's timestamps start at 0."""
    words = words_col.combine_chunks()
    starts = starts_col.combine_chunks()
    ends = ends_col.combine_chunks()
    chunk_starts = chunk_start_col.combine_chunks()

    # Broadcast per-row chunk_start to per-word, then subtract from flat values.
    counts = pc.list_value_length(words).to_numpy(zero_copy_only=False)
    chunk_starts_np = chunk_starts.to_numpy(zero_copy_only=False)
    broadcast = np.repeat(chunk_starts_np, counts).astype(np.float32)

    elem_type = starts.values.type
    flat_starts = pa.array(
        starts.values.to_numpy(zero_copy_only=False) - broadcast, type=elem_type
    )
    flat_ends = pa.array(
        ends.values.to_numpy(zero_copy_only=False) - broadcast, type=elem_type
    )

    struct = pa.StructArray.from_arrays(
        [words.values, flat_starts, flat_ends],
        names=["word", "start", "end"],
    )

    if pa.types.is_large_list(words.type):
        return pa.LargeListArray.from_arrays(words.offsets, struct)
    return pa.ListArray.from_arrays(words.offsets, struct)


def _extract_pose_codes_chunk(chunk, n_codebooks_to_keep):
    """Process one chunk of input_ids: filter to pose codes, keep only the
    first n_codebooks_to_keep codebooks per frame, and rescale to raw
    codebook indices in [0, POSE_CODEBOOK_SIZE). Frame-major order."""
    offsets = chunk.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    values = chunk.values.to_numpy(zero_copy_only=False)

    in_pose = (values >= POSE_TOKEN_MIN) & (values <= POSE_TOKEN_MAX)
    pose_idx = np.flatnonzero(in_pose).astype(np.int64)

    # rank-of-pose-token within its row, then mod 12 = codebook
    pose_row = np.searchsorted(offsets, pose_idx, side="right") - 1
    row_first_pose_idx = np.searchsorted(pose_idx, offsets[:-1], side="left")
    rank_within_row = np.arange(len(pose_idx), dtype=np.int64) - row_first_pose_idx[pose_row]
    codebook = rank_within_row % POSE_CODEBOOKS_PER_FRAME

    keep_cb = codebook < n_codebooks_to_keep
    final_pose_idx = pose_idx[keep_cb]
    final_codebook = codebook[keep_cb].astype(values.dtype)

    final_keep = np.zeros(values.size, dtype=bool)
    final_keep[final_pose_idx] = True
    cumkeep = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(final_keep, dtype=np.int64)])
    new_offsets = cumkeep[offsets]

    bases = POSE_TOKEN_MIN + final_codebook * POSE_CODEBOOK_SIZE
    final_values = values[final_pose_idx] - bases

    return pa.LargeListArray.from_arrays(
        pa.array(new_offsets, type=pa.int64()),
        pa.array(final_values, type=chunk.values.type),
    )


def extract_pose_codes(input_ids_col, n_codebooks_to_keep):
    """Map a ChunkedArray of input_ids to a ChunkedArray of large_list<int32>
    holding only pose codes (first n_codebooks_to_keep codebooks per frame),
    rescaled to 0..POSE_CODEBOOK_SIZE-1. Processed chunk-by-chunk because the
    full flat values exceed int32 offset range (~4.8B values)."""
    chunks = input_ids_col.chunks if isinstance(input_ids_col, pa.ChunkedArray) else [input_ids_col]
    return pa.chunked_array(
        [_extract_pose_codes_chunk(c, n_codebooks_to_keep) for c in chunks]
    )


def _truncate_audio_codebooks_chunk(chunk, n_keep):
    """Keep the first n_keep outer entries (codebooks) of each row."""
    offsets = chunk.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    row_lens = np.diff(offsets)
    if not np.all(row_lens >= n_keep):
        raise ValueError(f"some rows have fewer than {n_keep} audio codebooks")

    n_rows = len(offsets) - 1
    starts = offsets[:-1]
    take_idx = (starts[:, None] + np.arange(n_keep, dtype=np.int64)[None, :]).ravel()
    new_inner = chunk.values.take(pa.array(take_idx))
    new_offsets = np.arange(n_rows + 1, dtype=np.int64) * n_keep
    return pa.LargeListArray.from_arrays(
        pa.array(new_offsets, type=pa.int64()), new_inner
    )


def truncate_audio_codebooks(mimi_col, n_keep):
    """Map mimi_codes (large_list<large_list<int32>>, 32 codebooks per row)
    to the same shape with only the first n_keep codebooks per row."""
    chunks = mimi_col.chunks if isinstance(mimi_col, pa.ChunkedArray) else [mimi_col]
    return pa.chunked_array(
        [_truncate_audio_codebooks_chunk(c, n_keep) for c in chunks]
    )


def main():
    pose_files = sorted(glob.glob(f"{POSE_DATA_DIR}/*.parquet"))
    audio_files = sorted(glob.glob(f"{AUDIO_DATA_DIR}/*.parquet"))
    print(f"pose:  {len(pose_files)} file(s) under {POSE_DATA_DIR}")
    print(f"audio: {len(audio_files)} file(s) under {AUDIO_DATA_DIR}")

    out_dir = Path(OUT_DIR)

    # Load pose once and keep resident: it's a single file with the smaller
    # of the two columns we need (input_ids only, no 32-codebook lists),
    # and we'll slice it per audio shard via take(). Building a row_id ->
    # pose_row_idx map lets each shard look up its matching pose rows in O(1).
    print("loading pose columns (row_id, input_ids) once...")
    pose = pads.dataset(pose_files, format="parquet").to_table(
        columns=["row_id", "input_ids"]
    )
    # Source schema is list<int32> (32-bit offsets). pose.take(...) below
    # concatenates all chunks into a single contiguous array per column, and
    # the flat values count (~4.8B) overflows int32 offsets. Cast to
    # large_list<int32> so take() can build 64-bit offsets. extract_pose_codes
    # already returns LargeListArray and reads .offsets/.values generically,
    # so downstream is unaffected.
    input_ids_large = pose["input_ids"].cast(pa.large_list(pa.int32()))
    pose = pa.table({"row_id": pose["row_id"], "input_ids": input_ids_large})
    print(f"  pose rows={pose.num_rows:,}")
    pose_idx_map = {rid: i for i, rid in enumerate(pose["row_id"].to_pylist())}

    total_written = 0
    for shard_idx, audio_file in enumerate(audio_files):
        shard_name = Path(audio_file).stem
        print(f"\n[{shard_idx + 1}/{len(audio_files)}] {shard_name}")

        audio = pads.dataset([audio_file], format="parquet").to_table(
            columns=[
                "row_id",
                "chunk_start_seconds",
                "mimi_codes",
                "word_texts",
                "word_clip_start_seconds",
                "word_clip_end_seconds",
            ]
        )
        print(f"  audio rows={audio.num_rows:,}")

        # Inner-join: keep only audio rows whose row_id is in pose.
        audio_row_ids = audio["row_id"].to_pylist()
        pose_indices = np.fromiter(
            (pose_idx_map.get(rid, -1) for rid in audio_row_ids),
            dtype=np.int64,
            count=len(audio_row_ids),
        )
        missing = int((pose_indices == -1).sum())
        if missing:
            print(f"  dropping {missing} audio rows with no pose match")
            keep = pose_indices != -1
            audio = audio.filter(pa.array(keep))
            pose_indices = pose_indices[keep]
        pose_shard = pose.select(["input_ids"]).take(pa.array(pose_indices))
        print(f"  aligned rows={audio.num_rows:,}")

        text_col = build_text_column(
            audio["word_texts"],
            audio["word_clip_start_seconds"],
            audio["word_clip_end_seconds"],
            audio["chunk_start_seconds"],
        )
        if EXTRACT_ONLY_POSE_TOKENS:
            pose_tokens_col = extract_pose_codes(pose_shard["input_ids"], NUM_POSE_CODEBOOKS)
        else:
            pose_tokens_col = pose_shard["input_ids"]
        audio_tokens_col = truncate_audio_codebooks(audio["mimi_codes"], NUM_AUDIO_CODEBOOKS)

        shard_table = pa.table(
            {
                "row_id": audio["row_id"],
                "text": text_col,
                "audio_tokens": audio_tokens_col,
                "pose_tokens": pose_tokens_col,
            }
        )

        out_path = out_dir / f"{shard_name}.parquet"
        print(f"  writing {out_path} ({shard_table.num_rows:,} rows)")
        pq.write_table(shard_table, out_path)
        total_written += shard_table.num_rows

        # Drop per-shard buffers before next iteration so RSS doesn't drift up.
        del audio, pose_shard, text_col, pose_tokens_col, audio_tokens_col, shard_table
        gc.collect()

    print(f"\ndone. wrote {total_written:,} rows across {len(audio_files)} shards to {out_dir}")


if __name__ == "__main__":
    main()
