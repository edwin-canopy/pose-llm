from __future__ import annotations

import csv
import hashlib
import random
import sys
import warnings
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .kinematic import (
    SELECTED_INDICES,
    NUM_JOINTS,
    ROOT_LEFT,
    ROOT_RIGHT,
    positions_to_offsets,
)

MIN_SHOULDER_WIDTH_NORM = 0.02  # ~5px at 240w — skip clips with tiny/broken detections

_MAX_SKIP = 64
_COLLATE_LONG_T_WARN = 4096
_collate_long_t_warned = False


def _process_poses_to_clip(
    poses: np.ndarray,
    orig_h: float,
    orig_w: float,
    norm_center: np.ndarray | None,
    norm_scale: np.ndarray | None,
) -> dict[str, np.ndarray] | None:
    """Apply kinematic-chain processing to a dense (T, 133, 3) float32 array.

    Returns None if the clip should be skipped (shoulder width too small).
    """
    T = poses.shape[0]
    if T == 0:
        return None

    selected = poses[:, SELECTED_INDICES]                  # (T, NUM_JOINTS, 3)
    positions = selected[..., :2].astype(np.float32, copy=True)
    positions[..., 0] /= orig_h
    positions[..., 1] /= orig_w
    confidence = selected[..., 2].astype(np.float32, copy=True)

    # Per-frame shoulder width (x-distance in normalised coords).
    shoulder_width = np.abs(
        positions[:, ROOT_LEFT, 1] - positions[:, ROOT_RIGHT, 1]
    )
    median_sw = float(np.median(shoulder_width))

    if median_sw < MIN_SHOULDER_WIDTH_NORM:
        return None

    # Root = mid-shoulder
    root = (positions[:, ROOT_LEFT, :] + positions[:, ROOT_RIGHT, :]) / 2.0

    _, kc_keypoints = positions_to_offsets(positions)      # (T, 50, 2)
    kc_keypoints /= median_sw
    keypoints_flat = kc_keypoints.reshape(T, -1)           # (T, 100)

    if norm_center is not None and norm_scale is not None:
        keypoints_flat = (keypoints_flat - norm_center) / norm_scale

    global_state = np.stack(
        [root[:, 0], root[:, 1], shoulder_width], axis=-1
    )  # (T, 3)

    return {
        "keypoints": keypoints_flat,
        "confidence": confidence,
        "global_state": global_state,
    }


def _append_confidence_features(
    keypoints_flat: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Return interleaved per-joint features ``(y, x, conf)``.

    The legacy model input is ``(T, NUM_JOINTS * 2)`` with all coordinates
    flattened as ``[j0_y, j0_x, j1_y, j1_x, ...]``. Confidence-aware models use
    the same joint order but append each joint's detector confidence beside its
    coordinates: ``[j0_y, j0_x, j0_conf, j1_y, ...]``.
    """
    T = keypoints_flat.shape[0]
    coords = keypoints_flat.reshape(T, NUM_JOINTS, 2)
    features = np.concatenate([coords, confidence[..., None]], axis=-1)
    return features.reshape(T, NUM_JOINTS * 3).astype(np.float32, copy=False)


def load_clip(
    path: Path,
    orig_h: float | None = None,
    orig_w: float | None = None,
    norm_center: np.ndarray | None = None,
    norm_scale: np.ndarray | None = None,
    start: int | None = None,
    length: int | None = None,
) -> dict[str, np.ndarray] | None:
    """Load a dense pose file and return scale-normalised kinematic-chain keypoints.

    The on-disk format is uncompressed ``.npy`` of shape ``(T, 133, 3)``
    ``float32`` (no person axis — solo-signer dataset). We memmap and slice
    so only the requested window is paged in.

    Args:
        path:    Path to the ``.npy`` (or legacy ``.npz``) pose file.
        orig_h, orig_w: Frame dimensions in pixels for image-normalised coords.
                  Must be supplied externally (e.g. dims.csv) for the .npy path
                  since the dense format only contains poses.
        norm_center, norm_scale: Optional per-dimension z-score standardisation.
        start:   Optional source-frame start index. If None, the entire clip
                  is processed (clip mode / scanning).
        length:  Optional number of source frames to process. Required if
                  ``start`` is given; ignored otherwise.

    Returns None if the clip should be skipped (shoulder width too small,
    file unreadable, or requested window out of range).
    """
    is_npy = str(path).endswith(".npy")

    if is_npy:
        # Dense uncompressed format — requires dims from dims.csv (file has poses only).
        if orig_h is None or orig_w is None:
            return None
        try:
            arr = np.load(path, mmap_mode="r")
        except FileNotFoundError:
            raise
        except Exception:
            return None
        if start is not None and length is not None:
            T = arr.shape[0]
            if start < 0 or length <= 0 or start + length > T:
                return None
            poses = np.array(arr[start : start + length], dtype=np.float32, copy=True)
        else:
            poses = np.asarray(arr, dtype=np.float32)
    else:
        # Legacy compressed NPZ (e.g. test_set/ before conversion). May carry
        # its own orig_h/orig_w; supplied dims override if present.
        try:
            with np.load(path, allow_pickle=True) as data:
                poses = np.asarray(data["poses"], dtype=np.float32)
                if orig_h is None or orig_w is None:
                    orig_h = float(data["orig_h"])
                    orig_w = float(data["orig_w"])
        except FileNotFoundError:
            raise
        except Exception:
            return None
        if start is not None and length is not None:
            if start < 0 or length <= 0 or start + length > poses.shape[0]:
                return None
            poses = poses[start : start + length]

    if poses.ndim != 3 or poses.shape[1:] != (133, 3):
        return None

    return _process_poses_to_clip(
        poses, float(orig_h), float(orig_w), norm_center, norm_scale,
    )


# ---------------------------------------------------------------------------
# Clip length scanning (for segment-mode window index)
# ---------------------------------------------------------------------------

def _read_clip_length(path: Path) -> int:
    """Read just the number of frames from a pose file.

    For dense .npy (the default uncompressed format), this is essentially
    free — only the npy header (~80 bytes) is touched. For legacy .npz the
    full ``poses`` member's npy header is parsed via NpzFile.
    """
    try:
        if str(path).endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            return int(arr.shape[0])
        with np.load(path, allow_pickle=True) as data:
            return int(data["poses"].shape[0])
    except Exception:
        return 0


def _scan_clip_lengths(
    files: list[Path],
    cache_dir: Path | None = None,
    num_threads: int = 100,
) -> np.ndarray:
    """Return an int32 array of frame counts, one per file.  Caches to disk."""
    n = len(files)
    tag = f"{n}_{files[0].stem[:20]}_{files[-1].stem[:20]}" if n else "empty"
    cache_key = hashlib.md5(tag.encode()).hexdigest()[:10]
    cache_name = f".segment_lengths_{cache_key}.npy"

    search_dirs = [d for d in [cache_dir, Path(".")] if d is not None]
    for loc in search_dirs:
        cp = loc / cache_name
        if cp.is_file():
            arr = np.load(cp)
            if arr.shape == (n,):
                return arr

    print(f"Scanning {n:,} clip lengths ({num_threads} threads)…", file=sys.stderr, flush=True)
    lengths = np.zeros(n, dtype=np.int32)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futs = {pool.submit(_read_clip_length, files[i]): i for i in range(n)}
        done = 0
        for fut in as_completed(futs):
            lengths[futs[fut]] = fut.result()
            done += 1
            if done % 20_000 == 0:
                print(f"  {done:,}/{n:,}", file=sys.stderr, flush=True)

    valid = lengths[lengths > 0]
    print(
        f"  Done — {len(valid):,}/{n:,} readable, "
        f"median T={int(np.median(valid)) if len(valid) else 0}",
        file=sys.stderr, flush=True,
    )

    for loc in search_dirs:
        try:
            np.save(loc / cache_name, lengths)
            break
        except OSError:
            continue

    return lengths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _resample_clip(
    clip: dict[str, np.ndarray],
    source_fps: int,
    target_fps: int,
) -> dict[str, np.ndarray]:
    """Downsample all arrays in *clip* from source_fps to target_fps by selecting
    evenly-spaced frames (no interpolation — keeps original detections intact)."""
    if source_fps == target_fps:
        return clip
    T_src = clip["keypoints"].shape[0]
    T_tgt = round(T_src * target_fps / source_fps)
    if T_tgt >= T_src or T_tgt < 2:
        return clip
    indices = np.round(np.linspace(0, T_src - 1, T_tgt)).astype(int)
    return {key: arr[indices] for key, arr in clip.items()}


class PoseDataset(Dataset):
    """Clip-level or segment-level dataset of pose .npz files.

    **Clip mode** (default): each sample is one full .npz file.
    **Segment mode** (``segment_frames > 0``): each clip is sliced into
    non-overlapping windows of ``segment_frames`` frames, and every window
    becomes its own sample.  This fully utilises long clips and yields
    uniform-length batches (except for short tail segments, padded by collate).

    Args:
        data_dir:        Root directory containing the .npz files.
        manifest:        Text file with one relative .npz path per line.
        dims_csv:        CSV with columns ``clip_stem``, ``height``, ``width``.
        source_fps:      FPS of the raw .npz data.
        target_fps:      Resample to this FPS before segmentation/training.
        segment_frames:  Fixed window size (target_fps frames). 0 = full clip.
        max_frames:      Legacy clip-level cap (ignored when segment_frames > 0).
        random_crop:     Random window for max_frames (ignored in segment mode).
        random_segment_offset:
            When True (default for training), each __getitem__ in segment mode
            samples a uniform random start in [0, T - segment_frames] at access
            time instead of using the fixed multiples-of-segment_frames index.
            This is essential for fully-convolutional encoders: without it, the
            model only ever sees windows aligned to multiples of segment_frames,
            so frames at within-window position p are *always* the (k*seg + p)-th
            frame of their source clip, biasing the learned edge-prediction
            behaviour. Set to False for validation to keep deterministic samples.
        file_indices:    Subset of manifest indices to use (for train/val split).
    """

    def __init__(
        self,
        data_dir: str | Path,
        manifest: str | Path | None = None,
        dims_csv: str | Path | None = None,
        source_fps: int = 30,
        target_fps: int = 30,
        segment_frames: int | None = None,
        max_frames: int | None = None,
        random_crop: bool = False,
        random_segment_offset: bool = False,
        prefix_frames: int = 0,
        file_indices: list[int] | None = None,
        norm_stats: str | Path | None = None,
        load_cache_size: int = 256,
        include_confidence: bool = False,
    ):
        # Per-worker LRU cache of processed clips. With random_segment_offset
        # the same file can be sampled multiple times across nearby batches —
        # caching the full processed result (kinematic offsets, normalisation,
        # etc.) lets subsequent segment accesses skip all that work. Workers
        # are forked once (persistent_workers=True) so the cache persists for
        # the lifetime of the worker. ~1-2MB/clip × 256 ≈ 250-500MB per worker.
        self._load_cache: OrderedDict[int, dict] = OrderedDict()
        self._load_cache_size = load_cache_size
        self.include_confidence = include_confidence
        self.data_dir = Path(data_dir)
        self.norm_center: np.ndarray | None = None
        self.norm_scale: np.ndarray | None = None
        if norm_stats is not None:
            ns = Path(norm_stats)
            if ns.is_file():
                st = np.load(ns)
                self.norm_center = st["center"].astype(np.float32)
                self.norm_scale = st["scale"].astype(np.float32)
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.max_frames = max_frames if (max_frames is not None and max_frames > 0) else None
        self.random_crop = bool(random_crop) and self.max_frames is not None
        self.segment_frames = segment_frames if (segment_frames is not None and segment_frames > 0) else None
        self.random_segment_offset = bool(random_segment_offset) and self.segment_frames is not None
        # Causal warmup prefix (target-fps frames prepended before the supervised
        # window). Only meaningful in segment mode. See TrainConfig.prefix_frames.
        self.prefix_frames = max(0, int(prefix_frames)) if self.segment_frames is not None else 0
        # Every segment sample reads EXACTLY this many real frames (prefix +
        # supervised), so batches are uniform-length and require NO zero padding.
        # Zero-padded frames would feed degenerate all-zero vectors into the
        # quantizer's L2 normalisation and produce NaN gradients, so we avoid
        # padding entirely by only emitting windows that fully fit in a clip.
        self.read_frames = (
            self.segment_frames + self.prefix_frames
            if self.segment_frames is not None
            else None
        )

        # Load full manifest
        if manifest is not None:
            with open(manifest) as f:
                all_files = [self.data_dir / line.strip() for line in f if line.strip()]
        else:
            all_files = sorted(self.data_dir.rglob("*_poses.npy"))
            if not all_files:
                all_files = sorted(self.data_dir.rglob("*_poses.npz"))

        if not all_files:
            raise FileNotFoundError(
                f"No files found. data_dir={data_dir}, manifest={manifest}"
            )

        # Apply file subset (train/val split happens in train.py)
        if file_indices is not None:
            self.files = [all_files[i] for i in file_indices]
        else:
            self.files = all_files

        # Dims lookup
        self.dims: dict[str, tuple[float, float]] = {}
        if dims_csv is not None:
            with open(dims_csv) as f:
                for row in csv.DictReader(f):
                    self.dims[row["clip_stem"]] = (
                        float(row["height"]),
                        float(row["width"]),
                    )

        # Build segment window index when segment_frames is set.
        # Scan lengths once for the full manifest (shared cache across splits).
        self._windows: list[tuple[int, int]] | None = None
        if self.segment_frames is not None:
            all_lengths = _scan_clip_lengths(all_files, cache_dir=self.data_dir)
            if file_indices is not None:
                lengths = np.array([all_lengths[i] for i in file_indices])
            else:
                lengths = all_lengths
            # Segment windows are in target_fps frames
            if self.source_fps != self.target_fps:
                ratio = self.target_fps / self.source_fps
                lengths = np.round(lengths * ratio).astype(np.int32)
            self._build_segment_index(lengths)

    # -- segment helpers ---------------------------------------------------

    def _build_segment_index(self, lengths: np.ndarray) -> None:
        """Emit only fully-contained windows. The trailing partial segment of
        each clip — and any clip shorter than ``segment_frames`` — is dropped
        rather than zero-padded. Padding tails forced the tokenizer to learn
        the artificial (0,0) endings; dropping them costs a few frames per
        clip but keeps every training sample's signal real."""
        # Windows are sized by the full read length (prefix + supervised) so that
        # every emitted window fully fits in its clip with no padding. Clips
        # shorter than the read length produce no windows (dropped, not padded).
        read = self.read_frames
        self._windows = []
        for file_idx, t in enumerate(lengths):
            # range stops at the last `start` for which `start + read <= t`,
            # so `T < read` naturally produces no windows.
            for start in range(0, int(t) - read + 1, read):
                self._windows.append((file_idx, start))

    # -- core interface ----------------------------------------------------

    def __len__(self) -> int:
        if self._windows is not None:
            return len(self._windows)
        return len(self.files)

    def _dims_for(self, file_idx: int) -> tuple[float | None, float | None]:
        path = self.files[file_idx]
        stem = path.stem
        # filenames are "<clip>_poses.(npy|npz)"; dims.csv keys are "<clip>".
        if stem.endswith("_poses"):
            stem = stem[: -len("_poses")]
        return self.dims.get(stem, (None, None))

    def _load_clip(self, file_idx: int) -> dict[str, np.ndarray] | None:
        """Whole-clip load. Used by clip-mode __getitem__ (legacy)."""
        if file_idx in self._load_cache:
            self._load_cache.move_to_end(file_idx)
            return self._load_cache[file_idx]

        path = self.files[file_idx]
        orig_h, orig_w = self._dims_for(file_idx)
        clip = load_clip(
            path, orig_h=orig_h, orig_w=orig_w,
            norm_center=self.norm_center, norm_scale=self.norm_scale,
        )
        if clip is not None and self.source_fps != self.target_fps:
            clip = _resample_clip(clip, self.source_fps, self.target_fps)

        if clip is not None and self._load_cache_size > 0:
            self._load_cache[file_idx] = clip
            if len(self._load_cache) > self._load_cache_size:
                self._load_cache.popitem(last=False)
        return clip

    def _load_segment(
        self, file_idx: int, target_start: int, target_length: int,
    ) -> dict[str, np.ndarray] | None:
        """Memmap the clip and process only the requested ``target_length``-frame
        window starting at ``target_start`` (in *target_fps* units).

        For .npy files this reads only the bytes for the chosen frames — the
        whole-clip processing path is skipped.
        """
        path = self.files[file_idx]
        orig_h, orig_w = self._dims_for(file_idx)

        if self.source_fps == self.target_fps:
            src_start = target_start
            src_length = target_length
        else:
            # Conservative envelope: read enough source frames to cover the
            # requested target window after evenly-spaced resampling, then
            # resample down to exactly target_length frames.
            ratio = self.source_fps / self.target_fps
            src_start = int(target_start * ratio)
            src_end = int(np.ceil((target_start + target_length) * ratio))
            src_length = src_end - src_start

        clip = load_clip(
            path, orig_h=orig_h, orig_w=orig_w,
            norm_center=self.norm_center, norm_scale=self.norm_scale,
            start=src_start, length=src_length,
        )
        if clip is None:
            return None

        if self.source_fps != self.target_fps:
            # Resample the source window down to exactly target_length frames.
            T_src = clip["keypoints"].shape[0]
            if T_src < target_length:
                return None
            indices = np.round(np.linspace(0, T_src - 1, target_length)).astype(int)
            clip = {k: v[indices] for k, v in clip.items()}

        return clip

    def _model_keypoints(self, clip: dict[str, np.ndarray]) -> np.ndarray:
        if not self.include_confidence:
            return clip["keypoints"]
        return _append_confidence_features(clip["keypoints"], clip["confidence"])

    def _to_tensors(self, clip: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            "keypoints": torch.from_numpy(self._model_keypoints(clip)),
            "confidence": torch.from_numpy(clip["confidence"]),
            "global_state": torch.from_numpy(clip["global_state"]),
        }

    # -- segment-mode __getitem__ ------------------------------------------

    def _getitem_segment(self, idx: int) -> dict[str, torch.Tensor]:
        n = len(self._windows)
        seg = self.segment_frames
        read = self.read_frames        # seg + prefix (== seg when no prefix)
        n_prefix = read - seg          # warmup frames (constant across samples)
        # A processing failure for a given file (tiny shoulder width, malformed
        # shape, etc.) is permanent regardless of which window we ask for, so
        # retrying other windows of the same file is wasted work. Skip the
        # entire file on first failure and cap retries by distinct bad files.
        bad_files: set[int] = set()
        i = 0
        while i < n and len(bad_files) < _MAX_SKIP:
            file_idx, indexed_start = self._windows[(idx + i) % n]
            i += 1
            if file_idx in bad_files:
                continue

            # Pick the target-fps read start without loading data first. We need
            # the clip's target-fps length for random_segment_offset;
            # _read_clip_length on .npy is essentially free (header only). Every
            # read is EXACTLY ``read`` frames so batches are uniform (no padding).
            if self.random_segment_offset or indexed_start > 0:
                T_src = _read_clip_length(self.files[file_idx])
                if T_src <= 0:
                    bad_files.add(file_idx)
                    continue
                if self.source_fps != self.target_fps:
                    T_tgt = int(round(T_src * self.target_fps / self.source_fps))
                else:
                    T_tgt = T_src
                if T_tgt < read:
                    bad_files.add(file_idx)
                    continue
                if self.random_segment_offset:
                    read_start = random.randint(0, T_tgt - read)
                else:
                    read_start = min(indexed_start, T_tgt - read)
            else:
                read_start = 0

            clip = self._load_segment(file_idx, read_start, read)
            if clip is None:
                bad_files.add(file_idx)
                continue

            return {
                "keypoints": torch.from_numpy(self._model_keypoints(clip)),
                "confidence": torch.from_numpy(clip["confidence"]),
                "global_state": torch.from_numpy(clip["global_state"]),
                "loss_mask": self._make_loss_mask(read, n_prefix),
            }
        raise RuntimeError(
            f"Could not load a segment after {len(bad_files)} distinct bad files. "
            f"Example: {self.files[self._windows[idx % n][0]]}"
        )

    @staticmethod
    def _make_loss_mask(length: int, n_prefix: int) -> torch.Tensor:
        """1.0 on the supervised tail, 0.0 on the ``n_prefix`` warmup frames."""
        mask = torch.ones(length, dtype=torch.float32)
        if n_prefix > 0:
            mask[:n_prefix] = 0.0
        return mask

    # -- clip-mode __getitem__ ---------------------------------------------

    def _getitem_clip(self, idx: int) -> dict[str, torch.Tensor]:
        n = len(self.files)
        for attempt in range(min(_MAX_SKIP, n)):
            clip = self._load_clip((idx + attempt) % n)
            if clip is None:
                continue
            out = self._to_tensors(clip)
            kp, conf, gs = out["keypoints"], out["confidence"], out["global_state"]
            mf = self.max_frames
            if mf is not None and kp.shape[0] > mf:
                if self.random_crop:
                    s = random.randrange(0, kp.shape[0] - mf + 1)
                    sl = slice(s, s + mf)
                    kp, conf, gs = kp[sl], conf[sl], gs[sl]
                else:
                    kp, conf, gs = kp[:mf], conf[:mf], gs[:mf]
            return {
                "keypoints": kp,
                "confidence": conf,
                "global_state": gs,
                "loss_mask": torch.ones(kp.shape[0], dtype=torch.float32),
            }
        raise RuntimeError(
            f"Could not load a clip after {_MAX_SKIP} tries. "
            f"Example: {self.files[idx % n]}"
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._windows is not None:
            return self._getitem_segment(idx)
        return self._getitem_clip(idx)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def make_collate_fn(downsampling_factor: int):
    """Return a collate function that pads clips to a length divisible by *downsampling_factor*."""

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        global _collate_long_t_warned
        max_t = max(b["keypoints"].shape[0] for b in batch)
        padded_t = ((max_t + downsampling_factor - 1) // downsampling_factor) * downsampling_factor
        if padded_t > _COLLATE_LONG_T_WARN and not _collate_long_t_warned:
            _collate_long_t_warned = True
            warnings.warn(
                f"A training batch is being padded to T={padded_t} because at least one "
                f"clip in the batch is that long. That makes the first forward very slow "
                f"and can look like a hang. Consider shorter clips, a smaller batch_size, "
                f"or bucketing by length.",
                stacklevel=2,
            )

        has_loss_mask = "loss_mask" in batch[0]
        out: dict[str, list] = defaultdict(list)
        for b in batch:
            pad = padded_t - b["keypoints"].shape[0]
            out["keypoints"].append(F.pad(b["keypoints"], (0, 0, 0, pad)))
            out["confidence"].append(F.pad(b["confidence"], (0, 0, 0, pad)))
            out["global_state"].append(F.pad(b["global_state"], (0, 0, 0, pad)))
            if has_loss_mask:
                # 1D (T,) mask; right-pad with 0 so padded frames are unsupervised.
                out["loss_mask"].append(F.pad(b["loss_mask"], (0, pad)))

        collated = {
            "keypoints": torch.stack(out["keypoints"]),
            "confidence": torch.stack(out["confidence"]),
            "global_state": torch.stack(out["global_state"]),
        }
        if has_loss_mask:
            collated["loss_mask"] = torch.stack(out["loss_mask"])
        return collated

    return collate_fn
