"""Video rendering utilities for side-by-side pose comparison.

Two modes:
  1. **Video overlay** (``video_path`` provided): skeleton drawn over original frames.
  2. **Skeleton only** (no video): skeleton on a black canvas.

Both produce a side-by-side MP4: original (green, left) vs reconstructed (red, right).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import _resample_clip, load_clip
from .kinematic import (
    NUM_JOINTS,
    offsets_to_positions,
    FACE_CONNECTIONS,
    BODY_CONNECTIONS,
    LEFT_HAND_CONNECTIONS,
    RIGHT_HAND_CONNECTIONS,
)

_ALL_CONNECTIONS = (
    FACE_CONNECTIONS + BODY_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS
)
_ALL_JOINTS = sorted({idx for pair in _ALL_CONNECTIONS for idx in pair})

_GREEN = (0, 220, 0)
_RED = (220, 50, 50)
_LABEL_H = 40
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def kp_flat_to_positions(kp_flat: np.ndarray, global_state: np.ndarray) -> np.ndarray:
    """Convert normalised kinematic offsets ``(T, 110)`` back to absolute positions ``(T, 55, 2)``."""
    kc_offsets = kp_flat.reshape(-1, NUM_JOINTS, 2)
    shoulder_width = global_state[:, 2:3, np.newaxis]
    kc_offsets = kc_offsets * shoulder_width
    root = global_state[:, :2]
    return offsets_to_positions(root, kc_offsets)


def _draw_skeleton(
    img: np.ndarray,
    kpts_norm: np.ndarray,
    color: tuple[int, int, int],
    conf: np.ndarray | None = None,
    conf_thresh: float = 0.3,
    thickness: int = 2,
    radius: int = 3,
) -> np.ndarray:
    """Draw a monochrome skeleton on an RGB image (mutates *img*)."""
    h, w = img.shape[:2]
    y_px = kpts_norm[:, 0] * h
    x_px = kpts_norm[:, 1] * w
    visible = conf >= conf_thresh if conf is not None else np.ones(NUM_JOINTS, dtype=bool)

    for i, j in _ALL_CONNECTIONS:
        if visible[i] and visible[j]:
            cv2.line(
                img,
                (int(x_px[i]), int(y_px[i])),
                (int(x_px[j]), int(y_px[j])),
                color, thickness, cv2.LINE_AA,
            )
    for idx in _ALL_JOINTS:
        if visible[idx]:
            cv2.circle(img, (int(x_px[idx]), int(y_px[idx])), radius, color, -1, cv2.LINE_AA)
    return img


def render_comparison_video(
    orig_kp: np.ndarray,
    recon_kp: np.ndarray,
    global_state: np.ndarray,
    confidence: np.ndarray,
    out_path: str | Path,
    *,
    video_path: str | Path | None = None,
    fps: float = 25.0,
    max_seconds: float = 10.0,
    canvas_size: tuple[int, int] = (960, 540),
    conf_thresh: float = 0.3,
    recon_confidence: np.ndarray | None = None,
) -> Path:
    """Render a side-by-side comparison video (original vs reconstructed).

    Args:
        orig_kp:      ``(T, 100)`` original normalised keypoints.
        recon_kp:     ``(T', 100)`` reconstructed keypoints from the model.
        global_state: ``(T, 3)`` per-frame root + shoulder width.
        confidence:   ``(T, 55)`` per-joint confidence scores for the original.
        out_path:     Destination ``.mp4`` path.
        video_path:   Optional source video for background frames.
        fps:          Output frame rate.
        max_seconds:  Cap video duration.
        canvas_size:  ``(h, w)`` for skeleton-only mode (no video).
        conf_thresh:  Hide joints below this confidence.

    Returns:
        The resolved *out_path*.
    """
    T = min(orig_kp.shape[0], recon_kp.shape[0], int(max_seconds * fps))
    orig_pos = kp_flat_to_positions(orig_kp[:T], global_state[:T])
    recon_pos = kp_flat_to_positions(recon_kp[:T], global_state[:T])
    conf = confidence[:T]
    recon_conf = recon_confidence[:T] if recon_confidence is not None else conf

    cap = None
    src_fps = fps
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps_raw = cap.get(cv2.CAP_PROP_FPS)
        if src_fps_raw > 0:
            src_fps = src_fps_raw
    else:
        frame_h, frame_w = canvas_size

    canvas_w = frame_w * 2
    canvas_h = frame_h + _LABEL_H

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(
        str(tmp_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (canvas_w, canvas_h),
    )

    # Time-align source video frames to pose output rate. When src_fps != fps
    # (e.g. 30 fps mp4 + 25 fps pose data) we need source frame round(t * src_fps / fps)
    # for output frame t. Advance via grab() (fast — no decode) and only read()
    # the target frame.
    fps_ratio = src_fps / fps
    cur_src = 0  # next frame index that cap.grab()/read() will return

    for t in range(T):
        if cap is not None:
            target_src = int(round(t * fps_ratio))
            while cur_src < target_src:
                if not cap.grab():
                    target_src = cur_src  # video shorter than expected — stop advancing
                    break
                cur_src += 1
            ret, frame = cap.read()
            if not ret:
                break
            cur_src += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        left = _draw_skeleton(rgb.copy(), orig_pos[t], _GREEN, conf[t], conf_thresh)
        right = _draw_skeleton(rgb.copy(), recon_pos[t], _RED, recon_conf[t], conf_thresh)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[_LABEL_H:, :frame_w] = left
        canvas[_LABEL_H:, frame_w:] = right

        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, "Original", (frame_w // 2 - 50, 28), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, "Reconstructed", (frame_w + frame_w // 2 - 80, 28), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(bgr)

    if cap is not None:
        cap.release()
    writer.release()

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-movflags", "+faststart", str(out_path)],
            check=True, capture_output=True,
        )
        tmp_path.unlink(missing_ok=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        tmp_path.rename(out_path)

    return out_path


# ---------------------------------------------------------------------------
# Test-clip preparation
# ---------------------------------------------------------------------------

def prepare_test_clips(
    data_dir: str | Path,
    dims: dict[str, tuple[float, float]],
    file_paths: list[Path],
    test_dir: str | Path | None = None,
    n_clips: int = 10,
    max_frames: int = 250,
    seed: int = 42,
    source_fps: int = 30,
    target_fps: int = 25,
) -> list[dict]:
    """Build a list of test clip dicts for comparison rendering.

    If *test_dir* is provided, clips are loaded from there (with optional
    matching ``.mp4`` videos).  Otherwise *n_clips* are randomly sampled
    from *file_paths* (typically the val split).

    Clips are resampled from *source_fps* to *target_fps* so the model sees
    the same time base it was trained on. *max_frames* is in *target_fps* units.

    Each returned dict has keys:
        ``keypoints``, ``global_state``, ``confidence`` (numpy, capped to *max_frames*),
        ``clip_id`` (str), ``video_path`` (Path or None).
    """
    import random as _random

    clips: list[dict] = []

    def _finalize(clip: dict, clip_id: str, video: Path | None) -> dict:
        clip = _resample_clip(clip, source_fps, target_fps)
        return {**{k: v[:max_frames] for k, v in clip.items()},
                "clip_id": clip_id,
                "video_path": video if (video is not None and video.exists()) else None}

    if test_dir:
        test_dir = Path(test_dir)
        # Recursive: pose files may live in subfolders (e.g. test_set/actor_*/).
        # ``n_clips <= 0`` means "use every clip in test_dir, one per top-level
        # subfolder (typically one segment per actor) to keep render time bounded".
        all_npz = sorted(test_dir.rglob("*_poses.npz"))
        if n_clips <= 0:
            # Keep the first sorted file under each top-level subdir; clips
            # directly under test_dir are kept as-is.
            seen_dirs: set[Path] = set()
            npz_files: list[Path] = []
            for npz in all_npz:
                rel_parent = npz.parent.relative_to(test_dir)
                top = rel_parent.parts[0] if rel_parent.parts else ""
                key = Path(top)  # "" for files directly under test_dir
                if str(top) and key in seen_dirs:
                    continue
                seen_dirs.add(key)
                npz_files.append(npz)
        else:
            npz_files = all_npz[:n_clips]
        for npz in npz_files:
            stem = npz.stem
            oh, ow = dims.get(stem, (None, None))
            clip = load_clip(npz, orig_h=oh, orig_w=ow)
            if clip is None:
                continue
            base = stem.removesuffix("_poses")
            # Match .mp4 next to the npz, not at the top of test_dir.
            video = npz.with_name(f"{base}.mp4")
            # Uniquify clip_id by including the relative subdir so e.g.
            # actor_1/segment_01 and actor_2/segment_01 don't clobber each other.
            rel_parent = npz.parent.relative_to(test_dir)
            clip_id = base if rel_parent == Path(".") else f"{'__'.join(rel_parent.parts)}__{base}"
            clips.append(_finalize(clip, clip_id, video))
    else:
        rng = _random.Random(seed)
        indices = list(range(len(file_paths)))
        rng.shuffle(indices)
        for idx in indices:
            if len(clips) >= n_clips:
                break
            path = file_paths[idx]
            stem = path.stem
            oh, ow = dims.get(stem, (None, None))
            clip = load_clip(path, orig_h=oh, orig_w=ow)
            if clip is None:
                continue
            clip_id = stem.removesuffix("_poses")
            clips.append(_finalize(clip, clip_id, None))

    return clips


@torch.no_grad()
def _reconstruct_clip(
    model: torch.nn.Module,
    keypoints: np.ndarray,
    device,
    *,
    chunk_size: int | None = None,
    stride: int | None = None,
) -> np.ndarray:
    """Encode + decode a full clip with optional overlapping-chunk inference.

    Why chunking at all? The encoder is fully convolutional and was trained on
    fixed-length segments (``segment_frames``). Passing a longer clip in one
    shot puts the model in a different input-length regime than training, and
    its per-frame output distribution drifts.

    Why *overlapping* chunks? With non-overlapping chunks (stride == chunk_size),
    every chunk's output is a function of *only* its own chunk's input — adjacent
    chunks share no input at the seam, so their outputs can disagree there.
    On top of that, the model's first/last few output frames in any chunk have
    less context than interior frames (no future / no past inside that chunk).
    Both problems are fixed by running overlapping chunks with a Hann window
    blend: each output frame is a weighted average of every chunk that contains
    it, with weights peaking at the chunk's center and tapering to (near) zero
    at its edges. With ``stride == chunk_size // 2`` Hann gives perfect COLA
    reconstruction in the interior, and every interior frame is dominated by
    the chunk where it sits closest to the center.

    Args:
        chunk_size: window length for inference (should match training-time
            ``segment_frames``). ``None`` or T <= chunk_size → single pass.
        stride: gap between successive window starts. ``None`` defaults to
            ``chunk_size // 2``. Set ``stride == chunk_size`` to disable overlap
            (legacy non-overlapping behaviour with end-of-clip anchored tail).
    """
    T, F = keypoints.shape

    if chunk_size is None or T <= chunk_size:
        kp = torch.from_numpy(keypoints).float().unsqueeze(0).to(device)
        return model(kp)["reconstruction"].squeeze(0).cpu().numpy()

    if stride is None:
        stride = max(1, chunk_size // 2)
    stride = min(stride, chunk_size)

    # Hann window for blending. Floor at a small epsilon so divisions by
    # weight_sum at the very first/last frames (covered by only one chunk
    # at near-zero Hann weight) are well-defined.
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(chunk_size) / (chunk_size - 1)))
    w = np.maximum(w, 1e-2).astype(np.float32)

    # Window starts over the raw clip. Always include a tail-anchored window
    # so the last frames get full coverage. NOTE: we deliberately do NOT
    # reflection-pad the input — that would create palindromic context at
    # the model's first/last chunks (OOD wrt training), which is exactly
    # the failure mode the external reflect-pad training protocol exhibited.
    # Without input-side padding, boundary chunks feed the model the same
    # regime it saw during training (real chunk + internal zero-padding
    # inside each conv), so the boundary predictions are in-distribution
    # — just produced from one-sided context.
    starts: list[int] = list(range(0, T - chunk_size + 1, stride))
    if not starts or starts[-1] + chunk_size < T:
        starts.append(T - chunk_size)
    starts = sorted(set(starts))

    out = np.zeros((T, F), dtype=np.float32)
    weight_sum = np.zeros(T, dtype=np.float32)
    for s in starts:
        e = s + chunk_size
        chunk = torch.from_numpy(keypoints[s:e]).float().unsqueeze(0).to(device)
        r = model(chunk)["reconstruction"].squeeze(0).cpu().numpy()
        out[s:e] += r * w[:, None]
        weight_sum[s:e] += w
    return out / weight_sum[:, None]


def render_test_comparisons(
    model: torch.nn.Module,
    test_clips: list[dict],
    out_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    fps: float = 25.0,
    max_seconds: float = 10.0,
    chunk_size: int | None = None,
    chunk_stride: int | None = None,
) -> list[Path]:
    """Run reconstruction on each test clip and render comparison videos.

    Args:
        chunk_size: if set, inference is sliced into windows of this length
            (typically training-time segment_frames). None = single full-clip pass.
        chunk_stride: gap between window starts when chunking. None defaults
            to chunk_size // 2 (Hann-blended overlap-add).

    Returns list of output video paths.
    """
    out_dir = Path(out_dir)
    model.eval()
    paths: list[Path] = []

    for clip in test_clips:
        recon = _reconstruct_clip(
            model, clip["keypoints"], device,
            chunk_size=chunk_size, stride=chunk_stride,
        )
        out_path = out_dir / f"{clip['clip_id']}_comparison.mp4"
        render_comparison_video(
            orig_kp=clip["keypoints"],
            recon_kp=recon,
            global_state=clip["global_state"],
            confidence=clip["confidence"],
            out_path=out_path,
            video_path=clip.get("video_path"),
            fps=fps,
            max_seconds=max_seconds,
        )
        paths.append(out_path)

    return paths
