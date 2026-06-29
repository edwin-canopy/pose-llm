"""Per-step logging, periodic validation, and comparison-video rendering."""

from __future__ import annotations

import time
from pathlib import Path

import wandb

from pose_tokenizer.data.rendering import render_test_comparisons

from .validation import validate


# Module-level state for the throughput EWMA. Reset on process start; only the
# main process calls log_train_step so we don't need rank-isolation.
_STEP_TIMER: dict = {"t_last": None, "step_last": None, "it_s_ema": None}
_EMA_ALPHA = 0.2  # responsiveness vs smoothness; ≈10-print half-life


def _format_eta(seconds: float) -> str:
    """Format seconds as 'Xh Ym', 'Mm Ss', or 'Ss'."""
    if seconds < 0 or seconds != seconds:  # NaN-safe
        return "?"
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"


def log_train_step(
    accelerator,
    global_step: int,
    epoch: int,
    metrics: dict,
    lr: float,
    *,
    max_steps: int | None = None,
) -> None:
    """Log a single training step to wandb + print a one-liner with throughput.

    The printed line includes the step (and `step/max  pct` when ``max_steps``
    is provided), all loss components, the current LR, an EWMA-smoothed
    throughput in iterations/sec, and an ETA in `Xh Ym` form. Throughput is
    intentionally not logged to wandb because short dataloader/NFS stalls make
    it noisy beside real training metrics. Throughput and ETA are estimated
    from the wall-clock delta between successive log calls (i.e. over the most
    recent ``log_interval`` steps).

    Args:
        metrics: dict of float losses ``{rec_loss, vel_loss, commitment_loss,
            codebook_loss, loss}``. The caller owns this dict so the loss math
            stays close to the loop. ``epoch`` and ``lr`` are added here and
            prefixed with ``train/`` for wandb.
        max_steps: optional total step count; when >0 enables the pct and ETA
            fields. Pass 0 / None for indefinite training.
    """
    payload = {f"train/{k}": v for k, v in metrics.items()}
    payload["train/epoch"] = epoch
    payload["train/lr"] = lr

    now = time.perf_counter()
    t_last = _STEP_TIMER["t_last"]
    step_last = _STEP_TIMER["step_last"]
    it_s_ema = _STEP_TIMER["it_s_ema"]
    if t_last is not None and step_last is not None and global_step > step_last:
        dt = now - t_last
        if dt > 0:
            it_s = (global_step - step_last) / dt
            it_s_ema = it_s if it_s_ema is None else (
                _EMA_ALPHA * it_s + (1.0 - _EMA_ALPHA) * it_s_ema
            )
    _STEP_TIMER["t_last"] = now
    _STEP_TIMER["step_last"] = global_step
    _STEP_TIMER["it_s_ema"] = it_s_ema

    wandb.log(payload, step=global_step)

    if max_steps and max_steps > 0:
        pct = 100.0 * global_step / max_steps
        step_field = f"step {global_step}/{max_steps} ({pct:5.2f}%)"
    else:
        step_field = f"step {global_step}"
    if it_s_ema is not None:
        rate_field = f"{it_s_ema:5.2f} it/s"
        if max_steps and max_steps > global_step:
            eta_field = f"ETA {_format_eta((max_steps - global_step) / it_s_ema)}"
        else:
            eta_field = "ETA ?"
    else:
        rate_field = "  -- it/s"
        eta_field = "ETA --"

    conf_field = (
        f"conf={metrics['conf_loss']:.6f}  " if "conf_loss" in metrics else ""
    )
    accelerator.print(
        f"  {step_field}  "
        f"loss={metrics['loss']:.6f}  "
        f"rec={metrics['rec_loss']:.6f}  "
        f"vel={metrics['vel_loss']:.6f}  "
        f"{conf_field}"
        f"commit={metrics['commitment_loss']:.6f}  "
        f"cb={metrics['codebook_loss']:.6f}  "
        f"lr={lr:.2e}  |  {rate_field}  |  {eta_field}"
    )


def run_validation(
    accelerator, model, val_loader, cfg, feat_weights, global_step, *, label: str = "val",
) -> None:
    """Run a single validation pass, log to wandb, and print a one-line summary.

    Switches the model back to train mode after.
    """
    vm = validate(
        model, val_loader, cfg,
        silent=not accelerator.is_local_main_process,
        feat_weights=feat_weights,
        accelerator=accelerator,
    )
    if accelerator.is_main_process:
        wandb.log({f"val/{k}": v for k, v in vm.items()}, step=global_step)
        per_cb = sorted(
            (int(k.split("_")[-1]), v)
            for k, v in vm.items() if k.startswith("cb_usage_")
        )
        per_cb_str = (
            "  [" + "  ".join(f"cb{i}={u:.0f}%" for i, u in per_cb) + "]"
            if per_cb else ""
        )
        accelerator.print(
            f"  ── {label} @ step {global_step} ──  "
            f"loss={vm['loss']:.4f}  rec={vm['rec_loss']:.4f}  "
            f"face={vm['rec_face']:.4f}  "
            f"body={vm['rec_body']:.4f}  hand={vm['rec_hand']:.4f}  "
            f"vel_rec(face/body/hand)={vm['vel_recovery_face']*100:.0f}%/"
            f"{vm['vel_recovery_body']*100:.0f}%/"
            f"{vm['vel_recovery_hand']*100:.0f}%  "
            f"cb={vm['cb_usage']:.0f}%"
            f"{per_cb_str}"
        )
    model.train()


def render_comparisons(accelerator, model, test_clips, ckpt_dir, cfg, global_step: int) -> None:
    """Render comparison videos for test clips, save to ckpt_dir, log to wandb.

    Inference is a single full-clip forward pass — the TCN is translation-
    equivariant so interior frames are processed identically regardless of
    input length. The only positions affected by clip-length change are
    within ~receptive_field/2 of the absolute clip boundaries, which is the
    intrinsic limit of a bidirectional model trained on chunks (no inference
    protocol can recover information the model never saw at training).
    Chunked inference is still available via _reconstruct_clip(chunk_size=...)
    for memory-bounded streaming use.
    """
    if not test_clips or ckpt_dir is None:
        return
    unwrapped = accelerator.unwrap_model(model)
    video_dir = Path(ckpt_dir) / "comparisons"
    fps = cfg.target_fps or cfg.source_fps
    paths = render_test_comparisons(
        unwrapped, test_clips, video_dir,
        device=accelerator.device, fps=fps, max_seconds=cfg.test_max_seconds,
    )
    if paths:
        # Pass format="mp4" explicitly — without it wandb transcodes every clip
        # to gif (much larger + slower upload), and warns on each .Video()
        # call. The `fps` argument is intentionally omitted: wandb ignores it
        # for file paths anyway (the mp4's container fps governs playback).
        wandb.log(
            {"test/comparisons": [wandb.Video(str(p), format="mp4") for p in paths]},
            step=global_step,
        )
    accelerator.print(f"  Rendered {len(paths)} comparison videos → {video_dir}")
