"""One-time run setup: CLI, wandb init, sweep overrides, distributed broadcast, run dir."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import wandb

from pose_tokenizer.config import PoseTokenizerConfig, TrainConfig

_SWEEP_TRAIN_FIELDS = {f.name for f in fields(TrainConfig)}
_SWEEP_MODEL_FIELDS = {f.name for f in fields(PoseTokenizerConfig)}


@dataclass
class RunContext:
    """Resolved post-bootstrap state passed into the training loop."""
    cfg: TrainConfig
    model_overrides: dict | None
    effective_name: str | None
    run_id: str | None
    run_dir: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Exact resume from a step dir, training_state.pt, or legacy checkpoint.pt",
    )
    p.add_argument(
        "--init-from-checkpoint",
        default=None,
        help=(
            "Initialize model weights from a step dir or checkpoint file for a new run. "
            "Unlike --checkpoint, this does not restore optimizer, step, "
            "wandb id, run dir, or config."
        ),
    )
    p.add_argument("--run-name", default=None)
    args, _ = p.parse_known_args()
    if args.checkpoint and args.init_from_checkpoint:
        p.error("--checkpoint and --init-from-checkpoint are mutually exclusive")
    return args


def resolve_config_path(args: argparse.Namespace) -> Path:
    """When resuming, prefer the full TrainConfig saved alongside the run
    (one level above the step_* dir) over whatever --config points at."""
    config_path = Path(args.config)
    if args.checkpoint:
        cp = Path(args.checkpoint).parent.parent / "config.yaml"
        if cp.exists():
            config_path = cp
    return config_path


def _apply_sweep_overrides(cfg: TrainConfig) -> dict | None:
    """If running inside a wandb sweep, override TrainConfig fields and return
    any model-config overrides (or None if not a sweep)."""
    sweep_cfg = dict(wandb.config)
    if not sweep_cfg:
        return None

    model_overrides: dict = {}
    for k, v in sweep_cfg.items():
        if k in _SWEEP_TRAIN_FIELDS:
            setattr(cfg, k, v)
        elif k in _SWEEP_MODEL_FIELDS:
            model_overrides[k] = v
    return model_overrides or None


def _init_wandb(args, cfg, accelerator) -> tuple[object | None, str | None]:
    """Init wandb on main process (with checkpoint resume-peek). Returns (wb, run_id)."""
    if not accelerator.is_main_process:
        return None, None

    resume_wandb_id = None
    if args.checkpoint:
        try:
            meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            resume_wandb_id = meta.get("wandb_run_id")
        except Exception as e:
            print(f"[warn] Could not peek wandb_run_id from checkpoint: {e}")

    run_name = args.run_name or cfg.wandb_run_name or ""
    wb_kwargs: dict = {"project": cfg.wandb_project}
    if resume_wandb_id:
        # resume="allow" + explicit id resumes the existing run if it exists,
        # or creates a new one otherwise. The original name is preserved.
        wb_kwargs["id"] = resume_wandb_id
        wb_kwargs["resume"] = "allow"
    else:
        wb_kwargs["name"] = run_name or None

    wb = wandb.init(**wb_kwargs)
    return wb, wb.id


def _broadcast_config(accelerator, cfg, model_overrides, effective_name):
    """Broadcast (cfg, model_overrides, effective_name) from rank 0 to all procs.

    Returns the (possibly broadcasted) values; on the main process these are
    just passed through unchanged.
    """
    if accelerator.num_processes <= 1:
        return cfg, model_overrides, effective_name

    import pickle
    cfg_values = [cfg, model_overrides, effective_name]
    if accelerator.is_main_process:
        buf = pickle.dumps(cfg_values)
        size = torch.tensor([len(buf)], device=accelerator.device)
    else:
        size = torch.tensor([0], device=accelerator.device)
    torch.distributed.broadcast(size, src=0)
    if accelerator.is_main_process:
        data = torch.frombuffer(buf, dtype=torch.uint8).to(accelerator.device)
    else:
        data = torch.empty(size.item(), dtype=torch.uint8, device=accelerator.device)
    torch.distributed.broadcast(data, src=0)
    if not accelerator.is_main_process:
        cfg, model_overrides, effective_name = pickle.loads(data.cpu().numpy().tobytes())
    return cfg, model_overrides, effective_name


def _setup_run_dir(args, cfg, effective_name, config_path: Path, accelerator) -> str:
    if args.checkpoint:
        run_dir = str(Path(args.checkpoint).parent.parent)
    else:
        run_dir = str(Path(cfg.output_dir) / (effective_name or "unnamed"))
    if accelerator.is_main_process:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        if not args.checkpoint:
            shutil.copy2(config_path, Path(run_dir) / "config.yaml")
    return run_dir


def bootstrap_run(
    args: argparse.Namespace,
    cfg: TrainConfig,
    config_path: Path,
    yaml_model_overrides: dict | None,
    accelerator,
) -> RunContext:
    """Glue: init wandb, apply sweep overrides, broadcast cfg, create run dir.

    On non-main processes wandb is skipped and (cfg, overrides, name) arrive
    via the broadcast. The returned RunContext is identical on every rank.
    """
    wb, run_id = _init_wandb(args, cfg, accelerator)

    model_overrides = yaml_model_overrides or None
    if accelerator.is_main_process:
        sweep_overrides = _apply_sweep_overrides(cfg)
        if sweep_overrides:
            model_overrides = {**(model_overrides or {}), **sweep_overrides}

    effective_name = wb.name if (wb is not None) else None
    cfg, model_overrides, effective_name = _broadcast_config(
        accelerator, cfg, model_overrides, effective_name,
    )

    run_dir = _setup_run_dir(args, cfg, effective_name, config_path, accelerator)

    return RunContext(
        cfg=cfg,
        model_overrides=model_overrides,
        effective_name=effective_name,
        run_id=run_id,
        run_dir=run_dir,
    )
