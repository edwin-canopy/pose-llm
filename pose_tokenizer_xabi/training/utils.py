from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch


def load_model_weights(path: str | Path, model, map_location="cpu") -> dict:
    """Load model weights from a checkpoint file or HF-format checkpoint dir."""
    path = Path(path)
    if path.is_dir():
        loaded = model.__class__.from_pretrained(str(path))
        model.load_state_dict(loaded.state_dict())
        return {"source": str(path)}

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        return ckpt

    # A new-format training_state.pt stores optimizer metadata only. Pair it
    # with the sibling model.safetensors/config.json files in the same step dir.
    if path.name == "training_state.pt":
        loaded = model.__class__.from_pretrained(str(path.parent))
        model.load_state_dict(loaded.state_dict())
        return ckpt

    model.load_state_dict(ckpt)
    return {"source": str(path)}


def save_run_metadata(ckpt_dir: str, model) -> None:
    """Save model config and architecture summary into a checkpoint directory."""
    rd = Path(ckpt_dir)
    with open(rd / "model_config.json", "w") as f:
        json.dump(asdict(model.config), f, indent=2)
    with open(rd / "model_summary.txt", "w") as f:
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}\n")
        f.write(f"Decoder parameters: {sum(p.numel() for p in model.decoder.parameters()):,}\n")
        if model.quantizer is not None:
            f.write(f"Quantizer parameters: {sum(p.numel() for p in model.quantizer.parameters()):,}\n")


def build_wandb_config(cfg, model) -> dict:
    """Merge training config + model architecture config into a flat dict for wandb."""
    model_cfg = asdict(model.config)
    return {
        **{f"train/{k}": v for k, v in vars(cfg).items()},
        **{f"model/{k}": v for k, v in model_cfg.items()},
        "total_params": sum(p.numel() for p in model.parameters()),
        "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
        "decoder_params": sum(p.numel() for p in model.decoder.parameters()),
        "downsampling_factor": model.downsampling_factor,
    }


def save_checkpoint(
    accelerator,
    model,
    optimizer,
    global_step,
    epoch,
    wandb_run_id,
    run_dir,
    is_final=False,
    norm_stats=None,
    save_optimizer_state=True,
):
    """Save HF-format model weights and optional exact-resume trainer state.

    Returns the checkpoint directory path (or None on non-main processes).
    """
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None
    unwrapped = accelerator.unwrap_model(model)
    tag = f"step_{global_step}"
    ckpt_dir = Path(run_dir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped.save_pretrained(str(ckpt_dir))

    trainer_state = {
        "global_step": global_step,
        "epoch": epoch,
        "wandb_run_id": wandb_run_id,
    }
    if save_optimizer_state:
        trainer_state["optimizer"] = optimizer.state_dict()
    torch.save(trainer_state, ckpt_dir / "training_state.pt")

    if norm_stats is not None and Path(norm_stats).is_file():
        import shutil
        shutil.copy2(str(norm_stats), str(ckpt_dir / "norm_stats.npz"))
    save_run_metadata(str(ckpt_dir), unwrapped)
    opt_msg = "with optimizer state" if save_optimizer_state else "model-only"
    accelerator.print(f"Saved checkpoint to {ckpt_dir} ({opt_msg})")
    return ckpt_dir


def load_checkpoint(path, model, optimizer, device):
    """Load checkpoint, returning (global_step, epoch, wandb_run_id)."""
    path = Path(path)
    if path.is_dir():
        state_path = path / "training_state.pt"
    elif path.name == "checkpoint.pt":
        # Legacy all-in-one checkpoints remain resumable.
        state_path = path
    else:
        state_path = path

    ckpt = load_model_weights(state_path, model, map_location=device)
    if not isinstance(ckpt, dict) or "global_step" not in ckpt:
        raise ValueError(
            f"{path} contains model weights only; use --init-from-checkpoint for a new run "
            "or resume from a checkpoint with training_state.pt."
        )
    if "optimizer" not in ckpt:
        raise ValueError(
            f"{state_path} does not contain optimizer state, so it cannot be used with "
            "--checkpoint for exact resume. Use --init-from-checkpoint instead."
        )
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["global_step"], ckpt["epoch"], ckpt.get("wandb_run_id")
