from __future__ import annotations

import torch
from tqdm import tqdm


@torch.no_grad()
def validate(
    model,
    val_loader,
    cfg,
    *,
    silent: bool = False,
    feat_weights: torch.Tensor | None = None,
    accelerator=None,
) -> dict[str, float]:
    """Run one validation pass and return cross-rank-equivalent metrics.

    Single sync primitive: ``accelerator.gather_for_metrics`` is called once per
    batch on every tensor we need to aggregate globally. It (a) gathers across
    ranks and (b) removes the duplicate samples that Accelerate's default
    ``even_batches=True`` adds when ``len(val_set) % (world * batch_size) ≠ 0``.

    Every accumulator stays identical across ranks because each iteration adds
    the *global-batch* contribution computed from gathered tensors — no final
    ``reduce`` step is needed. With ``accelerator=None`` (or ``num_processes==1``)
    the gather degrades to a passthrough and the metrics match a pre-DDP run.

    Metrics returned:
      loss               combined val objective (w_rec*rec_loss + w_commit*commit
                         + w_cb*cb_loss). The two quantizer terms are scalars
                         already reduced over the batch dim in vq.py, so we
                         gather them per-iteration and take mean-of-means; this
                         introduces a small (<1%) DDP-vs-single discrepancy when
                         the last batch on each rank is padded by Accelerate's
                         even_batches. Acceptable for monitoring; bit-fixing
                         would require threading per-sample quantizer losses
                         out of vq.py.
      rec_loss           feat-weighted L1 over all entries (mirrors train obj).
                         True global mean: sum of weighted abs-diff over every
                         sampled element divided by total element count. This
                         is bit-identical between single-GPU and N-GPU DDP.
      rec_face           L1 over the 5 face keypoints with detector confidence
                         ≥ cfg.val_confidence_threshold (default 0.5)
      rec_body           same, the 8 upper-body keypoints
      rec_hand           same, the 42 hand keypoints
      vel_recovery_face  1 − vel_err / true_motion on the confident subset.
                         1.0 = perfect motion reconstruction, ≤0 = static collapse.
      vel_recovery_body  same, upper-body keypoints
      vel_recovery_hand  same, hand keypoints (the key static-collapse alarm)
      cb_usage           aggregate VQ usage % across codebooks (0 if VQ disabled)
      cb_usage_{i}       per-codebook % (only when n_codebooks > 1)

    If a val batch has no ``confidence`` tensor (or
    ``cfg.val_confidence_threshold ≤ 0``) the masked metrics fall back to
    unmasked averages over all entries, so the metric names stay stable across
    datasets.
    """
    model.eval()

    unwrapped = model.module if hasattr(model, "module") else model
    mc = unwrapped.config
    device = next(unwrapped.parameters()).device
    # Flat-feature layout (see pose_tokenizer/data/kinematic.py):
    #   [0 : face_end)        5 face kps         (nose, eyes, ears)
    #   [face_end : body_end) 8 upper-body kps   (shoulders, elbows, wrists, hips)
    #   [body_end : F)       42 hand kps         (21 left + 21 right)
    n_hand_dims = 42 * mc.keypoint_dim
    n_face_dims = 5 * mc.keypoint_dim
    face_end = n_face_dims
    body_end = mc.input_features - n_hand_dims
    val_conf_thresh = getattr(cfg, "val_confidence_threshold", 0.5)
    cb_size = mc.codebook_size

    # gather_for_metrics shim: in single-process (or no-accelerator) mode this
    # is a passthrough. In DDP it gathers across ranks AND strips even_batches
    # padding duplicates. Always returns a tuple matching the input length.
    is_ddp = accelerator is not None and getattr(accelerator, "num_processes", 1) > 1
    def gfm(*tensors):
        return accelerator.gather_for_metrics(tensors) if is_ddp else tensors

    z = lambda: torch.zeros((), device=device)
    # rec_loss is computed as true sum / count over every weighted-diff element
    # (gives bit-identical DDP / single-GPU). loss is rec_loss * w_rec + the
    # commit/cb scalar losses (each gathered + averaged across ranks per batch,
    # then mean-of-batch-means across iterations).
    rec_sum, rec_count = z(), z()
    sum_commit, sum_cb, n_batches = z(), z(), z()
    rec_face = [z(), z()]
    rec_body = [z(), z()]
    rec_hand = [z(), z()]
    vel_err_face = [z(), z()]
    vel_err_body = [z(), z()]
    vel_err_hand = [z(), z()]
    true_mot_face = [z(), z()]
    true_mot_body = [z(), z()]
    true_mot_hand = [z(), z()]
    seen_codes: list[torch.Tensor] | None = None

    for batch in tqdm(val_loader, desc="Validation", ncols=100, leave=False, disable=silent):
        feats = batch["keypoints"].clamp(-10.0, 10.0)
        out = model(feats)
        recon = out["reconstruction"]
        diff = (recon - feats).abs()

        vel_diff = (recon[:, 1:, :] - recon[:, :-1, :]
                    - (feats[:, 1:, :] - feats[:, :-1, :])).abs()
        true_mag = (feats[:, 1:, :] - feats[:, :-1, :]).abs()

        # Build per-feature mask: either a confidence threshold or all-ones.
        if val_conf_thresh > 0.0 and "confidence" in batch:
            conf = batch["confidence"]                  # (B, T, NUM_JOINTS)
            m_joint = (conf >= val_conf_thresh).to(feats.dtype)
            m_xy = m_joint.repeat_interleave(mc.keypoint_dim, dim=-1)
        else:
            m_xy = torch.ones_like(feats)
        m_face = m_xy[..., :face_end]
        m_body = m_xy[..., face_end:body_end]
        m_hand = m_xy[..., body_end:]
        vm_xy = m_xy[:, 1:, :] * m_xy[:, :-1, :]
        vm_face = vm_xy[..., :face_end]
        vm_body = vm_xy[..., face_end:body_end]
        vm_hand = vm_xy[..., body_end:]

        # One gather call: combines every per-sample tensor we need to sum
        # AND the per-rank scalar quantizer losses (wrapped as 1-D one-sample
        # tensors so they gather along the batch dim). After this each rank
        # has the global, padding-stripped view; accumulators stay in sync.
        (diff_g, m_face_g, m_body_g, m_hand_g,
         vm_face_g, vm_body_g, vm_hand_g, vel_diff_g, true_mag_g,
         commit_g, cb_g) = gfm(
            diff, m_face, m_body, m_hand,
            vm_face, vm_body, vm_hand, vel_diff, true_mag,
            out["commitment_loss"].unsqueeze(0), out["codebook_loss"].unsqueeze(0),
        )

        # rec_loss as true sum / count over every gathered weighted-diff element.
        # feat_weights normalises so sum(w)=F → (w * diff).mean() ≡ mean(w * diff)
        # ≡ sum(w * diff) / numel(diff). Accumulating sum and numel gives the
        # bit-identical global mean regardless of world_size.
        weighted = (feat_weights.to(feats.device) * diff_g
                    if feat_weights is not None else diff_g)
        rec_sum   += weighted.sum()
        rec_count += weighted.numel()

        # Quantizer losses: already scalar per rank-batch in vq.py
        # (.mean([1,2]).mean()). Gather across ranks and average for this
        # global iteration, then sum across iterations for the final mean.
        sum_commit += commit_g.mean()
        sum_cb     += cb_g.mean()
        n_batches  += 1

        # Ratio metrics: numerator and denominator both sum across the gathered
        # samples. Padding duplicates were already stripped by gather_for_metrics.
        rec_face[0] += (diff_g[..., :face_end] * m_face_g).sum()
        rec_face[1] += m_face_g.sum()
        rec_body[0] += (diff_g[..., face_end:body_end] * m_body_g).sum()
        rec_body[1] += m_body_g.sum()
        rec_hand[0] += (diff_g[..., body_end:] * m_hand_g).sum()
        rec_hand[1] += m_hand_g.sum()
        vel_err_face[0] += (vel_diff_g[..., :face_end] * vm_face_g).sum()
        vel_err_face[1] += vm_face_g.sum()
        vel_err_body[0] += (vel_diff_g[..., face_end:body_end] * vm_body_g).sum()
        vel_err_body[1] += vm_body_g.sum()
        vel_err_hand[0] += (vel_diff_g[..., body_end:] * vm_hand_g).sum()
        vel_err_hand[1] += vm_hand_g.sum()
        true_mot_face[0] += (true_mag_g[..., :face_end] * vm_face_g).sum()
        true_mot_face[1] += vm_face_g.sum()
        true_mot_body[0] += (true_mag_g[..., face_end:body_end] * vm_body_g).sum()
        true_mot_body[1] += vm_body_g.sum()
        true_mot_hand[0] += (true_mag_g[..., body_end:] * vm_hand_g).sum()
        true_mot_hand[1] += vm_hand_g.sum()

        if out["codes"] is not None:
            if seen_codes is None:
                seen_codes = [torch.zeros(cb_size, dtype=torch.long, device=device)
                              for _ in out["codes"]]
            for cb_idx, codes_i in enumerate(out["codes"]):
                (codes_g,) = gfm(codes_i.contiguous())
                seen_codes[cb_idx].index_fill_(0, codes_g.flatten().unique(), 1)

    def r(p) -> float:
        n, d = p[0].item(), p[1].item()
        return n / d if d > 0 else 0.0

    def vrec(err, mot) -> float:
        # In [0, 1]: 1 = perfect motion, ≤0 = static (vel_err ≈ true_motion).
        # Collapse threshold ≈ 0.2 — below that, model is predicting the mean pose.
        return 1.0 - r(err) / max(r(mot), 1e-9)

    n_b = max(int(n_batches.item()), 1)
    rec_count_v = rec_count.item()
    rec_loss = (rec_sum.item() / rec_count_v) if rec_count_v > 0 else 0.0
    commit_loss = sum_commit.item() / n_b
    cb_loss = sum_cb.item() / n_b
    metrics: dict[str, float] = {
        "loss": (cfg.reconstruction_weight * rec_loss
                 + cfg.commitment_weight * commit_loss
                 + cfg.codebook_weight * cb_loss),
        "rec_loss": rec_loss,
        "rec_face": r(rec_face),
        "rec_body": r(rec_body),
        "rec_hand": r(rec_hand),
        "vel_recovery_face": vrec(vel_err_face, true_mot_face),
        "vel_recovery_body": vrec(vel_err_body, true_mot_body),
        "vel_recovery_hand": vrec(vel_err_hand, true_mot_hand),
        "cb_usage": 0.0,
    }
    if seen_codes is not None:
        per_cb_usage = [(s > 0).sum().item() / cb_size * 100.0 for s in seen_codes]
        metrics["cb_usage"] = sum(per_cb_usage) / len(per_cb_usage)
        if len(per_cb_usage) > 1:
            for i, u in enumerate(per_cb_usage):
                metrics[f"cb_usage_{i}"] = u
    return metrics
