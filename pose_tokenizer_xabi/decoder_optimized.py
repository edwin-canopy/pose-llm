"""Progressive BF16 decode-path optimizations for the pose tokenizer.

Each optimization composes left-to-right on a single ``OptimizedDecodePath``
instance (apply step 1, then 2, then 3, then 4). The original
``PoseTokenizerModel`` source code is unchanged — these are wrappers and
in-place mutations of the loaded model, not a model rewrite.

Scope is **decode only**: the encoder is left untouched because the
benchmarks compare against ``decode_baseline_bf16/`` which fixes the codes
(encoder output) by construction.

    from pose_tokenizer import PoseTokenizer
    from pose_tokenizer.decoder_optimized import OptimizedDecodePath

    tok = PoseTokenizer.from_pretrained("xabirizar9/pose-tokenizer-8cb-mask", device="cuda")
    opt = OptimizedDecodePath(tok)
    opt.apply_step_1_remove_weight_norm()
    opt.apply_step_2_native_bf16()
    opt.apply_step_3_compile()
    opt.apply_step_4_cuda_graph(sample_codes)

    recon = opt.decode(codes)   # always returns fp32 (B, T, F) tensor
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import remove_parametrizations


class OptimizedDecodePath:
    """Mutates a PoseTokenizer's decoder path in place for inference.

    The model lives in the wrapped tokenizer. ``decode()`` dispatches based on
    which optimizations have been applied; callers should treat instances as a
    drop-in replacement for ``tokenizer.model.detokenize``.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.model = tokenizer.model
        self.device = tokenizer.device
        self.cfg = tokenizer.config

        # _mode is what decode() does on each call.
        # "autocast_bf16" matches the saved baseline (matches what
        # decode_baseline_bf16 was generated with).
        self._mode = "autocast_bf16"
        # CUDA-graph state for step 4.
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_codes: list[torch.Tensor] | None = None
        self._static_out: torch.Tensor | None = None
        # Which steps have been applied (for logging / sanity).
        self.applied_steps: list[int] = []

    # ------------------------------------------------------------------
    # Public decode API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """codes -> recon. Always returns fp32 (B, T, F)."""
        if self._mode == "autocast_bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model.decode(codes)
        elif self._mode == "native_bf16" or self._mode == "compiled":
            # Inputs are int codes — no dtype concerns. The decoder and
            # quantizer.out_proj/codebook are already in BF16 after step 2.
            out = self.model.decode(codes)
        elif self._mode == "graph":
            assert self._graph is not None
            for static, live in zip(self._static_codes, codes):
                static.copy_(live)
            self._graph.replay()
            out = self._static_out.clone()
        else:
            raise RuntimeError(f"unknown mode {self._mode!r}")

        # Always return fp32 to the caller so downstream metrics math is
        # dtype-agnostic. The .float() is a no-op if out is already fp32.
        return out.float()

    # ------------------------------------------------------------------
    # Step 1 — bake out weight_norm
    # ------------------------------------------------------------------

    def apply_step_1_remove_weight_norm(self) -> int:
        """Bake parametrized weight_norm into the underlying ``.weight`` tensor.

        ``weight_norm`` keeps a ``(g, v)`` decomposition and recomputes
        ``w = g * v / ||v||`` on every forward — visible in the profile as
        ``weight_norm_fwd_first_dim_kernel`` (38 launches per decode). The
        precomputed weight is mathematically identical to what was being
        recomputed each call, so this is bit-exact.

        Only touches the decode path: decoder convs + quantizer ``out_proj``s
        (used by ``from_codes``). Encoder convs and quantizer ``in_proj``s are
        left alone since they don't run during decode.
        """
        if 1 in self.applied_steps:
            return 0

        n = 0
        for m in self.model.decoder.modules():
            if (isinstance(m, nn.Conv1d)
                and hasattr(m, "parametrizations")
                and "weight" in m.parametrizations):
                remove_parametrizations(m, "weight", leave_parametrized=True)
                n += 1

        for q in self.model.quantizer.quantizers:
            if (hasattr(q.out_proj, "parametrizations")
                and "weight" in q.out_proj.parametrizations):
                remove_parametrizations(q.out_proj, "weight", leave_parametrized=True)
                n += 1

        self.applied_steps.append(1)
        return n

    # ------------------------------------------------------------------
    # Step 2 — native BF16 weights (no autocast)
    # ------------------------------------------------------------------

    def apply_step_2_native_bf16(self) -> None:
        """Convert the decode path to native BF16 storage.

        Replaces ``torch.autocast(bf16)`` (which keeps fp32 master weights and
        casts on every op) with permanent BF16 weights. The 113 ``bf16_copy``
        kernels per decode that show up under autocast disappear, at the cost
        of a small precision drift in LayerNorms (which now compute in BF16
        internally instead of being auto-upcast to fp32).
        """
        if 2 in self.applied_steps:
            return
        if 1 not in self.applied_steps:
            raise RuntimeError("step 2 requires step 1 (weight_norm removal) first")

        # Decoder + its norms + activations.
        self.model.decoder = self.model.decoder.to(torch.bfloat16)

        # Quantizer pieces touched by from_codes: codebook embedding and
        # out_proj. We don't touch in_proj since the encoder uses that.
        for q in self.model.quantizer.quantizers:
            q.codebook = q.codebook.to(torch.bfloat16)
            q.out_proj = q.out_proj.to(torch.bfloat16)

        self._mode = "native_bf16"
        self.applied_steps.append(2)

    # ------------------------------------------------------------------
    # Step 3 — torch.compile the decoder
    # ------------------------------------------------------------------

    def apply_step_3_compile(self) -> None:
        """Compile the decoder with ``torch.compile(mode='default')``.

        Default mode fuses pointwise chains into Triton kernels and lets
        Inductor pick a single layout for the whole fused subgraph (which is
        what should eliminate the layout transposes around every conv,
        ~60% of the un-optimized decoder). It does **not** wrap the result
        in CUDA Graphs — that's step 4's job, so the per-step attribution
        stays clean. ``mode='reduce-overhead'`` would do both fusion and
        cudagraphs in a single pass, but then step 4 has nothing to add.
        """
        if 3 in self.applied_steps:
            return
        if 2 not in self.applied_steps:
            raise RuntimeError("step 3 expects step 2 (native bf16) first")

        self.model.decoder = torch.compile(
            self.model.decoder, mode="default", fullgraph=False, dynamic=False,
        )
        # Mode stays "native_bf16" — decode() goes through model.decode which
        # now calls the compiled decoder.
        self.applied_steps.append(3)

    # ------------------------------------------------------------------
    # Step 4 — CUDA Graph capture of the full decode call
    # ------------------------------------------------------------------

    def apply_step_4_cuda_graph(self, sample_codes: Sequence[torch.Tensor]) -> None:
        """Capture the entire decode call (from_codes + decoder) as a CUDA Graph.

        ``sample_codes`` fixes the shape — all subsequent decode() calls must
        use codes with the same (B, T_codes) shape and dtype. We pre-allocate
        static input/output buffers and replay the captured kernel sequence
        for each call.

        For test_set/ this works because every resampled clip is 500 frames →
        T_codes = 125 with B=1. Variable-shape consumers would need one graph
        per shape.
        """
        if 4 in self.applied_steps:
            return
        if 3 not in self.applied_steps:
            raise RuntimeError("step 4 expects step 3 (compile) first")

        # Static inputs cloned from the sample. .copy_ in decode() will
        # update them in place before each replay.
        self._static_codes = [c.clone() for c in sample_codes]

        # Warm up the compiled decoder on a side stream so the capture sees
        # already-compiled kernels (compile happens lazily on first call).
        # Also lets cuDNN finalize algo selection.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.model.decode(self._static_codes)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_out = self.model.decode(self._static_codes)

        self._mode = "graph"
        self.applied_steps.append(4)


def apply_optimizations(
    tokenizer, steps: Sequence[int], sample_codes=None
) -> OptimizedDecodePath:
    """Convenience: build OptimizedDecodePath, apply the listed steps in order."""
    opt = OptimizedDecodePath(tokenizer)
    for step in sorted(set(steps)):
        if step == 1:
            opt.apply_step_1_remove_weight_norm()
        elif step == 2:
            opt.apply_step_2_native_bf16()
        elif step == 3:
            opt.apply_step_3_compile()
        elif step == 4:
            if sample_codes is None:
                raise ValueError("step 4 requires sample_codes for graph capture")
            opt.apply_step_4_cuda_graph(sample_codes)
        else:
            raise ValueError(f"unknown step {step}")
    return opt
