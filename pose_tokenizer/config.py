from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import List


@dataclass
class PoseTokenizerConfig:
    # Data shape
    num_keypoints: int = 55  # 5 face + 8 upper body + 21 left hand + 21 right hand
    keypoint_dim: int = 2
    input_features: int = 110  # num_keypoints * keypoint_dim (flattened)

    # Architecture (TCN encoder/decoder).
    # With DAC-ramp logic and downsampling_rates=[1, 2, 2], channels progress:
    #   input(110) -> init(384) -> block0(384) -> block1(768) -> block2/final(1536)
    first_channel_size: int = 384
    # Deprecated: the quantized latent width is now derived from the final
    # encoder channel count, matching DAC/SNAC more closely.
    latent_dim: int | None = None
    num_blocks: int = 3
    downsampling_rates: List[int] = field(default_factory=lambda: [1, 2, 2])
    kernel_size: List[int] = field(default_factory=lambda: [3, 3, 3])
    num_residual_blocks: int = 2
    residual_units_per_block: int = 2
    d_model: int = 1536
    num_groups: bool = True

    # Quantization (RVQ)
    codebook_size: int = 4096
    codebook_dim: int = 8
    n_codebooks: int = 4
    vq_strides: List[int] = field(
        default_factory=lambda: [1] * 16
    )
    apply_rotation_trick: bool = False
    enable_quantization: bool = True
    # DAC-style quantizer dropout: fraction of each batch for which we use a
    # random subset of quantizers (1..N) instead of all N. 0.0 = always all N
    # (legacy). 0.5 = DAC paper default; teaches the decoder to reconstruct at
    # variable bitrate so late codebooks can be dropped at inference time.
    quantizer_dropout: float = 0.0
    # DAC/ViT-VQGAN cosine-similarity nearest-neighbour lookup. When True,
    # L2-normalise both encoder output and codebook rows before computing the
    # squared-Euclidean distance in `decode_latents`, which is equivalent to
    # argmax cosine-similarity. The stored codebook embeddings remain
    # un-normalised, and downstream MSE commit/code losses are still computed
    # in the un-normalised space — exactly matching descript-audio-codec's
    # implementation. Helps codebook utilisation by making lookup invariant
    # to encoder-output magnitude drift.
    normalize_codes: bool = False
    # Dead-code revival (SoundStream/EnCodec/lucidrains; NOT part of DAC, which
    # relies only on factorized + L2-normalized codes — already enabled above).
    # Tracks an EMA of per-code selection counts; any code whose EMA usage falls
    # below ``dead_code_threshold`` is reset to a random encoder vector from the
    # current batch, preventing residual-VQ collapse when codebooks outnumber the
    # signal's information content. 0.0 disables (prior behaviour); 2.0 matches
    # EnCodec/lucidrains defaults.
    dead_code_threshold: float = 0.0
    dead_code_ema_decay: float = 0.99

    # Optional toggles
    use_noise_blocks: bool = False
    # Encoder dilation placement -> controls the temporal receptive field:
    #   "shallow" (default): dilate early blocks (i<2) — legacy behaviour.
    #   "deep": dilate the deepest blocks (mirrors decoder); largest RF.
    #   "none": no dilation -> narrow RF (<< segment length).
    encoder_dilation: str = "shallow"
    # Back-compat: True maps to encoder_dilation="deep".
    encoder_deep_dilation: bool = False
    norm: str = "LN"
    # When True, kernel>1 temporal convolutions use left-only padding.
    # Default False preserves the non-causal architecture and checkpoint keys.
    causal: bool = False
    conv_weight_norm: bool = False

    # ------------------------------------------------------------------
    # Backend selector + SEANet/transformer backend (model_type ==
    # "seanet_transformer"). These are IGNORED by the default conv backend.
    # The RVQ fields above (n_codebooks/codebook_size/codebook_dim/
    # quantizer_dropout/normalize_codes/dead_code_*) are reused unchanged so
    # the codebook + bitrate are identical to the conv tokenizer for a clean
    # A/B; only the encoder/decoder architecture differs.
    # ------------------------------------------------------------------
    model_type: str = "conv"            # "conv" | "seanet_transformer"
    # SEANet conv front/back end (kept modest; the transformer is under test).
    st_seanet_dim: int = 512            # SEANet output / decoder-input channels
    st_n_filters: int = 64              # SEANet base width
    st_n_residual_layers: int = 1
    st_ratios: List[int] = field(default_factory=lambda: [1])  # SEANet internal stride (1 = none)
    st_seanet_kernel: int = 7
    st_residual_kernel: int = 3
    st_compress: int = 2
    # Causal transformer bottleneck (param-heavy: this is the capability tested).
    st_d_model: int = 1024
    st_num_heads: int = 16
    st_num_layers: int = 8
    st_dim_feedforward: int = 4096
    st_layer_scale: float = 0.01
    st_rope_max_period: float = 10000.0
    # Net temporal downsample via a separate causal resample AFTER the encoder
    # transformer (transformer runs at the full input rate). Token rate =
    # target_fps / st_downsample; this is the model's downsampling_factor.
    st_downsample: int = 2

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PoseTokenizerConfig":
        with open(Path(path) / "config.json") as f:
            return cls(**json.load(f))


@dataclass
class TrainConfig:
    # Data
    data_dir: str = ""
    manifest: str = ""
    dims_csv: str = ""
    test_manifest: str = ""
    test_dir: str = ""               # dir with .mp4 + _poses.npz pairs for test rendering
    n_test_clips: int = 10           # number of clips for comparison video rendering
    test_max_seconds: float = 10.0   # cap comparison videos at this duration
    num_files: int = 0               # 0 = use all
    val_count: int = 10_000
    source_fps: int = 30             # fps of the raw .npz pose data
    target_fps: int = 25             # resample to this fps before training
    segment_frames: int = 0
    # Causal "warmup" context: number of real frames prepended *before* the
    # supervised ``segment_frames`` window. The model runs on ``prefix_frames +
    # segment_frames`` frames but the loss is computed only on the trailing
    # ``segment_frames`` (the prefix just primes the causal receptive field, so
    # every supervised frame sees a full history instead of cold-starting).
    # Fixed-length / no-padding: every sample reads EXACTLY prefix+segment real
    # frames, so batches are uniform and never zero-padded (padded frames would
    # NaN the quantizer's L2 normalisation in the backward pass). Clips shorter
    # than prefix+segment are dropped, not padded. ``prefix_frames + segment_frames``
    # must be a multiple of the model downsampling factor. 0 = disabled (legacy).
    prefix_frames: int = 0
    max_frames: int = 0
    random_crop_frames: bool = False
    # When True, each training sample re-samples a uniform random start offset in
    # [0, T - segment_frames] instead of using the static stride-segment_frames
    # window index. Required for clean chunked inference: without it the model
    # only ever sees frames at within-window positions p mod segment_frames.
    random_segment_offset: bool = True
    # Root convention for the kinematic-tree root joints (nose + shoulders).
    # False: per-frame mid-shoulder root (translation-invariant, legacy).
    # True:  fixed frame-0 mid-shoulder root — root joints carry body
    #        translation/drift. Applied process-wide via
    #        pose_tokenizer.data.dataset.set_first_frame_root() so training,
    #        validation and comparison rendering all share one convention.
    first_frame_root: bool = False

    # Inference-time chunking for long-clip rendering. `inference_chunk_stride`
    # controls overlap between successive chunks: stride < segment_frames yields
    # Hann-blended overlap-add reconstruction (recommended when chunking is
    # required). 0 = default to segment_frames // 2 (50% overlap). Set equal to
    # segment_frames for legacy non-overlapping behaviour.
    inference_chunk_stride: int = 0

    # Training
    epochs: int = 100
    max_steps: int = 0               # 0 = train full epochs; >0 = stop after N gradient steps
    batch_size: int = 64
    gradient_accumulation_steps: int = 1  # optimizer step every N micro-batches
    lr: float = 3e-4
    lr_schedule: str = "cosine"
    lr_min: float = 0.0
    # Linear warmup steps before cosine decay kicks in. 0 = no warmup. Only
    # applied when lr_schedule="cosine"; ignored for "constant".
    warmup_steps: int = 0
    # AdamW weight decay applied to "decay" param group (Conv1d weights, etc.).
    # The "no-decay" group (1D params: biases, LayerNorm γβ; plus codebook
    # embeddings) always uses 0.0 — pulling codebook entries toward origin
    # would collapse the discrete latent space.
    weight_decay: float = 0.01
    reconstruction_weight: float = 1.0
    commitment_weight: float = 0.25
    codebook_weight: float = 1.0
    velocity_weight: float = 0.0
    # Causal latency budget: compare recon[t] to input[t-causal_loss_shift]
    # instead of input[t], aligning the loss with the causal model's intrinsic
    # lag so it can commit to sharper outputs at a fixed, known delay. 0 = off.
    # The model still consumes the full sequence; only the loss target is shifted.
    causal_loss_shift: int = 0
    hand_loss_weight: float = 1.0
    # Chain-aware reconstruction term (only consumed by train_chain_loss.py;
    # standard train.py ignores it so 0.0 = behaviourally inert).
    # relative_l1_weight: L1 normalised by per-joint mean |GT_off| — equalises
    #     gradient pressure across joints regardless of their offset magnitude
    #     (upper-body offsets ≈ 0.25 vs fingertip ≈ 0.04, so plain L1 barely
    #     punishes finger errors as a fraction of the joint's "budget").
    relative_l1_weight: float = 0.0
    confidence_weight_mode: str = "none" # "none" | "soft" | "hard"
    confidence_mask_threshold: float = 0.0
    confidence_loss_weight: float = 1.0
    confidence_loss_type: str = "smooth_l1" # "bce" | "smooth_l1" | "l1"
    # Optional denoising input corruption for train_confidence.py. When enabled,
    # the model sees corrupted xy/confidence inputs but is still supervised
    # against the original clean high-confidence targets.
    pose_corruption_prob: float = 0.0
    pose_corruption_modes: List[str] = field(default_factory=list)
    pose_corruption_min_conf: float = 0.7
    pose_corruption_span_min: int = 1
    pose_corruption_span_max: int = 3
    pose_corruption_joints_min: int = 1
    pose_corruption_joints_max: int = 4
    pose_corruption_conf_min: float = 0.0
    pose_corruption_conf_max: float = 0.15
    pose_corruption_xy_noise_std: float = 0.0
    pose_corruption_xy_outlier_prob: float = 0.0
    pose_corruption_xy_outlier_std: float = 0.0
    freeze_encoder: bool = False
    freeze_quantizer: bool = False
    freeze_decoder: bool = False
    norm_stats: str = ""
    num_workers: int = 4
    log_interval: int = 10               # wandb log every N steps
    val_batch_size: int = 0            # 0 = same as batch_size
    val_confidence_threshold: float = 0.5 # threshold for the masked val metrics
    validation_interval: int = 1000    # validate every N steps
    save_interval: int = 10000         # checkpoint every N steps
    # Save AdamW optimizer state for exact training resume. This roughly doubles
    # checkpoint storage for large models, so short fine-tunes can disable it
    # and still keep model.safetensors for inference / weight initialization.
    save_optimizer_state: bool = True
    # Render comparison videos every N steps. Defaults to 0 = render at every
    # save_interval (legacy behaviour). Set higher than save_interval to keep
    # cheap checkpointing while skipping the expensive video rendering most of
    # the time (38 test clips × ffmpeg encode + wandb upload can stall training
    # for several minutes).
    render_interval: int = 0
    test_frame_idx: int = 0
    output_dir: str = "checkpoints"

    # Wandb
    wandb_project: str = "pose-tokenizer"
    wandb_run_name: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> tuple["TrainConfig", dict]:
        """Return (TrainConfig, model_overrides_dict).

        Any keys in the YAML that match PoseTokenizerConfig fields (but not
        TrainConfig fields) are collected into model_overrides so the caller
        can pass them when constructing the model.
        """
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        train_keys = cls.__dataclass_fields__
        model_keys = PoseTokenizerConfig.__dataclass_fields__
        train_kw = {k: v for k, v in raw.items() if k in train_keys}
        model_kw = {k: v for k, v in raw.items() if k in model_keys and k not in train_keys}
        return cls(**train_kw), model_kw
