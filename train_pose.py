"""
Testing script that trains only using interleaved sequence generation
Use RESUME_FROM_CHECKPOINT=1 to start from last available checkpoint
"""

import os

_TRITON_LOCAL_RANK = os.environ.get("LOCAL_RANK", "0")
_TRITON_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
_TRITON_USER = os.environ.get("USER", "user")
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/triton_cache_{_TRITON_USER}_{_TRITON_JOB_ID}_{_TRITON_LOCAL_RANK}",
)
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from accelerate import PartialState
from datasets import load_from_disk
from liger_kernel.transformers import apply_liger_kernel_to_llama
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

from collators import PoseSpeechMonoCollator
from models.backbone_model import DEFAULT_BACKBONE_ARCH, EndToEndModel
from train_utils.length_matching_sampler import LengthBudgetBatchSampler


DATASET_DIR = "/mnt/somfs/pose_cond/merged_pose_audio_dataset/hf_pose_dataset_filtered"


CONFIG = yaml.safe_load(open("config.yaml"))
BACKBONE_CONFIG = CONFIG["backbone"]
POSE_DEPTH_CONFIG = CONFIG["pose_depth_model"]
SPECIAL_TOKEN_CONFIG = CONFIG["special_tokens"]
TRAINING_CONFIG = CONFIG["training"]
WANDB_CONFIG = CONFIG["wandb"]


# wandb -----------------------------------------------------------------------
# Project + entity must be in the process env before HF Trainer's WandbCallback
# initialises (it reads these on first log()).
os.environ["WANDB_PROJECT"] = WANDB_CONFIG["project"]
if WANDB_CONFIG["entity"]:
    os.environ["WANDB_ENTITY"] = WANDB_CONFIG["entity"]


# tokenizer ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE_ARCH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# model ----------------------------------------------------------------------
# EndToEndModel: Qwen3 backbone + audio_depth_model + pose_depth_model
if BACKBONE_CONFIG["weights_path"]:
    model = EndToEndModel.from_pretrained(BACKBONE_CONFIG["weights_path"], config=CONFIG)
else:
    backbone_arch_config = AutoConfig.from_pretrained(
        DEFAULT_BACKBONE_ARCH,
        attn_implementation=TRAINING_CONFIG["attn_implementation"],
    )
    model = EndToEndModel(CONFIG, backbone_arch_config)

model.resize_token_embeddings(
    SPECIAL_TOKEN_CONFIG["pose_tokens_end"] + 1, mean_resizing=False
)


# Untie input/output embeddings on every sub-model. FSDP cannot shard the same
# tensor twice — Qwen3 and SmolLM2 tie embed_tokens.weight to lm_head.weight by
# default, which trips "Parameter is shared with a parameter already managed by
# another FSDP group". Untying is HF's recommended fix.
def _untie_word_embeddings(m):
    in_emb = m.get_input_embeddings()
    out_emb = m.get_output_embeddings()
    if out_emb is not None and out_emb.weight is in_emb.weight:
        out_emb.weight = torch.nn.Parameter(in_emb.weight.detach().clone())
    m.config.tie_word_embeddings = False

_untie_word_embeddings(model)
_untie_word_embeddings(model.audio_depth_model)
_untie_word_embeddings(model.pose_depth_model)


# Liger Kernel patches. HF Trainer's use_liger_kernel only handles the top-level
# model (Qwen3 backbone); manually patch the two Llama depth submodules the same
# way qwen-train does for its light_llm.
if TRAINING_CONFIG["liger_kernels"]:
    liger_kernel_config = TRAINING_CONFIG["liger_kernel_config"]
    apply_liger_kernel_to_llama(model=model.audio_depth_model, **liger_kernel_config)
    apply_liger_kernel_to_llama(model=model.pose_depth_model, **liger_kernel_config)


model.config.use_cache = False
model.audio_depth_model.config.use_cache = False
model.pose_depth_model.config.use_cache = False


# data -----------------------------------------------------------------------
POSE_CODEBOOKS = POSE_DEPTH_CONFIG["residual_depth"]


class PoseSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        pose_flat = torch.tensor(row["pose_tokens"], dtype=torch.long)
        pose = pose_flat.view(-1, POSE_CODEBOOKS).T.contiguous()
        audio = torch.tensor(row["audio_tokens"], dtype=torch.long)
        return {"text": row["text"], "audio_tokens": audio, "pose_tokens": pose}


def _add_length(row):
    # one entry per audio frame; pose is at the same rate
    return {"length": len(row["audio_tokens"][0])}


raw = load_from_disk(DATASET_DIR)
if "length" not in raw.column_names:
    with PartialState().main_process_first():
        raw = raw.map(_add_length, num_proc=16, desc="adding length column")
eval_rows = TRAINING_CONFIG["eval_rows"]
train_dataset = PoseSpeechDataset(raw.select(range(eval_rows, len(raw))))
eval_dataset = PoseSpeechDataset(raw.select(range(eval_rows))) if eval_rows > 0 else None

collator = PoseSpeechMonoCollator(tokenizer, CONFIG)


# trainer --------------------------------------------------------------------
gpu_minibatch_size = TRAINING_CONFIG["gpu_minibatch_size"]
args = TrainingArguments(
    output_dir="outputs",
    remove_unused_columns=False,
    per_device_train_batch_size=gpu_minibatch_size,
    per_device_eval_batch_size=gpu_minibatch_size,
    bf16=TRAINING_CONFIG["bf16"],
    gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
    dataloader_num_workers=TRAINING_CONFIG["dataloader_num_workers"],
    fsdp=TRAINING_CONFIG["fsdp"],
    fsdp_config=TRAINING_CONFIG["fsdp_config"],
    use_liger_kernel=TRAINING_CONFIG["liger_kernels"],
    liger_kernel_config=TRAINING_CONFIG["liger_kernel_config"],
    save_total_limit=TRAINING_CONFIG["keep_checkpoints"],
    logging_steps=TRAINING_CONFIG["logging_steps"],
    report_to="wandb",
    run_name=os.environ.get("RUN_NAME") or WANDB_CONFIG["run_name"],
)

class LengthPackedTrainer(Trainer):
    # Override get_train_dataloader to use LengthBudgetBatchSampler. Bypasses
    # accelerator.prepare so accelerate doesn't wrap our already-rank-aware
    # batch_sampler in BatchSamplerShard (which would re-shard and corrupt the
    # per-rank streams). Device placement still happens in _prepare_inputs.
    def get_train_dataloader(self):
        sampler = LengthBudgetBatchSampler(
            self.train_dataset.ds,
            target_frames=TRAINING_CONFIG["target_frames"],
            world_size=self.args.world_size,
            rank=self.args.process_index,
            shuffle=True,
            seed=self.args.seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )


trainer = LengthPackedTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    processing_class=tokenizer,
)

resume = os.environ.get("RESUME_FROM_CHECKPOINT") or None
if resume in ("1", "true", "True"):
    resume = True
trainer.train(resume_from_checkpoint=resume)
