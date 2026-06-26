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
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments

from collators import PoseSpeechMonoCollator
from models.backbone_model import DEFAULT_BACKBONE_ARCH, EndToEndModel


DATASET_DIR = "/mnt/somfs/pose_cond/merged_pose_audio_dataset/hf_pose_dataset_filtered"


CONFIG = yaml.safe_load(open("config.yaml"))
BACKBONE_CONFIG = CONFIG["backbone"]
POSE_DEPTH_CONFIG = CONFIG["pose_depth_model"]
SPECIAL_TOKEN_CONFIG = CONFIG["special_tokens"]
TRAINING_CONFIG = CONFIG["training"]


# tokenizer ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE_ARCH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# model ----------------------------------------------------------------------
# EndToEndModel: Qwen3 backbone + audio_depth_model + pose_depth_model
if BACKBONE_CONFIG["weights_path"]:
    model = EndToEndModel.from_pretrained(BACKBONE_CONFIG["weights_path"], config=CONFIG)
else:
    backbone_arch_config = AutoConfig.from_pretrained(DEFAULT_BACKBONE_ARCH)
    model = EndToEndModel(CONFIG, backbone_arch_config)

model.resize_token_embeddings(
    SPECIAL_TOKEN_CONFIG["pose_tokens_end"] + 1, mean_resizing=False
)

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


raw = load_from_disk(DATASET_DIR)
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
)

trainer = Trainer(
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
