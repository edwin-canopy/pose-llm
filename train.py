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

import yaml
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

from collators.monostream_pose_collator import PoseSpeechMonoCollator
from models.backbone_model import DEFAULT_BACKBONE_ARCH, EndToEndModel

# TODO: implement these
# from collators.asr_collator import ASRCollator
# from collators.conversational_pose_collator import ConversationalPoseCollator
# from dataset import load_pose_speech_dataset, load_asr_dataset, load_conversational_dataset
# from training_components.custom_trainer import MultiTaskTrainer
from collators import (
    PoseSpeechMonoCollator,
    ASRCollator
)

CONFIG = yaml.safe_load(open("config.yaml"))
BACKBONE_CONFIG = CONFIG["backbone"]
AUDIO_DEPTH_CONFIG = CONFIG["audio_depth_model"]
POSE_DEPTH_CONFIG = CONFIG["pose_depth_model"]
SPECIAL_TOKEN_CONFIG = CONFIG["special_tokens"]
WANDB_CONFIG = CONFIG.get("wandb", {})
MULTITASK_CONFIG = CONFIG.get("multitask", {})
TRAINING_ARGS_CONFIG = CONFIG.get("training_args", {})
EVAL_CONFIG = CONFIG.get("eval")
SCHEDULED_TASKS = list(dict.fromkeys(MULTITASK_CONFIG.get("schedule", [])))

os.environ["WANDB_PROJECT"] = WANDB_CONFIG.get("project", "pose-llm")


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
    model = EndToEndModel(backbone_arch_config, config=CONFIG)

# Resize backbone embeddings to cover the special tokens + audio/pose ranges
# (last reserved id from config.yaml is pose_tokens_end)
model.resize_token_embeddings(
    SPECIAL_TOKEN_CONFIG["pose_tokens_end"] + 1, mean_resizing=False
)

# disable KV cache during training
model.config.use_cache = False
model.audio_depth_model.config.use_cache = False
model.pose_depth_model.config.use_cache = False


# collators ------------------------------------------------------------------
collators = {
    "speech_pose": PoseSpeechMonoCollator(tokenizer, CONFIG),
    # "asr": ASRCollator(tokenizer, CONFIG),
    # "conversational": ConversationalPoseCollator(tokenizer, CONFIG),
}

# Hold out first eval_n rows of each dataset for eval; train on the rest.
eval_n = EVAL_CONFIG["num_examples"] if EVAL_CONFIG else 0
tasks = {
    name: (raw[name].select(range(eval_n, len(raw[name]))), collators[name], 1)
    for name in SCHEDULED_TASKS
    if name in raw
}

eval_tasks = {}
if EVAL_CONFIG is not None:
    eval_tasks = {
        name: (raw[name].select(range(eval_n)), collators[name], 1)
        for name in SCHEDULED_TASKS
        if name in raw
    }


# trainer --------------------------------------------------------------------
args = TrainingArguments(**TRAINING_ARGS_CONFIG, run_name=WANDB_CONFIG.get("run_name"))


trainer = MultiTaskTrainer(
    model=model,
    args=args,
    train_dataset=next(iter(tasks.values()))[0],
    tasks=tasks,
    schedule=MULTITASK_CONFIG["schedule"],
    length=MULTITASK_CONFIG["length"],
    eval_tasks=eval_tasks,
    eval_steps=EVAL_CONFIG["steps"] if EVAL_CONFIG else None,
)

resume = os.environ.get("RESUME_FROM_CHECKPOINT") or None
if resume in ("1", "true", "True"):
    resume = True
trainer.train(resume_from_checkpoint=resume)
