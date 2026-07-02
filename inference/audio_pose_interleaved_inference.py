"""
Audio + pose interleaved inference (mono-stream, like training).

Loads the held-out eval slice and the trained EndToEndModel from a hardcoded
checkpoint, then generates audio and pose codes frame-by-frame for the first
NUM_GENERATE_SAMPLES samples.

Per frame, after appending the (predetermined) text token text_k, we:
  1. Forward backbone with [..., text_k, dummy=0, dummy=0]
     - sample audio0_k from the text-position logit (slice over audio vocab).
     - cache the text-position hidden state for the depth models.
  2. Run audio depth model autoregressively conditioned on text_hidden + audio0_k
     to get audio codes 1..7.
  3. Forward backbone again with [..., text_k, audio0_k, dummy=0] and the now-
     known audio tail in the audio-depth slot.
     - sample pose0_k from the audio-position logit (slice over pose vocab).
  4. Run pose depth model autoregressively conditioned on text_hidden + pose0_k
     to get pose codes 1..7.

This matches the training distribution: the backbone produces audio0 and pose0
in the same conditional order it was trained on (audio0 | prefix+text; pose0 |
prefix+text+audio0+audio_tail). Cost is ~2 backbone forwards + 2 depth passes
per frame, no KV-cache.
"""

import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import yaml
from datasets import load_from_disk
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

from collators import PoseSpeechMonoCollator
from models.backbone_model import DEFAULT_BACKBONE_ARCH, EndToEndModel
from paths import MERGED_DATASET_DIR


DISABLE_AUDIO_DEPTH_MODEL = False
DISABLE_POSE_DEPTH_MODEL = False
TEACHER_FORCE_TEXT = True

DATASET_DIR = f"{MERGED_DATASET_DIR}/hf_pose_dataset_filtered"
CHECKPOINTS_DIR = "/home/edwin/pose-llm/checkpoints"
CHECKPOINT_NAME = "latest" # e.g. "checkpoint-33000" or "latest"
INFERENCE_OUTPUTS_DIR = "/home/edwin/pose-llm/inference_outputs"

NUM_GENERATE_SAMPLES = 4
MAX_GENERATE_FRAMES = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

DO_SAMPLE = True
TEMPERATURE = 2
TOP_P = None
TOP_K = 20


CONFIG = yaml.safe_load(open("/home/edwin/pose-llm/config.yaml"))
TRAINING_CONFIG = CONFIG["training"]
SPECIAL_TOKEN_CONFIG = CONFIG["special_tokens"]
AUDIO_DEPTH = CONFIG["audio_depth_model"]["residual_depth"]
POSE_DEPTH = CONFIG["pose_depth_model"]["residual_depth"]
AUDIO_CODEBOOK_SIZE = CONFIG["audio_depth_model"]["codebook_size"]
POSE_CODEBOOK_SIZE = CONFIG["pose_depth_model"]["codebook_size"]
AUDIO_TOKENS_START = SPECIAL_TOKEN_CONFIG["audio_tokens_start"]
POSE_TOKENS_START = SPECIAL_TOKEN_CONFIG["pose_tokens_start"]
WORD_PAD_TOKEN = SPECIAL_TOKEN_CONFIG["word_pad"]
NEW_WORD_TOKEN = SPECIAL_TOKEN_CONFIG["new_word"]
USE_REFERENCE_POSE = CONFIG["backbone"]["use_reference_pose"]


# clear previous inference outputs ------------------------------------------
if os.path.isdir(INFERENCE_OUTPUTS_DIR):
    shutil.rmtree(INFERENCE_OUTPUTS_DIR)
os.makedirs(INFERENCE_OUTPUTS_DIR, exist_ok=True)
print(f"cleared {INFERENCE_OUTPUTS_DIR}")


# data -----------------------------------------------------------------------
raw = load_from_disk(DATASET_DIR)
eval_rows = TRAINING_CONFIG["eval_rows"]
eval_dataset = raw.select(range(eval_rows))
print(f"loaded {len(eval_dataset):,} held-out eval rows from {DATASET_DIR}")


# tokenizer ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE_ARCH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# model ----------------------------------------------------------------------
backbone_arch_config = AutoConfig.from_pretrained(
    DEFAULT_BACKBONE_ARCH,
    attn_implementation=TRAINING_CONFIG["attn_implementation"],
)
model = EndToEndModel(CONFIG, backbone_arch_config)

model.resize_token_embeddings(
    SPECIAL_TOKEN_CONFIG["pose_tokens_end"] + 1, mean_resizing=False
)

# Untie input/output embeddings on every sub-model. The trained checkpoint was
# saved with untied weights; leaving them tied here would have load_state_dict
# write both saved tensors into the same memory and silently overwrite one.
def _untie_word_embeddings(m):
    in_emb = m.get_input_embeddings()
    out_emb = m.get_output_embeddings()
    if out_emb is not None and out_emb.weight is in_emb.weight:
        out_emb.weight = torch.nn.Parameter(in_emb.weight.detach().clone())
    m.config.tie_word_embeddings = False

_untie_word_embeddings(model)
_untie_word_embeddings(model.audio_depth_model)
_untie_word_embeddings(model.pose_depth_model)

if CHECKPOINT_NAME == "latest":
    candidates = glob.glob(os.path.join(CHECKPOINTS_DIR, "checkpoint-*"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint-* dirs in {CHECKPOINTS_DIR}")
    CHECKPOINT_PATH = max(
        candidates, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1))
    )
    print(f"resolved latest checkpoint: {CHECKPOINT_PATH}")
else:
    CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, CHECKPOINT_NAME)

state_dict = load_file(os.path.join(CHECKPOINT_PATH, "model.safetensors"))
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"warning: {len(missing)} missing keys, e.g. {missing[:3]}")
if unexpected:
    print(f"warning: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

model.to(device=DEVICE, dtype=DTYPE)
model.eval()
model.config.use_cache = False
model.audio_depth_model.config.use_cache = False
model.pose_depth_model.config.use_cache = False

print(f"loaded EndToEndModel from {CHECKPOINT_PATH} on {DEVICE} / {DTYPE}")


# collator (used only to build the training-format text track per sample) ----
collator = PoseSpeechMonoCollator(tokenizer, CONFIG)


# helpers --------------------------------------------------------------------
def _sample(logits, do_sample, temperature, top_p, top_k):
    """Sample one token id from a 1-D logits tensor over a single codebook."""
    if not do_sample or temperature == 0.0:
        return int(logits.argmax().item())
    if top_p is not None and top_k is not None:
        raise ValueError("top_p and top_k are mutually exclusive; set exactly one")
    logits = logits.float() / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    if top_p is not None:
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[0] = True
        sorted_probs = sorted_probs * keep
    elif top_k is not None:
        sorted_probs[top_k:] = 0
    sorted_probs = sorted_probs / sorted_probs.sum()
    pick = torch.multinomial(sorted_probs, 1)
    return int(sorted_idx[pick].item())


def _text_track_for_sample(sample):
    """Run the training collator's assemble() to get the per-frame text track.

    Returns a Python list of token ids of length num_frames, identical to what
    column 0 of backbone_ids would be at training time.
    """
    pose_flat = torch.tensor(sample["pose_tokens"], dtype=torch.long)
    pose = pose_flat.view(-1, POSE_DEPTH).T.contiguous()
    audio = torch.tensor(sample["audio_tokens"], dtype=torch.long)
    assembled = collator.assemble({
        "text": list(sample["text"]),
        "audio_tokens": audio,
        "pose_tokens": pose,
    })
    return assembled["ids"][:, 0].tolist()


def _build_backbone_input(text_frames, audio_frames, pose_frames):
    """Tensors matching the collator's per-frame layout for the current prefix.

    text_frames: list[int], length N
    audio_frames: list[list[int]] of length N, each inner list has AUDIO_DEPTH
                  raw (un-shifted) codes
    pose_frames:  list[list[int]] of length N, each inner list has POSE_DEPTH
                  raw codes
    Returns backbone_ids (1, N, 3), audio_depth_ids (1, N, AUDIO_DEPTH),
    pose_depth_ids (1, N, POSE_DEPTH), separator_mask (1, N).
    """
    audio_arr = torch.tensor(audio_frames, dtype=torch.long, device=DEVICE)
    pose_arr = torch.tensor(pose_frames, dtype=torch.long, device=DEVICE)

    backbone_ids = torch.empty((1, len(text_frames), 3), dtype=torch.long, device=DEVICE)
    backbone_ids[0, :, 0] = torch.tensor(text_frames, dtype=torch.long, device=DEVICE)
    backbone_ids[0, :, 1] = audio_arr[:, 0] + AUDIO_TOKENS_START
    backbone_ids[0, :, 2] = pose_arr[:, 0] + POSE_TOKENS_START

    audio_offsets = torch.arange(AUDIO_DEPTH, device=DEVICE) * AUDIO_CODEBOOK_SIZE
    pose_offsets = torch.arange(POSE_DEPTH, device=DEVICE) * POSE_CODEBOOK_SIZE
    audio_depth_ids = (audio_arr + audio_offsets).unsqueeze(0)
    pose_depth_ids = (pose_arr + pose_offsets).unsqueeze(0)

    separator_mask = torch.zeros((1, len(text_frames)), dtype=torch.bool, device=DEVICE)
    return backbone_ids, audio_depth_ids, pose_depth_ids, separator_mask


def _backbone_forward(
    model,
    backbone_ids,
    audio_depth_ids,
    pose_depth_ids,
    separator_mask,
    reference_pose_embed=None,
):
    """Mirror EndToEndModel.forward_speech_pose's embedding build, then call
    the bare Qwen3 forward so we get logits + hidden states without losses.

    If reference_pose_embed is provided (shape (1, 1, hidden)), it is concatenated
    at the front of the interleaved sequence, matching the use_reference_pose
    branch of forward_speech_pose (backbone_model.py:236-243).
    """
    not_separator_mask = ~separator_mask

    llm_embeddings = model.get_input_embeddings()(backbone_ids)
    text_embeds = llm_embeddings[:, :, 0]
    audio_embeds = llm_embeddings[:, :, 1].clone()
    pose_embeds = llm_embeddings[:, :, 2].clone()

    true_audio_depth_ids = audio_depth_ids[not_separator_mask]
    tail_audio_ids = true_audio_depth_ids[:, 1:]
    tail_audio_embeds = model.audio_embedding(tail_audio_ids).sum(dim=1)
    audio_embeds[not_separator_mask] = audio_embeds[not_separator_mask] + tail_audio_embeds

    true_pose_depth_ids = pose_depth_ids[not_separator_mask]
    tail_pose_ids = true_pose_depth_ids[:, 1:]
    tail_pose_embeds = model.pose_embedding(tail_pose_ids).sum(dim=1)
    pose_embeds[not_separator_mask] = pose_embeds[not_separator_mask] + tail_pose_embeds

    backbone_embeds = torch.stack([text_embeds, audio_embeds, pose_embeds], dim=2)
    interleaved_embeds = backbone_embeds.flatten(1, 2)  # (1, S*3, h)

    if reference_pose_embed is not None:
        interleaved_embeds = torch.cat([reference_pose_embed, interleaved_embeds], dim=1)

    return Qwen3ForCausalLM.forward(
        model,
        inputs_embeds=interleaved_embeds,
        output_hidden_states=True,
        use_cache=False,
    )


def _build_reference_pose_embed(model, sample):
    """Frame-0 full pose latent for prepending, identical in form to
    EndToEndModel.forward_speech_pose's reference-pose construction
    (backbone_model.py:236-243).

    cb0 token (offset by POSE_TOKENS_START so it indexes the extended LLM vocab)
    is embedded via the backbone's input embedding; cb1..cb{POSE_DEPTH-1} (offset
    by d * POSE_CODEBOOK_SIZE) are embedded via model.pose_embedding and summed
    across the depth axis. The two are added to form one hidden-size vector,
    returned with shape (1, 1, hidden) ready to concat at the start of the
    interleaved sequence. Uses the GT first-frame pose from the eval sample,
    matching the training distribution.
    """
    pose_flat = torch.tensor(sample["pose_tokens"], dtype=torch.long)
    pose = pose_flat.view(-1, POSE_DEPTH).T.contiguous()  # (POSE_DEPTH, T)

    cb0_id = torch.tensor(
        [pose[0, 0].item() + POSE_TOKENS_START], dtype=torch.long, device=DEVICE
    )
    cb0_embed = model.get_input_embeddings()(cb0_id)  # (1, h)

    tail_offsets = torch.arange(1, POSE_DEPTH, device=DEVICE) * POSE_CODEBOOK_SIZE
    tail_ids = pose[1:, 0].to(DEVICE) + tail_offsets  # (POSE_DEPTH-1,)
    tail_embed = model.pose_embedding(tail_ids).sum(dim=0, keepdim=True)  # (1, h)

    return (cb0_embed + tail_embed).unsqueeze(0)  # (1, 1, h)


def _generate_depth_codes(depth_model, projection, text_hidden, code0, depth, codebook_size):
    """Autoregressive depth decode for codebooks 1..depth-1.

    Mirrors the training-time depth input layout: a length-(depth+1) sequence
    of [projected_text_hidden, embed(code0_raw), embed(code1+codebook_size),
    embed(code2+2*codebook_size), ..., embed(code_{depth-1}+(depth-1)*codebook_size)].
    code0 is embedded with no offset because codebook 0's training shift is 0.
    """
    projected = projection(text_hidden)  # (1, depth_hidden)
    light_inputs = projected.unsqueeze(1)  # (1, 1, depth_hidden)

    code0_id = torch.tensor([code0], dtype=torch.long, device=DEVICE)
    light_inputs = torch.cat(
        [light_inputs, depth_model.get_input_embeddings()(code0_id).unsqueeze(1)],
        dim=1,
    )

    tail_codes = []
    for codebook in range(1, depth):
        out = depth_model(inputs_embeds=light_inputs)
        block_start = codebook * codebook_size
        block_logits = out.logits[0, -1, block_start : block_start + codebook_size]
        sampled = _sample(block_logits, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K)
        tail_codes.append(sampled)
        if codebook != depth - 1:
            depth_id = torch.tensor(
                [sampled + codebook * codebook_size], dtype=torch.long, device=DEVICE
            )
            light_inputs = torch.cat(
                [light_inputs, depth_model.get_input_embeddings()(depth_id).unsqueeze(1)],
                dim=1,
            )
    return tail_codes


# generation -----------------------------------------------------------------
generated = []
for sample_idx in range(NUM_GENERATE_SAMPLES):
    sample = eval_dataset[sample_idx]
    text_track = _text_track_for_sample(sample)
    num_frames = min(len(text_track), MAX_GENERATE_FRAMES)
    text_track = text_track[:num_frames]

    audio_history = []   # list[list[int]] of length k, each row has AUDIO_DEPTH raw codes
    pose_history = []    # list[list[int]] of length k, each row has POSE_DEPTH raw codes

    # text source for this sample. When teacher-forcing, we use the full GT text
    # track. Otherwise we seed with the first actual text token from the GT
    # (skipping word_pad and new_word markers) and let the backbone extend the
    # track autoregressively, one token per frame.
    if TEACHER_FORCE_TEXT:
        used_text_track = list(text_track)
    else:
        seed_idx = next(
            (
                i
                for i, t in enumerate(text_track)
                if t not in (WORD_PAD_TOKEN, NEW_WORD_TOKEN)
            ),
            0,
        )
        used_text_track = [text_track[seed_idx]]

    print(f"\nsample {sample_idx}: {num_frames} frames to generate")
    with torch.inference_mode():
        reference_pose_embed = (
            _build_reference_pose_embed(model, sample) if USE_REFERENCE_POSE else None
        )
        text_stride_offset = 1 if USE_REFERENCE_POSE else 0

        for k in range(num_frames):
            # extend with dummies for the current frame; real codes overwrite below
            audio_history.append([0] * AUDIO_DEPTH)
            pose_history.append([0] * POSE_DEPTH)

            # --- forward 1: predict audio0_k from text-position logit ---
            backbone_ids, audio_depth_ids, pose_depth_ids, separator_mask = _build_backbone_input(
                used_text_track[: k + 1], audio_history, pose_history
            )
            outputs = _backbone_forward(
                model, backbone_ids, audio_depth_ids, pose_depth_ids, separator_mask,
                reference_pose_embed=reference_pose_embed,
            )

            text_position = 3 * k + text_stride_offset
            audio0_logits = outputs.logits[
                0, text_position, AUDIO_TOKENS_START : AUDIO_TOKENS_START + AUDIO_CODEBOOK_SIZE
            ]
            audio0 = _sample(audio0_logits, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K)

            # column-0 (text-position) hidden: the state that produced audio_cb0. Seeds audio depth.
            text_hidden = outputs.hidden_states[-1][:, text_position, :]

            # --- audio depth pass ---
            if DISABLE_AUDIO_DEPTH_MODEL:
                audio_tail = [0] * (AUDIO_DEPTH - 1)
            else:
                audio_tail = _generate_depth_codes(
                    model.audio_depth_model,
                    model.audio_projection,
                    text_hidden,
                    audio0,
                    AUDIO_DEPTH,
                    AUDIO_CODEBOOK_SIZE,
                )
            audio_history[k] = [audio0] + audio_tail

            # --- forward 2: predict pose0_k from audio-position logit ---
            # audio0 + audio tail are now real in the history; pose still dummy
            backbone_ids, audio_depth_ids, pose_depth_ids, separator_mask = _build_backbone_input(
                used_text_track[: k + 1], audio_history, pose_history
            )
            outputs = _backbone_forward(
                model, backbone_ids, audio_depth_ids, pose_depth_ids, separator_mask,
                reference_pose_embed=reference_pose_embed,
            )

            audio_position = 3 * k + 1 + text_stride_offset
            pose0_logits = outputs.logits[
                0, audio_position, POSE_TOKENS_START : POSE_TOKENS_START + POSE_CODEBOOK_SIZE
            ]
            pose0 = _sample(pose0_logits, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K)

            # column-1 (audio-position) hidden: the state that produced pose_cb0. Seeds pose depth.
            audio_hidden = outputs.hidden_states[-1][:, audio_position, :]

            # --- pose depth pass ---
            if DISABLE_POSE_DEPTH_MODEL:
                pose_tail = [0] * (POSE_DEPTH - 1)
            else:
                pose_tail = _generate_depth_codes(
                    model.pose_depth_model,
                    model.pose_projection,
                    audio_hidden,
                    pose0,
                    POSE_DEPTH,
                    POSE_CODEBOOK_SIZE,
                )
            pose_history[k] = [pose0] + pose_tail

            # --- forward 3: predict text_{k+1} from pose-position logit ---
            # Required only when we are not teacher-forcing the text track and
            # there is a next frame to seed. Uses real pose0_k in the input
            # (mirrors training-time conditioning: text_{k+1} | prefix+text_k+
            # audio0_k+pose0_k). Restricted to the text-vocab slice [0,
            # AUDIO_TOKENS_START), the symmetric move to how audio0/pose0 are
            # sampled from their respective codebook slices only.
            if not TEACHER_FORCE_TEXT and k + 1 < num_frames:
                backbone_ids, audio_depth_ids, pose_depth_ids, separator_mask = _build_backbone_input(
                    used_text_track[: k + 1], audio_history, pose_history
                )
                outputs = _backbone_forward(
                    model, backbone_ids, audio_depth_ids, pose_depth_ids, separator_mask,
                    reference_pose_embed=reference_pose_embed,
                )
                pose_position = 3 * k + 2 + text_stride_offset
                text_logits = outputs.logits[0, pose_position, :AUDIO_TOKENS_START]
                next_text = _sample(text_logits, DO_SAMPLE, TEMPERATURE, TOP_P, TOP_K)
                used_text_track.append(next_text)

            if (k + 1) % 16 == 0 or k == num_frames - 1:
                print(
                    f"  frame {k + 1}/{num_frames}: audio0={audio0} pose0={pose0}",
                    flush=True,
                )

    audio_codes = torch.tensor(audio_history, dtype=torch.long).T.contiguous()  # (AUDIO_DEPTH, N)
    pose_codes = torch.tensor(pose_history, dtype=torch.long).T.contiguous()    # (POSE_DEPTH, N)
    generated.append({"audio_codes": audio_codes, "pose_codes": pose_codes})

    audio_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"audio_{sample_idx}.pt")
    pose_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"pose_{sample_idx}.pt")
    torch.save(audio_codes, audio_path)
    torch.save(pose_codes, pose_path)
    print(f"  saved {audio_path} and {pose_path}")
    print(
        f"  done. audio_codes={tuple(audio_codes.shape)} pose_codes={tuple(pose_codes.shape)}"
    )

print(f"\ngenerated {len(generated)} samples")
