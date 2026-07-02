"""
Uses Qwen 3-4B for backbone LLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    LlamaForCausalLM,
    Qwen3ForCausalLM,
)

DEFAULT_BACKBONE_ARCH = "Qwen/Qwen3-4B"
DEFAULT_DEPTH_ARCH = "HuggingFaceTB/SmolLM2-360M"

from models.swiglu import Swiglu

# check these against repo
START_OF_ASR_TRANSCRIPT_TOKEN = 151937 # start of asr transcription token
START_OF_ASR_TOKEN = 151938 # start of asr task token (before the transcribe this prompt)


class EndToEndModel(Qwen3ForCausalLM):
    
    def __init__(self, config, *args, yaml_config=None, **kwargs):
        # Two call conventions:
        #   fresh init:      EndToEndModel(yaml_dict, hf_config)
        #   from_pretrained: EndToEndModel(hf_pretrained_config, yaml_config=yaml_dict)
        # HF calls cls(hf_config, *model_args, **model_kwargs); yaml_config rides in via kwargs.
        if isinstance(config, dict):
            yaml_config = config
            super().__init__(*args, **kwargs)
        else:
            assert yaml_config is not None, "yaml_config kwarg required when first arg is a PreTrainedConfig"
            super().__init__(config, *args, **kwargs)

        backbone_cfg = yaml_config["backbone"]
        audio_cfg = yaml_config["audio_depth_model"]
        pose_cfg = yaml_config["pose_depth_model"]

        self.hidden_size = backbone_cfg["hidden_size"]
        self.audio_hidden_size = audio_cfg["hidden_size"]
        self.pose_hidden_size = pose_cfg["hidden_size"]

        self.audio_depth = audio_cfg["residual_depth"]
        self.pose_depth = pose_cfg["residual_depth"]
        self.audio_codebook_size = audio_cfg["codebook_size"]
        self.pose_codebook_size = pose_cfg["codebook_size"]

        attn_impl = yaml_config["training"]["attn_implementation"]
        audio_depth_config = AutoConfig.from_pretrained(
            DEFAULT_DEPTH_ARCH, attn_implementation=attn_impl
        )
        audio_depth_config.vocab_size = self.audio_depth * self.audio_codebook_size
        audio_depth_config.hidden_size = self.audio_hidden_size
        if audio_cfg["weights_path"]:
            self.audio_depth_model = LlamaForCausalLM.from_pretrained(
                audio_cfg["weights_path"], config=audio_depth_config
            )
        else:
            self.audio_depth_model = LlamaForCausalLM(audio_depth_config)

        pose_depth_config = AutoConfig.from_pretrained(
            DEFAULT_DEPTH_ARCH, attn_implementation=attn_impl
        )
        pose_depth_config.vocab_size = self.pose_depth * self.pose_codebook_size
        pose_depth_config.hidden_size = self.pose_hidden_size
        if pose_cfg["weights_path"]:
            self.pose_depth_model = LlamaForCausalLM.from_pretrained(
                pose_cfg["weights_path"], config=pose_depth_config
            )
        else:
            self.pose_depth_model = LlamaForCausalLM(pose_depth_config)

        # backbone-hidden -> modality-hidden seed used by each depth llm
        self.audio_projection = nn.Linear(self.hidden_size, self.audio_hidden_size)
        self.pose_projection = nn.Linear(self.hidden_size, self.pose_hidden_size)

        # backbone-side embedding tables for the summed codebook tail
        # (one slot per (codebook, code) pair, matches collator's per-codebook offsets)
        # NOTE weights for the zeroth codebooks here are never used, as we use the LLM's input embedding
        self.audio_embedding = nn.Embedding(
            self.audio_depth * self.audio_codebook_size, self.hidden_size
        )
        self.pose_embedding = nn.Embedding(
            self.pose_depth * self.pose_codebook_size, self.hidden_size
        )

        self.alpha_audio = yaml_config["training"]["alpha_audio"]
        self.alpha_pose = yaml_config["training"]["alpha_pose"]
        self.audio_model_train_split = audio_cfg["training_split"]
        self.pose_model_train_split = pose_cfg["training_split"]
        self.use_reference_pose = backbone_cfg["use_reference_pose"]

        # self.logmel_projection = Swiglu(
        #
        # )


    def forward(self, **kwargs):
        return self.forward_speech_pose(**kwargs)


    def forward_text(
        self,
        input_ids,
        attention_mask,
        labels,
        **kwargs,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    

    def forward_asr(
        self,
        input_ids,
        labels,
        audio_mask,
        audio_tokens,
        **kwargs
    ):
        pass


    def forward_speech(
        self,
        backbone_ids,
        depth_ids,
        separator_mask,
        **kwargs
    ):
        """
        backbone_ids: (S, 2) - text and zeroth audio codebook
        """
        not_separator_mask = ~separator_mask

        llm_embeddings = self.get_input_embeddings()(backbone_ids)
        text_embeds = llm_embeddings[:, :, 0]
        audio_embeds = llm_embeddings[:, :, 1].clone()

        true_depth_ids = depth_ids[not_separator_mask]
        tail_audio_ids = true_depth_ids[:, 1:] # codebooks 1-7
        tail_audio_embeds = self.audio_embedding(tail_audio_ids).sum(dim=1)
        audio_embeds[not_separator_mask] = audio_embeds[not_separator_mask] + tail_audio_embeds # add higher order codebooks

        backbone_embeds = torch.stack([text_embeds, audio_embeds], dim=2)
        interleaved_embeds = backbone_embeds.flatten(1, 2)  # (b, s, p, h) -> (b, s*p, h)

        labels = backbone_ids.clone()
        labels[separator_mask] = -100
        labels = labels.flatten(1, 2)  # (b, s, p) -> (b, s*p)

        backbone_outputs = super().forward(
            inputs_embeds=interleaved_embeds,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        ) # backbone zeroth token + text predictions

        backbone_loss = backbone_outputs.loss
        backbone_hidden_states = backbone_outputs.hidden_states[-1]
        
        text_hidden_states = backbone_hidden_states[:, 0::2][not_separator_mask]
        projected_hidden_states = self.audio_projection(text_hidden_states)
        audio_depth_embeds = self.audio_depth_model.get_input_embeddings()(true_depth_ids)
        depth_inputs = torch.cat([projected_hidden_states.unsqueeze(1), audio_depth_embeds], dim=1)

        # true depth ids: (F, 8) where F is the number of true (non-separator) frames
        # projected hidden states: (F, backbone_hidden)
        # audio depth embeds: (F, 8, depth_hidden)
        # depth inputs: (F, 9) - backbone embedding + codebook embeddings

        # label the first two embeddings of depth inputs as -100
        ignore = torch.full((true_depth_ids.size(0), 2), -100, dtype=torch.long, device=true_depth_ids.device)
        depth_labels = torch.cat([ignore, true_depth_ids[:, 1:]], dim=1)

        train_num = int(self.audio_model_train_split * depth_inputs.size(0))
        train_idxs = torch.randperm(depth_inputs.size(0), device=depth_inputs.device)[:train_num]
        depth_inputs = depth_inputs[train_idxs]
        depth_labels = depth_labels[train_idxs]

        depth_outputs = self.audio_depth_model(
            inputs_embeds=depth_inputs,
            labels=depth_labels
        )

        loss = (1 - self.alpha_audio) * backbone_loss + self.alpha_audio * depth_outputs.loss
        return {"loss": loss}


    def forward_speech_pose(
        self,
        backbone_ids,
        audio_depth_ids,
        pose_depth_ids,
        separator_mask,
        lookahead_mask=None,
        **kwargs
    ):
        """
        backbone ids: (S, 3) - text, zeroth audio codebook, zeroth pose codebook
        lookahead_mask: (B, S, 3) - True where the target at that column is
            lookahead_padding_token (collator-emitted). Excluded from the
            backbone loss and from the audio/pose depth-tail passes.
        """
        not_separator_mask = ~separator_mask

        if lookahead_mask is None:
            lookahead_mask = torch.zeros((*backbone_ids.shape[:2], 3), dtype=torch.bool, device=backbone_ids.device)

        # Per-column "keep" masks: real (not separator, not lookahead pad).
        keep_audio = not_separator_mask & ~lookahead_mask[..., 1]
        keep_pose = not_separator_mask & ~lookahead_mask[..., 2]

        llm_embeddings = self.get_input_embeddings()(backbone_ids)
        text_embeds = llm_embeddings[:, :, 0]
        audio_embeds = llm_embeddings[:, :, 1].clone()
        pose_embeds = llm_embeddings[:, :, 2].clone()

        true_audio_depth_ids = audio_depth_ids[keep_audio]
        tail_audio_ids = true_audio_depth_ids[:, 1:] # codebooks 1-7
        tail_audio_embeds = self.audio_embedding(tail_audio_ids).sum(dim=1)
        audio_embeds[keep_audio] = audio_embeds[keep_audio] + tail_audio_embeds # add higher order codebooks

        true_pose_depth_ids = pose_depth_ids[keep_pose]
        tail_pose_ids = true_pose_depth_ids[:, 1:] # codebooks 1-7
        tail_pose_embeds = self.pose_embedding(tail_pose_ids).sum(dim=1)
        pose_embeds[keep_pose] = pose_embeds[keep_pose] + tail_pose_embeds # add higher order codebooks

        backbone_embeds = torch.stack([text_embeds, audio_embeds, pose_embeds], dim=2)
        interleaved_embeds = backbone_embeds.flatten(1, 2)  # (b, s, p, h) -> (b, s*p, h)

        labels = backbone_ids.clone()
        labels[separator_mask] = -100
        labels[lookahead_mask] = -100  # mask lookahead-pad targets per column
        labels = labels.flatten(1, 2)  # (b, s, p) -> (b, s*p)

        if self.use_reference_pose:
            # full first-frame pose latent (cb0 via LLM vocab + summed cb1..N via pose_embedding)
            ref_cb0_pose_embed = self.get_input_embeddings()(backbone_ids[:, 0, 2])
            ref_tail_pose_embeds = self.pose_embedding(pose_depth_ids[:, 0, 1:]).sum(dim=1)
            ref_pose_embed = (ref_cb0_pose_embed + ref_tail_pose_embeds).unsqueeze(1)
            interleaved_embeds = torch.cat([ref_pose_embed, interleaved_embeds], dim=1)
            ref_label = torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([ref_label, labels], dim=1)

        backbone_outputs = super().forward(
            inputs_embeds=interleaved_embeds,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        ) # backbone zeroth token + text predictions

        backbone_loss = backbone_outputs.loss
        backbone_hidden_states = backbone_outputs.hidden_states[-1]

        text_stride_offset = 1 if self.use_reference_pose else 0
        # Column-0 hidden (sits on the text column) is the state that predicts audio_cb0
        # — seeds the audio depth head. Column-1 hidden (sits on the audio-cb0 column) is
        # the state that predicts pose_cb0 — seeds the pose depth head.
        text_hidden_per_frame = backbone_hidden_states[:, text_stride_offset::3]
        audio_hidden_per_frame = backbone_hidden_states[:, text_stride_offset + 1::3]

        # per-column backbone losses (text / audio cb0 / pose cb0) for logging only.
        # outputs.logits is materialized when fused_linear_cross_entropy is off.
        with torch.no_grad():
            logits = backbone_outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            S_flat = labels.size(1)
            positions = torch.arange(S_flat, device=labels.device)
            adj = positions - text_stride_offset
            valid = positions >= text_stride_offset
            is_text_col = valid & (adj % 3 == 0)
            is_audio_col = valid & (adj % 3 == 1)
            is_pose_col = valid & (adj % 3 == 2)

            def _col_ce(col_mask):
                m = col_mask[1:]  # align with shift positions
                sel_logits = shift_logits[:, m, :].reshape(-1, shift_logits.size(-1))
                sel_labels = shift_labels[:, m].reshape(-1)
                return F.cross_entropy(sel_logits, sel_labels, ignore_index=-100)

            text_backbone_loss = _col_ce(is_text_col)
            audio_backbone_loss = _col_ce(is_audio_col)
            pose_backbone_loss = _col_ce(is_pose_col)


        # audio depth pass --------------

        projected_hidden_states = self.audio_projection(text_hidden_per_frame[keep_audio])
        audio_depth_embeds = self.audio_depth_model.get_input_embeddings()(true_audio_depth_ids)
        audio_depth_inputs = torch.cat([projected_hidden_states.unsqueeze(1), audio_depth_embeds], dim=1)

        # true depth ids: (F, 8) where F is the number of true (non-separator) frames
        # projected hidden states: (F, backbone_hidden)
        # audio depth embeds: (F, 8, depth_hidden)
        # depth inputs: (F, 9) - backbone embedding + codebook embeddings

        # label the first two embeddings of depth inputs as -100
        ignore = torch.full((true_audio_depth_ids.size(0), 2), -100, dtype=torch.long, device=true_audio_depth_ids.device)
        audio_depth_labels = torch.cat([ignore, true_audio_depth_ids[:, 1:]], dim=1)

        train_num = int(self.audio_model_train_split * audio_depth_inputs.size(0))
        train_idxs = torch.randperm(audio_depth_inputs.size(0), device=audio_depth_inputs.device)[:train_num]
        audio_depth_inputs = audio_depth_inputs[train_idxs]
        audio_depth_labels = audio_depth_labels[train_idxs]

        audio_depth_outputs = self.audio_depth_model(
            inputs_embeds=audio_depth_inputs,
            labels=audio_depth_labels
        )

        audio_depth_loss = audio_depth_outputs.loss


        # pose depth pass --------------

        projected_hidden_states = self.pose_projection(audio_hidden_per_frame[keep_pose])
        pose_depth_embeds = self.pose_depth_model.get_input_embeddings()(true_pose_depth_ids)
        pose_depth_inputs = torch.cat([projected_hidden_states.unsqueeze(1), pose_depth_embeds], dim=1)

        ignore = torch.full((true_pose_depth_ids.size(0), 2), -100, dtype=torch.long, device=true_pose_depth_ids.device)
        pose_depth_labels = torch.cat([ignore, true_pose_depth_ids[:, 1:]], dim=1)

        train_num = int(self.pose_model_train_split * pose_depth_inputs.size(0))
        train_idxs = torch.randperm(pose_depth_inputs.size(0), device=pose_depth_inputs.device)[:train_num]
        pose_depth_inputs = pose_depth_inputs[train_idxs]
        pose_depth_labels = pose_depth_labels[train_idxs]

        pose_depth_outputs = self.pose_depth_model(
            inputs_embeds=pose_depth_inputs,
            labels=pose_depth_labels
        )

        pose_depth_loss = pose_depth_outputs.loss

        # per-codebook pose depth losses (codebooks 1..pose_depth-1). Input position i
        # in pose_depth_inputs predicts pose_depth_labels[:, i+1]; codebook k label sits
        # at label position k+1, so the predicting logit is at position k.
        with torch.no_grad():
            pose_depth_logits = pose_depth_outputs.logits
            per_codebook_pose_losses = {}
            for cb_idx in range(1, self.pose_depth):
                logits_cb = pose_depth_logits[:, cb_idx, :].reshape(-1, pose_depth_logits.size(-1))
                labels_cb = pose_depth_labels[:, cb_idx + 1].reshape(-1)
                per_codebook_pose_losses[f"pose_depth_cb{cb_idx}_loss"] = F.cross_entropy(
                    logits_cb, labels_cb, ignore_index=-100
                )

        loss = (1 - self.alpha_audio - self.alpha_pose) * backbone_loss + self.alpha_audio * audio_depth_loss + self.alpha_pose * pose_depth_loss
        return {
            "loss": loss,
            "backbone_loss": backbone_loss.detach(),
            "text_backbone_loss": text_backbone_loss,
            "audio_backbone_loss": audio_backbone_loss,
            "pose_backbone_loss": pose_backbone_loss,
            "audio_depth_loss": audio_depth_loss.detach(),
            "pose_depth_loss": pose_depth_loss.detach(),
            **per_codebook_pose_losses,
        }

    