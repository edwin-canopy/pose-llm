"""
Uses Qwen 3-4B for backbone LLM
"""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    Gemma4UnifiedForConditionalGeneration,
    Qwen3ForCausalLM,
)

from models.swiglu import Swiglu

import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)


START_OF_ASR_TRANSCRIPT_TOKEN = 151937 # start of asr transcription token
START_OF_ASR_TOKEN = 151938 # start of asr task token (before the transcribe this prompt)



class EndToEndModel(Qwen3ForCausalLM):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        backbone_cfg = config["backbone"]
        audio_cfg = config["audio_depth_model"]
        pose_cfg = config["pose_depth_model"]

        self.hidden_size = backbone_cfg["hidden_size"]
        self.audio_hidden_size = audio_cfg["hidden_size"]
        self.pose_hidden_size = pose_cfg["hidden_size"]

        self.audio_depth = audio_cfg["residual_depth"]
        self.pose_depth = pose_cfg["residual_depth"]
        self.audio_codebook_size = audio_cfg["codebook_size"]
        self.pose_codebook_size = pose_cfg["codebook_size"]

        # depth llms: mirror qwen-train/models/qwen_model.py:23-32 -- load
        # AutoConfig from a HF checkpoint to get the architecture defaults,
        # override vocab_size to (residual_depth * codebook_size), override
        # hidden_size from config.yaml, then instantiate the HF class directly
        # (random init -- no from_pretrained). One config per modality so the
        # two latent dims are independent.
        audio_depth_config = AutoConfig.from_pretrained(audio_cfg["path"])
        audio_depth_config.vocab_size = self.audio_depth * self.audio_codebook_size
        audio_depth_config.hidden_size = self.audio_hidden_size
        self.audio_depth_model = Gemma4UnifiedForConditionalGeneration(audio_depth_config)

        pose_depth_config = AutoConfig.from_pretrained(pose_cfg["path"])
        pose_depth_config.vocab_size = self.pose_depth * self.pose_codebook_size
        pose_depth_config.hidden_size = self.pose_hidden_size
        self.pose_depth_model = Gemma4UnifiedForConditionalGeneration(pose_depth_config)

        # backbone-hidden -> modality-hidden seed used by each depth llm
        self.audio_projection = nn.Linear(self.hidden_size, self.audio_hidden_size)
        self.pose_projection = nn.Linear(self.hidden_size, self.pose_hidden_size)

        # backbone-side embedding tables for the summed codebook tail
        # (one slot per (codebook, code) pair, matches collator's per-codebook offsets)
        self.audio_embedding = nn.Embedding(
            self.audio_depth * self.audio_codebook_size, self.hidden_size
        )
        self.pose_embedding = nn.Embedding(
            self.pose_depth * self.pose_codebook_size, self.hidden_size
        )

        self.alpha_audio = config["alpha_audio"]
        self.alpha_pose = config["alpha_pose"]
        self.audio_model_train_split = audio_cfg["training_split"]
        self.pose_model_train_split = pose_cfg["training_split"]

        # self.logmel_projection = Swiglu(
        #
        # )


    def forward(self):
        """
        depth model gets zeroth codebook + zeroth codebook latent and predicts next d-1 latents
        mask loss on first two conditioning latents
        """
        pass


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
        pose_pad_mask,
        **kwargs
    ):
        """
        backbone ids: (S, 3) - text, zeroth audio codebook, zeroth pose codebook

        note pose tokenizer is at 6.25 fps, so every other pose token is a pose padding token

        TODO pose pad handle at half of original fps
        """
        raise NotImplementedError("pose padding needs to be handled")

        not_separator_mask = ~separator_mask

        llm_embeddings = self.get_input_embeddings()(backbone_ids)
        text_embeds = llm_embeddings[:, :, 0]
        audio_embeds = llm_embeddings[:, :, 1].clone()
        pose_embeds = llm_embeddings[:, :, 2].clone()

        true_audio_depth_ids = audio_depth_ids[not_separator_mask]
        tail_audio_ids = true_audio_depth_ids[:, 1:] # codebooks 1-7
        tail_audio_embeds = self.audio_embedding(tail_audio_ids).sum(dim=1)
        audio_embeds[not_separator_mask] = audio_embeds[not_separator_mask] + tail_audio_embeds # add higher order codebooks

        true_pose_depth_ids = pose_depth_ids[not_separator_mask]
        tail_pose_ids = true_pose_depth_ids[:, 1:] # codebooks 1-7
        tail_pose_embeds = self.pose_embedding(tail_pose_ids).sum(dim=1)
        pose_embeds[not_separator_mask] = pose_embeds[not_separator_mask] + tail_pose_embeds # add higher order codebooks

        backbone_embeds = torch.stack([text_embeds, audio_embeds, pose_embeds], dim=2)
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

        text_hidden_states = backbone_hidden_states[:, 0::3][not_separator_mask]


        # audio depth pass --------------

        projected_hidden_states = self.audio_projection(text_hidden_states)
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
        
        projected_hidden_states = self.pose_projection(text_hidden_states)
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

        loss = (1 - self.alpha_audio - self.alpha_pose) * backbone_loss + self.alpha_audio * audio_depth_loss + self.alpha_pose * pose_depth_loss
        return {"loss": loss}

    