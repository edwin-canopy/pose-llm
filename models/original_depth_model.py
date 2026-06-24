import os

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoConfig, LlamaForCausalLM, Qwen3ForCausalLM
from models.swiglu import Swiglu


CONFIG = yaml.safe_load(open("config.yaml"))
MODEL_CONFIG = CONFIG["model"]
DATA_CONFIG = CONFIG["asr_collator"]
SPECIAL_TOKEN_CONFIG = CONFIG["special_tokens"]
_TEXT_BACKBONE_SHAPE_PRINTS = 0


class MultiTaskQwen(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        light_model_config = MODEL_CONFIG["light_model"]
        light_model_path = (
            light_model_config["path"]
            if isinstance(light_model_config, dict)
            else light_model_config
        )
        light_config = AutoConfig.from_pretrained(light_model_path)
        light_vocab_size = MODEL_CONFIG["depth"] * MODEL_CONFIG["codebook_size"]
        light_config.vocab_size = light_vocab_size
        self.light_llm = LlamaForCausalLM(light_config)
        self.projection = nn.Linear(config.hidden_size, light_config.hidden_size)
        self.depth = MODEL_CONFIG["depth"]
        self.audio_tokens_start = SPECIAL_TOKEN_CONFIG["audio_tokens_start"]
        self.code_embedding = nn.Embedding(
            self.depth * MODEL_CONFIG["codebook_size"], config.hidden_size)
        self.logmel_projection = Swiglu(
            MODEL_CONFIG["n_filterbanks"] * DATA_CONFIG["pool_factor"], 
            config.hidden_size
            )
        self.COMPUTE_RATIO = MODEL_CONFIG["compute_ratio"]
        self.ALPHA = MODEL_CONFIG["alpha"]
        self.AUDIO_TOKENS_START = MODEL_CONFIG["audio_tokens_start"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Swiglu):
            with torch.no_grad():
                module.projection.weight[: module.out_features].zero_()

    def forward(self, input_ids=None, attention_mask=None, labels=None, task=None, eval_metrics=False, **kwargs):
        if task is None:
            task_id = 0
        else:
            task_id = int(task.reshape(-1)[0].item())
        if task_id == 0:
            return self.forward_text(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        if task_id == 1:
            if eval_metrics:
                return self.forward_speech_eval_metrics(**kwargs)
            return self.forward_speech(**kwargs)
        return self.forward_asr(input_ids=input_ids, labels=labels, **kwargs)

    def forward_text(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        global _TEXT_BACKBONE_SHAPE_PRINTS
        max_prints = int(os.environ.get("PRINT_BACKBONE_SHAPES", "0"))
        if (
            max_prints > 0
            and _TEXT_BACKBONE_SHAPE_PRINTS < max_prints
            and os.environ.get("RANK", "0") == "0"
        ):
            label_tokens = int((labels != -100).sum().item()) if labels is not None else None
            print(
                "[qwen text backbone] "
                f"input_ids={tuple(input_ids.shape) if input_ids is not None else None} "
                f"attention_mask={tuple(attention_mask.shape) if attention_mask is not None else None} "
                f"labels={tuple(labels.shape) if labels is not None else None} "
                f"label_tokens={label_tokens}",
                flush=True,
            )
            _TEXT_BACKBONE_SHAPE_PRINTS += 1
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def forward_speech(
        self,
        backbone_ids=None,
        depth_ids=None,
        separator_mask=None,
        return_outputs=False,
        **kwargs,
    ):
        non_separator_mask = ~separator_mask

        llm_embeds = self.get_input_embeddings()(backbone_ids)
        text_embeds = llm_embeds[:, :, 0]
        code_embeds = llm_embeds[:, :, 1].clone()

        real_depth_ids = depth_ids[non_separator_mask]
        tail_code_ids = real_depth_ids[:, 1:]
        tail_code_embeds = self.code_embedding(tail_code_ids).sum(dim=1)
        code_embeds[non_separator_mask] = code_embeds[non_separator_mask] + tail_code_embeds

        paired_embeds = torch.stack([text_embeds, code_embeds], dim=2)
        interleaved_embeds = rearrange(paired_embeds, "b s p h -> b (s p) h")

        labels = backbone_ids.clone()
        labels[separator_mask] = -100
        labels = rearrange(labels, "b s p -> b (s p)")

        outputs = super().forward(
            inputs_embeds=interleaved_embeds,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )

        backbone_loss = outputs.loss
        hidden_states = outputs.hidden_states[-1]
        if return_outputs:
            return outputs

        even_hidden_states = hidden_states[:, 0::2]
        even_hidden_states = even_hidden_states[non_separator_mask]

        projected_hidden_states = self.projection(even_hidden_states)

        code_light_embeds = self.light_llm.get_input_embeddings()(real_depth_ids)
        light_inputs = torch.cat([projected_hidden_states.unsqueeze(1), code_light_embeds], dim=1)

        ignore = torch.full((real_depth_ids.size(0), 2), -100, dtype=torch.long, device=real_depth_ids.device)
        light_labels = torch.cat([ignore, real_depth_ids[:, 1:]], dim=1)

        keep = int(light_inputs.size(0) * self.COMPUTE_RATIO)
        subset = torch.randperm(light_inputs.size(0), device=light_inputs.device)[:keep]

        light_inputs = light_inputs[subset]
        light_labels = light_labels[subset]

        light_outputs = self.light_llm(inputs_embeds=light_inputs, labels=light_labels)
        loss = (1 - self.ALPHA) * backbone_loss + self.ALPHA * light_outputs.loss
        return {"loss": loss}

    def _ce_sum_for_mask(self, logits, labels, mask):
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous() & (shift_labels != -100)
        if not shift_mask.any():
            zero = torch.zeros((), dtype=torch.float32, device=logits.device)
            count = torch.zeros((), dtype=torch.long, device=logits.device)
            return zero, count
        loss_sum = F.cross_entropy(
            shift_logits[shift_mask].float(),
            shift_labels[shift_mask],
            reduction="sum",
        )
        return loss_sum, shift_mask.sum()

    def forward_speech_eval_metrics(
        self,
        backbone_ids=None,
        depth_ids=None,
        separator_mask=None,
        **kwargs,
    ):
        non_separator_mask = ~separator_mask
        llm_embeds = self.get_input_embeddings()(backbone_ids)
        text_embeds = llm_embeds[:, :, 0]
        code_embeds = llm_embeds[:, :, 1].clone()
        real_depth_ids = depth_ids[non_separator_mask]
        tail_code_ids = real_depth_ids[:, 1:]
        tail_code_embeds = self.code_embedding(tail_code_ids).sum(dim=1)
        code_embeds[non_separator_mask] = code_embeds[non_separator_mask] + tail_code_embeds
        paired_embeds = torch.stack([text_embeds, code_embeds], dim=2)
        interleaved_embeds = rearrange(paired_embeds, "b s p h -> b (s p) h")

        labels = backbone_ids.clone()
        labels[separator_mask] = -100
        interleaved_labels = rearrange(labels, "b s p -> b (s p)")

        outputs = super().forward(
            inputs_embeds=interleaved_embeds,
            labels=interleaved_labels,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        even_hidden_states = hidden_states[:, 0::2]

        positions = torch.arange(interleaved_labels.size(1), device=interleaved_labels.device)[None, :]
        text_mask = ((positions % 2) == 0).expand_as(interleaved_labels)
        codebook0_mask = ((positions % 2) == 1).expand_as(interleaved_labels)
        pad_id = SPECIAL_TOKEN_CONFIG["speech"]["pad"]
        new_word_id = SPECIAL_TOKEN_CONFIG["speech"]["word"]
        text_pad_mask = text_mask & (interleaved_labels == pad_id)
        text_newword_mask = text_mask & (interleaved_labels == new_word_id)
        text_word_mask = text_mask & (interleaved_labels != pad_id) & (interleaved_labels != new_word_id)
        metrics = {
            "text_word": self._ce_sum_for_mask(outputs.logits, interleaved_labels, text_word_mask),
            "text_newword": self._ce_sum_for_mask(outputs.logits, interleaved_labels, text_newword_mask),
            "text_pad": self._ce_sum_for_mask(outputs.logits, interleaved_labels, text_pad_mask),
            "codebook0": self._ce_sum_for_mask(outputs.logits, interleaved_labels, codebook0_mask),
        }

        even_hidden_states = even_hidden_states[non_separator_mask]
        projected_hidden_states = self.projection(even_hidden_states)
        code_light_embeds = self.light_llm.get_input_embeddings()(real_depth_ids)
        light_inputs = torch.cat([projected_hidden_states.unsqueeze(1), code_light_embeds], dim=1)

        ignore = torch.full((real_depth_ids.size(0), 2), -100, dtype=torch.long, device=real_depth_ids.device)
        light_labels = torch.cat([ignore, real_depth_ids[:, 1:]], dim=1)
        if light_inputs.size(0) == 0:
            zero = torch.zeros((), dtype=torch.float32, device=interleaved_embeds.device)
            count = torch.zeros((), dtype=torch.long, device=interleaved_embeds.device)
            for codebook in range(1, self.depth):
                metrics[f"codebook{codebook}"] = (zero, count)
            return metrics

        light_outputs = self.light_llm(inputs_embeds=light_inputs, labels=light_labels)
        for codebook in range(1, self.depth):
            label_pos = codebook + 1
            codebook_mask = torch.zeros_like(light_labels, dtype=torch.bool)
            codebook_mask[:, label_pos] = True
            metrics[f"codebook{codebook}"] = self._ce_sum_for_mask(
                light_outputs.logits,
                light_labels,
                codebook_mask,
            )
        return metrics

    def forward_asr(
        self,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        **kwargs,
    ):

        embed_inputs = self.get_input_embeddings()

        inputs_embeds = embed_inputs(input_ids.clamp(min=0))
        audio_projected = self.logmel_projection(audio_tokens)

        audio_projected = audio_projected.to(inputs_embeds.dtype)
        inputs_embeds[audio_mask] = audio_projected

        return super().forward(
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            **kwargs,
        )