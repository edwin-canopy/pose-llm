import torch
from transformers import QwenModelForCausalLM, PreTrainedModel, AutoTokenizer

import yaml
# claude
yaml.safe_load("config.yaml")

backbone_class = QwenModelForCausalLM
depth_model_class = QwenModelForCausalLM


class EndToEndModel(backbone_class):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depth_model = None


    def _add_depth_model(self, depth_model):
        assert self.depth_model is None, "Tried to add depth model but it already exists."
        self.depth_model = depth_model


    def forward(self):
        """
        depth model gets zeroth codebook + zeroth codebook latent and predicts next d-1 latents
        mask loss on first two conditioning latents
        """
        pass


class DepthTransformer(depth_model_class):

    def __init__(self, model_class, *args, **kwargs):
        super().__init__(*args, **kwargs)


