"""
Uses Gemma 300M for depth LLM
"""

from transformers import Gemma4UnifiedForConditionalGeneration



# for audio model
# 8 x 2048 vocab size

# for pose model
# 8 x 1024 vocab size


class DepthTransformer(Gemma4UnifiedForConditionalGeneration):
    """Make this configurable for use as both the audio transformer and pose tranformer"""

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pass


    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        pass


    def forward_speech(
        self,
    ):
        pass


    def forward_asr(
        self,
    ):
        pass
