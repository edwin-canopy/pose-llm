"""
Trainer subclass used for multi task training
and for speed profiling
"""

from transformers import TRainer

class PoseSpeechTrainer(Trainer):
    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)


    
    def training_step(sewlf, model, inputs, *args, **kwargs):
        pass