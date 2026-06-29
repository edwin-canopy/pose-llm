# pose-llm

## Training
* `train_pose.py` runs a simple interleaved training test - no question input, just interleaved generation.
* `train.py` runs a full single task question prepended interleaved run.

## Running inference
* Scripts under `inference/`, use `inference/decode_{pose/audio}.py` to convert generated samples in `inference_outputs` to audio and pose video (normalised and centered).

Note that each new infernce run wipes the outputs folder and re-writes its outputs to it.