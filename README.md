# pose-llm

## Training
* `train_pose.py` runs a simple interleaved training test - no question input, just interleaved generation.
* `train.py` runs a full single task question prepended interleaved run.

## Running inference
* Scripts under `inference/`, use `inference/decode_{pose/audio}.py` to convert generated samples in `inference_outputs` to audio and pose video (normalised and centered).
* Use correct pose tokenizer, by default we are working with `pose_tokenizer_xabi/`.
* Use `inference/comparisons.py` to plot comparison plots.
* !! important - inference scripts read config from `config.yaml` - make sure the settings we are using correspond to the correct weights, eg. the value of k-shifting and pose prepending

Note that each new infernce run wipes the outputs folder and re-writes its outputs to it.