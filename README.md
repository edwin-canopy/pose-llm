# pose-llm

Causal interleaved output for a question-answering / speech and pose prediction backbone.
Each sample looks like:

```
[pose0] [SOG] [text] [audio] [pose] [text] [audio] [pose]
```

where `[audio]` and `[pose]` contain the model's initial cb0 prediction and are modified retrospectively to contain embeddings for all eight codes. We condition the first token on the full pose embedding for the first frame. We optionally prepend a question to the whole sequence like:

```
[SOT] <question> [EOT] [pose0] [SOG] [text] etc.
```

The special tokens used and ranges of extended tokens can be found in `config.yaml`.

## Repository layout

```
.
|-- README.md   config.yaml   pyproject.toml
|
|-- train.py                        full QA-prepended interleaved training run (TODO)
|-- train_pose.py                   interleaved pose+audio training test (no question)
|-- preprocess_data.py              shard raw audio+alignments into per-source parquet chunks
|-- annotate_questions.py           Gemma 3 4B question-annotation over the merged dataset
|
|-- models/
|   |-- backbone_model.py           Qwen3-4B backbone + interleaved audio/pose heads
|   |-- swiglu.py                   SwiGLU MLP block (currently unused - this is for ASR)
|
|-- collators/
|   |-- conversational_pose_collator.py   pose/speech + user-text/audio conversational collators
|   |-- monostream_pose_collator.py       mono-stream interleaved pose+audio collator (training)
|   |-- text_collator.py                  text-only QA collator
|   |-- asr_collator.py                   ASR collator stub
|
|-- train_utils/
|   |-- generate_dataset.py         merge parquet shards + questions -> HF Dataset on disk
|   |-- filter_dataset.py           drop empty-text / bad rows from the HF dataset
|   |-- length_matching_sampler.py  length-packed distributed batch sampler
|   |-- trainer.py                  Trainer subclass for multi-task + speed profiling
|   |-- logger.py                   (empty) logging placeholder
|
|-- inference/
|   |-- audio_pose_interleaved_inference.py    mono-stream interleaved generation from checkpoint
|   |-- audio_pose_conversational_inference.py (empty) conversational-mode entrypoint
|   |-- decode_pose.py              pose codes -> joint positions -> gif via xabi tokenizer
|   |-- decode_audio.py             audio codes -> mp3 via kyutai/mimi codec
|   |-- eval_pose.py                render ground-truth pose gifs from the eval slice
|   |-- eval_audio.py               render ground-truth audio mp3s from the eval slice
|   |-- comparisons.py              side-by-side gt vs generated (+ random baseline) gifs
|   |-- full.sh                     run interleaved inference then decode + comparisons
|   |-- tokenizer_config.yaml       pose-tokenizer config used at decode time
```

## Setup
* `pyproject.toml` contains all the packages needed except for flash-attention 2. Install from prebuilt wheel.
* copy `example_dir.toml` to `dir.toml` in the root and edit each entry to point at the correct filesystem path for this vm. Every path used by the repo (datasets, tokenizer weights, etc.) is read from there via `paths.py`.

## Training
* `train_pose.py` runs a simple interleaved training test - no question input, just interleaved generation.
* `train.py` runs a full single task question prepended interleaved run. (TODO)

## Running inference
* Scripts under `inference/`, use `inference/decode_{pose/audio}.py` to convert generated samples in `inference_outputs` to audio and pose video (normalised and centered).
* Use correct pose tokenizer, by default we are working with `pose_tokenizer_xabi/`.
* Use `inference/comparisons.py` to plot comparison plots.
* Run `bash full.sh` to simultaneously generate outputs and plot them against references.
* note - inference scripts read config from `config.yaml` - make sure the settings we are using correspond to the correct weights, eg. the value of k-shifting and pose prepending

Note that each new infernce run wipes the outputs folder and re-writes its outputs to it.