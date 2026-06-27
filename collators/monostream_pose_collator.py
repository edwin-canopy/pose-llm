
import torch


class PoseSpeechMonoCollator:
    """
    data format:
    words = [{}] with {word, start time, end time}
    pose_tokens = Tensor (N, T)   # pre-upsampled to audio rate (~12.5 fps)
    audio_tokens = Tensor (N, T)
    """

    def __init__(
        self,
        text_tokenizer,
        config,
    ):
        self.text_tokenizer = text_tokenizer

        # special tokens
        special = config["special_tokens"]
        self.word_pad = special["word_pad"]
        self.new_word = special["new_word"]
        self.separator = special["separator"]
        self.audio_tokens_start = special["audio_tokens_start"]
        self.pose_tokens_start = special["pose_tokens_start"]
        self.pose_padding_token = special["pose_padding_token"]

        self.pose_depth = config["pose_depth_model"]["residual_depth"]
        self.pose_codebook_size = config["pose_depth_model"]["codebook_size"]
        self.audio_depth = config["audio_depth_model"]["residual_depth"]
        self.audio_codebook_size = config["audio_depth_model"]["codebook_size"]
        self.frame_duration = config["audio_depth_model"]["frame_duration"]
        self.max_sequence_length = config["training"]["max_sequence_length"]


    def mark_word_starts(self, frames):
        for pos in range(1, len(frames)):
            if frames[pos] not in (self.word_pad, self.new_word) and frames[pos - 1] == self.word_pad:
                frames[pos - 1] = self.new_word
        return frames


    def shift_left(self, frames):
        for pos in range(1, len(frames)):
            if frames[pos] != self.word_pad and frames[pos - 1] == self.word_pad:
                frames[pos - 1] = frames[pos]
                frames[pos] = self.word_pad
        return frames


    def assemble(self, sample):
        """
        sample:
        - text {word, start, end, tokens (added)}
        - pose_tokens shape (Np, T)
        - audio_tokens shape (Na, T)
        """

        # 1. tokenize text
        # note: maybe faster to do this once in dataset creation

        text = sample["text"]
        assert isinstance(text[0], dict)
        text[0]["tokens"] = self.text_tokenizer.encode(text[0]["word"], add_special_tokens=False)
        for i in range(1, len(text)):
            text[i]["tokens"] = self.text_tokenizer.encode(" " + text[i]["word"], add_special_tokens=False)

        # 1b. initialise tensors with pad everywhere

        pose_tokens = sample["pose_tokens"]
        audio_tokens = sample["audio_tokens"]
        # both streams should be at audio rate; tolerate at most one frame of
        # drift (source tokenizer rounding) by truncating to the shorter.
        T_audio = audio_tokens.shape[1]
        T_pose = pose_tokens.shape[1]
        assert abs(T_audio - T_pose) <= 1, (
            f"audio/pose frame counts disagree by more than one: "
            f"audio={T_audio} pose={T_pose}"
        )
        if T_audio != T_pose:
            pass
            # print(
            #     f"warning: off-by-one audio/pose mismatch (audio={T_audio} "
            #     f"pose={T_pose}); truncating to shorter",
            #     flush=True,
            # )
        num_frames = min(T_audio, T_pose)
        audio_tokens = audio_tokens[:, :num_frames]
        pose_tokens = pose_tokens[:, :num_frames]
        frames = [self.word_pad] * num_frames

        # 2. distribute tokens so that we have one word token per frame

        write_pos = -1
        for w in text:
            pos = max(round(w["start"] / self.frame_duration), write_pos + 1)
            for tok in w["tokens"]:
                if pos >= num_frames:
                    break
                frames[pos] = tok
                write_pos = pos
                pos += 1

        # 2b. shift text to give lookahead (where possible)

        frames = self.shift_left(frames)

        # 2c. add new word tokens before subsequent new words

        frames = self.mark_word_starts(frames)

        # 3. stack codebook tokens with text tokens

        sequence_ids = torch.empty((num_frames, 3), dtype=torch.long)
        sequence_ids[:, 0] = torch.tensor(frames, dtype=torch.long)

        # 4. add audio/pose offsets so zeroth-codebook tokens fall in their reserved ranges

        sequence_ids[:, 1] = audio_tokens[0] + self.audio_tokens_start
        sequence_ids[:, 2] = pose_tokens[0] + self.pose_tokens_start

        pose_code_columns = [
            pose_tokens[d] + d * self.pose_codebook_size for d in range(self.pose_depth)
        ]
        pose_codes = torch.stack(pose_code_columns, dim=1)

        audio_code_columns = [
            audio_tokens[d] + d * self.audio_codebook_size for d in range(self.audio_depth)
        ]
        audio_codes = torch.stack(audio_code_columns, dim=1)

        return {
            "ids": sequence_ids,
            "audio_codes": audio_codes,
            "pose_codes": pose_codes,
        }


    def __call__(self, samples):
        """
        Takes in a list of samples
        Collates samples into a single sample separated by special tokens.

        Returns dict of
        - backbone_ids: Tensor (1, S, 3) # text + zeroth audio + zeroth pose
        - audio_depth_ids: Tensor (1, S, audio_depth)
        - pose_depth_ids: Tensor (1, S, pose_depth)
        - separator_mask: Tensor (1, S)
        """
        assert isinstance(samples, list)

        # call assmble and concatenate with separators
        sequences = [self.assemble(sample) for sample in samples]

        sequence_ids = [s["ids"] for s in sequences]
        audio_codes = [s["audio_codes"] for s in sequences]
        pose_codes = [s["pose_codes"] for s in sequences]
        masks = [torch.zeros((s.shape[0],), dtype=torch.bool) for s in sequence_ids]

        # interleave separator frames between consecutive samples
        sep_backbone = torch.full((1, 3), self.separator, dtype=torch.long)
        sep_audio = torch.full((1, self.audio_depth), -1, dtype=torch.long)
        sep_pose = torch.full((1, self.pose_depth), -1, dtype=torch.long)
        sep_mask = torch.ones((1,), dtype=torch.bool)
        for i in range(len(sequences) - 1, 0, -1):
            sequence_ids.insert(i, sep_backbone)
            audio_codes.insert(i, sep_audio)
            pose_codes.insert(i, sep_pose)
            masks.insert(i, sep_mask)

        backbone_ids = torch.cat(sequence_ids, dim=0).unsqueeze(0)
        audio_depth_ids = torch.cat(audio_codes, dim=0).unsqueeze(0)
        pose_depth_ids = torch.cat(pose_codes, dim=0).unsqueeze(0)
        separator_mask = torch.cat(masks, dim=0).unsqueeze(0)

        # hard cap on per-step sequence length (post-concat)
        backbone_ids = backbone_ids[:, : self.max_sequence_length]
        audio_depth_ids = audio_depth_ids[:, : self.max_sequence_length]
        pose_depth_ids = pose_depth_ids[:, : self.max_sequence_length]
        separator_mask = separator_mask[:, : self.max_sequence_length]

        return {
            "backbone_ids": backbone_ids,
            "audio_depth_ids": audio_depth_ids,
            "pose_depth_ids": pose_depth_ids,
            "separator_mask": separator_mask,
        }


if __name__ == "__main__":
    # quick sanity check: load a few rows of the preprocessed dataset, run
    # __call__, and print output shapes / dtypes.
    import yaml
    from datasets import load_dataset
    from transformers import AutoTokenizer

    DATA_DIR = "/mnt/somfs/pose_cond/merged_pose_audio_dataset"
    N_SAMPLES = 2
    POSE_CODEBOOKS = 8  # matches NUM_POSE_CODEBOOKS in preprocess_data.py

    config = yaml.safe_load(open("../config.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(config["large_model"])

    ds = load_dataset(
        "parquet", data_files=f"{DATA_DIR}/*.parquet", split="train"
    )
    print(f"loaded dataset: {len(ds):,} rows, columns={ds.column_names}")

    def to_tensors(row):
        # pose_tokens is a flat frame-major list (f0_cb0..f0_cb7, f1_cb0..),
        # reshape to (T', 8) then transpose -> (8, T') as the collator expects.
        pose_flat = torch.tensor(row["pose_tokens"], dtype=torch.long)
        pose = pose_flat.view(-1, POSE_CODEBOOKS).T.contiguous()
        audio = torch.tensor(row["audio_tokens"], dtype=torch.long)  # (8, T)
        return {"text": row["text"], "audio_tokens": audio, "pose_tokens": pose}

    samples = [to_tensors(ds[i]) for i in range(N_SAMPLES)]
    for i, s in enumerate(samples):
        print(
            f"  sample {i}: text words={len(s['text'])}, "
            f"audio={tuple(s['audio_tokens'].shape)}, "
            f"pose={tuple(s['pose_tokens'].shape)}"
        )

    collator = PoseSpeechMonoCollator(tokenizer, config)
    out = collator(samples)

    print("collator output:")
    for k, v in out.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

