from torch.utils.data import DistributedSampler


class LengthBudgetBatchSampler(DistributedSampler):
    """Greedy length-packed distributed batch sampler.

    Each rank walks its DistributedSampler index stream and accumulates samples
    into a batch until the cumulative `length` reaches `target_frames`, then
    yields the batch. This keeps per-rank token counts roughly equal across
    ranks so FSDP barriers don't wait on a single straggler.

    The dataset must expose a `length` column.
    """

    def __init__(
        self,
        dataset,
        target_frames: int,
        *,
        world_size: int = 1,
        rank: int = 0,
        shuffle: bool = False,
        seed: int = 0,
    ):
        super().__init__(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        self.target_frames = target_frames
        # cheap per-row length lookup; avoids materialising audio/pose token lists
        self.length_dataset = dataset.select_columns(["length"])

    def __len__(self):
        # estimated batch count per rank: per-rank token total / target_frames.
        # HF Trainer uses this for max_steps and the progress-bar ETA. Without
        # the override we'd inherit DistributedSampler's sample count, which
        # over-counts steps by a factor of avg_pack_size and inflates the ETA.
        total_length = sum(self.length_dataset["length"])
        per_rank_length = total_length // self.num_replicas
        return max(per_rank_length // self.target_frames, 1)

    def _length(self, idx):
        return min(int(self.length_dataset[idx]["length"]), self.target_frames)

    def __iter__(self):
        batch, total = [], 0
        for idx in super().__iter__():
            idx = int(idx)
            batch.append(idx)
            total += self._length(idx)
            if total >= self.target_frames:
                yield batch
                batch, total = [], 0
        if batch:
            yield batch
