"""
Step 6: Dataset and collation for the CNN-LSTM.

Each sample's input is the ordered sequence of cached ResNet18 embeddings
(512-dim) for a pair's input_filepaths, concatenated at each timestep with
that step's normalized day_after_planting (1-dim) -- so per-timestep
feature dim is 513. Sequences are variable length (1-14 frames per Step 3's
distribution); collate_fn pads to the batch max and returns true lengths
for pack_padded_sequence.

Embeddings are loaded from the same sharded cache Step 4 produced and the
same embedding_index.parquet Step 5 used, so no image ever needs to be
re-read.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CNNLSTMDataset(Dataset):
    def __init__(self, pairs_df, embeddings_dir, day_mean, day_std, target_mean, target_std):
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.embeddings_dir = embeddings_dir
        self.day_mean = day_mean
        self.day_std = day_std
        self.target_mean = target_mean
        self.target_std = target_std

        index_df = pd.read_parquet(os.path.join(embeddings_dir, "embedding_index.parquet"))
        self.fp_to_loc = index_df.set_index("image_id")[["shard_file", "offset"]]
        self._shard_cache = {}

    def __len__(self):
        return len(self.pairs_df)

    def _load_embedding(self, filepath):
        shard_file, offset = self.fp_to_loc.loc[filepath, ["shard_file", "offset"]]
        if shard_file not in self._shard_cache:
            self._shard_cache[shard_file] = torch.load(
                os.path.join(self.embeddings_dir, shard_file), weights_only=True
            )
        shard = self._shard_cache[shard_file]
        return shard["embeddings"][offset]  # (512,)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        filepaths = row["input_filepaths"]
        days = np.asarray(row["input_days_after_planting"], dtype=np.float32)
        days_norm = (days - self.day_mean) / self.day_std

        embs = torch.stack([self._load_embedding(fp) for fp in filepaths])  # (T, 512)
        days_t = torch.from_numpy(days_norm).unsqueeze(1)  # (T, 1)
        features = torch.cat([embs, days_t.float()], dim=1)  # (T, 513)

        target_norm = (row["target_diameter"] - self.target_mean) / self.target_std

        return {
            "features": features,
            "length": features.shape[0],
            "target_norm": torch.tensor(target_norm, dtype=torch.float32),
            "target_raw": torch.tensor(row["target_diameter"], dtype=torch.float32),
            "pair_id": row["pair_id"],
            "plant_id": row["plant_id"],
        }


def collate_fn(batch):
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_len = lengths.max().item()
    feat_dim = batch[0]["features"].shape[1]

    padded = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    for i, b in enumerate(batch):
        padded[i, :b["length"]] = b["features"]

    targets_norm = torch.stack([b["target_norm"] for b in batch])
    targets_raw = torch.stack([b["target_raw"] for b in batch])
    pair_ids = [b["pair_id"] for b in batch]
    plant_ids = [b["plant_id"] for b in batch]

    return {
        "features": padded,
        "lengths": lengths,
        "target_norm": targets_norm,
        "target_raw": targets_raw,
        "pair_id": pair_ids,
        "plant_id": plant_ids,
    }
