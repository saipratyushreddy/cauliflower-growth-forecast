"""
Phase 2: Dataset and collation for the CNN-Transformer.

Unlike the CNN-LSTM dataset (cnn_lstm_dataset.py), embeddings and
day_after_planting are kept as SEPARATE tensors rather than concatenated
into one feature vector -- the Transformer projects the 512-dim embedding
into d_model and adds a positional encoding DERIVED from the raw day
value, rather than treating day-after-planting as just another input
feature. Days are returned in RAW (unnormalized) units for this reason
(see cnn_transformer_model.py's continuous_sinusoidal_pe docstring for
why raw units matter for the encoding's frequency spectrum).

Reuses the same cached embedding shards/index Step 4 produced -- no image
is ever re-read.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CNNTransformerDataset(Dataset):
    def __init__(self, pairs_df, embeddings_dir, target_mean, target_std):
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.embeddings_dir = embeddings_dir
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
        days_raw = np.asarray(row["input_days_after_planting"], dtype=np.float32)

        embs = torch.stack([self._load_embedding(fp) for fp in filepaths])  # (T, 512)
        days_t = torch.from_numpy(days_raw)  # (T,) RAW day values, unnormalized

        target_norm = (row["target_diameter"] - self.target_mean) / self.target_std

        return {
            "embeddings": embs,
            "days_raw": days_t,
            "length": embs.shape[0],
            "target_norm": torch.tensor(target_norm, dtype=torch.float32),
            "target_raw": torch.tensor(row["target_diameter"], dtype=torch.float32),
            "pair_id": row["pair_id"],
            "plant_id": row["plant_id"],
        }


def collate_fn(batch):
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_len = lengths.max().item()
    emb_dim = batch[0]["embeddings"].shape[1]

    padded_embs = torch.zeros(len(batch), max_len, emb_dim, dtype=torch.float32)
    padded_days = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, b in enumerate(batch):
        padded_embs[i, :b["length"]] = b["embeddings"]
        padded_days[i, :b["length"]] = b["days_raw"]

    targets_norm = torch.stack([b["target_norm"] for b in batch])
    targets_raw = torch.stack([b["target_raw"] for b in batch])
    pair_ids = [b["pair_id"] for b in batch]
    plant_ids = [b["plant_id"] for b in batch]

    return {
        "embeddings": padded_embs,
        "days_raw": padded_days,
        "lengths": lengths,
        "target_norm": targets_norm,
        "target_raw": targets_raw,
        "pair_id": pair_ids,
        "plant_id": plant_ids,
    }
