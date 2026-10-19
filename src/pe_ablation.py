"""
Positional-encoding ablation for the diameter CNN-Transformer (answers the review point that src/check_pe_dmodel_sensitivity.py scales d_model, heads and feed-forward width
together and so cannot isolate the encoding). Architecture and training are FIXED at the sweep winner (d_model 64, 4 heads, 2 layers, feed-forward 128, dropout 0.2, lr 3e-4);
only how observation time enters changes:
  sinusoid  continuous-day sinusoidal encoding (the original)
  linear    a learned Linear(day/100) added to every token
  none      no timing information at all (tokens in order only)
3 seeds per mode, epoch chosen on validation MAE, test scored once per run (N = 1,060).
Usage: python src/pe_ablation.py --pairs-split data/pairs_split.parquet --embeddings-dir data/embeddings --norm-stats data/norm_stats.json --out-dir outputs --checkpoint-dir checkpoints --device cuda
"""
import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

import cnn_transformer_train_core as T
from cnn_transformer_dataset import collate_fn
from cnn_transformer_model import CNNTransformer

CFG = {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.2, "lr": 3e-4, "dim_feedforward": 128, "batch_size": 64, "max_epochs": 200, "patience": 15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()
    device = torch.device(a.device)
    ns = json.load(open(a.norm_stats))
    tm, ts = ns["target_diameter_mean"], ns["target_diameter_std"]
    pairs = pd.read_parquet(a.pairs_split)
    pairs["pair_id"] = T.make_pair_id(pairs)
    train_ds, val_ds, test_ds, *_ = T.build_datasets(pairs, a.embeddings_dir, ns)
    os.makedirs(os.path.join(a.checkpoint_dir, "pe_ablation"), exist_ok=True)
    rows = []
    for mode in ("sinusoid", "linear", "none"):
        for seed in a.seeds:
            ck = os.path.join(a.checkpoint_dir, "pe_ablation", f"{mode}_s{seed}.pt")
            vmae, _, hist, _ = T.train_one_config(train_ds, val_ds, {**CFG, "pe_mode": mode}, device, ck, tm, ts, seed=seed, verbose=False)
            m = CNNTransformer(input_dim=512, d_model=CFG["d_model"], nhead=CFG["nhead"], num_layers=CFG["num_layers"], dim_feedforward=CFG["dim_feedforward"],
                               dropout=CFG["dropout"], pe_mode=mode).to(device)
            m.load_state_dict(torch.load(ck, weights_only=True))
            tp = T.predict(m, DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn), device, tm, ts)
            mae, rmse = T.mae_rmse(tp["y_true"], tp["y_pred"])
            rows.append({"pe_mode": mode, "seed": seed, "val_mae": vmae, "epochs": len(hist), "test_n": len(tp), "test_mae": mae, "test_rmse": rmse})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "pe_ablation_results.csv"), index=False)
    r = pd.DataFrame(rows)
    print(r.groupby("pe_mode")[["val_mae", "test_mae", "test_rmse"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
