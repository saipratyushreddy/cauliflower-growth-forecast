"""
Target-day (forecast-gap) input for the diameter CNN-LSTM.

DESIGN (written down before any run; see docs/01 "Target-day input"):
  * baseline arm  = the current CNN-LSTM sweep winner (hidden 256, 2 layers, dropout 0.2, lr 1e-3), input = [ResNet18 embedding, day of the photo].
  * target-day arm = the same, plus ONE extra input per timestep: (target_day - photo_day), z-scored with the TRAIN per-timestep mean/SD. 514 inputs instead of 513.
  * Everything else is identical: split, batch size, 200 epochs max, patience 15, epoch chosen on validation MAE, test scored once per run, no new hyper-parameter search.
  * Both arms are retrained here (seeds 42, 43, 44) so the comparison is like for like; neither arm receives any earlier diameter measurement.
  * Analysis: paired per-example MAE difference (target-day minus baseline, averaged over the 3 seeds per arm), overall and by field x forecast-gap bin, with sample counts and
    plant-level bootstrap 95% intervals. Forecast gaps are 2-9 days in Field2 and up to 40 in Field1, so horizon bins are largely a field split; long gaps (>= 29 days) are Field1 only.
Outputs: outputs/target_day_results.csv (per run), outputs/target_day_per_pair.csv (per run x example), outputs/target_day_paired.csv (paired summary).

Usage:
    python src/target_day_experiment.py --pairs-split data/pairs_split.parquet --embeddings-dir data/embeddings --norm-stats data/norm_stats.json \
        --out-dir outputs --checkpoint-dir checkpoints --device cuda
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import cnn_lstm_train_core as L
from cnn_lstm_dataset import CNNLSTMDataset, collate_fn
from cnn_lstm_model import CNNLSTM

CFG = {"hidden_dim": 256, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "max_epochs": 200, "patience": 15}
BINS = [(0, 7, "<=7 d"), (8, 14, "8-14 d"), (29, 100, ">=29 d")]


class GapDataset(CNNLSTMDataset):
    """CNNLSTMDataset + one extra feature per timestep: z-scored (target_day - photo_day)."""

    def __init__(self, *a, gap_mean, gap_std, **k):
        super().__init__(*a, **k)
        self.gap_mean, self.gap_std = gap_mean, gap_std

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        row = self.pairs_df.iloc[idx]
        gaps = (float(row["target_day"]) - np.asarray(row["input_days_after_planting"], dtype=np.float32) - self.gap_mean) / self.gap_std
        item["features"] = torch.cat([item["features"], torch.from_numpy(gaps.astype(np.float32)).unsqueeze(1)], dim=1)
        return item


def boot_ci(diff, plants, n=2000, seed=0):
    d = pd.DataFrame({"d": diff, "p": plants}).groupby("p")["d"].agg(["sum", "count"])
    i = np.random.RandomState(seed).randint(0, len(d), (n, len(d)))
    m = d["sum"].values[i].sum(1) / d["count"].values[i].sum(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = torch.device(a.device)
    ns = json.load(open(a.norm_stats))
    tm, ts = ns["target_diameter_mean"], ns["target_diameter_std"]
    pairs = pd.read_parquet(a.pairs_split)
    pairs["pair_id"] = L.make_pair_id(pairs)
    tr = pairs[pairs.split == "train"]
    allgaps = np.concatenate([float(r.target_day) - np.asarray(r.input_days_after_planting, dtype=np.float64) for r in tr.itertuples()])
    gm, gs = float(allgaps.mean()), float(allgaps.std())
    print(f"per-timestep gap (train): mean {gm:.2f} sd {gs:.2f}", flush=True)
    kw = dict(embeddings_dir=a.embeddings_dir, day_mean=ns["day_after_planting_mean"], day_std=ns["day_after_planting_std"], target_mean=tm, target_std=ts)
    sets = {s: pairs[pairs.split == s] for s in ("train", "val", "test")}
    arms = {"baseline": {s: CNNLSTMDataset(d, **kw) for s, d in sets.items()},
            "target_day": {s: GapDataset(d, gap_mean=gm, gap_std=gs, **kw) for s, d in sets.items()}}
    seeds, cfg0 = a.seeds, dict(CFG)
    if a.smoke:
        seeds, cfg0["max_epochs"] = seeds[:1], 2
    os.makedirs(os.path.join(a.checkpoint_dir, "target_day"), exist_ok=True)
    rows, per_pair = [], []
    for arm, ds in arms.items():
        cfg = {**cfg0, "input_dim": 514 if arm == "target_day" else 513}
        for seed in seeds:
            ck = os.path.join(a.checkpoint_dir, "target_day", f"{arm}_s{seed}.pt")
            vmae, _, hist, _ = L.train_one_config(ds["train"], ds["val"], cfg, device, ck, tm, ts, seed=seed, verbose=False)
            m = CNNLSTM(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
            m.load_state_dict(torch.load(ck, weights_only=True))
            tp = L.predict(m, DataLoader(ds["test"], batch_size=64, shuffle=False, collate_fn=collate_fn), device, tm, ts)
            mae, rmse = L.mae_rmse(tp["y_true"], tp["y_pred"])
            rows.append({"arm": arm, "seed": seed, "val_mae": vmae, "epochs": len(hist), "test_n": len(tp), "test_mae": mae, "test_rmse": rmse})
            per_pair.append(tp.assign(arm=arm, seed=seed))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "target_day_results.csv"), index=False)
    pp = pd.concat(per_pair)
    pp["abs_error"] = (pp["y_pred"] - pp["y_true"]).abs()
    pp.to_csv(os.path.join(a.out_dir, "target_day_per_pair.csv"), index=False)

    meta = sets["test"].assign(gap=lambda d: d["target_day"] - d["last_input_day"])[["pair_id", "plant_id", "field", "gap"]].set_index("pair_id")
    e = pp.groupby(["pair_id", "arm"])["abs_error"].mean().unstack()           # seed-averaged abs error per example
    e = e.join(meta)
    e["diff"] = e["target_day"] - e["baseline"]                                # negative = target-day input helps
    out = []
    def add(label, d):
        lo, hi = boot_ci(d["diff"].values, d["plant_id"].values) if len(d) > 1 else (np.nan, np.nan)
        out.append({"subset": label, "n": len(d), "n_plants": d.plant_id.nunique(), "baseline_mae": d.baseline.mean(), "target_day_mae": d.target_day.mean(),
                    "paired_diff": d["diff"].mean(), "ci_lo": lo, "ci_hi": hi, "share_improved": (d["diff"] < 0).mean()})
    add("all", e)
    for f in sorted(e.field.unique()):
        add(f"{f}", e[e.field == f])
        for lo_, hi_, nm in BINS:
            s = e[(e.field == f) & (e.gap >= lo_) & (e.gap <= hi_)]
            if len(s):
                add(f"{f}, gap {nm}", s)
    paired = pd.DataFrame(out)
    paired.to_csv(os.path.join(a.out_dir, "target_day_paired.csv"), index=False)
    r = pd.DataFrame(rows)
    print(r.groupby("arm")[["val_mae", "test_mae", "test_rmse"]].agg(["mean", "std"]).round(3).to_string())
    print(paired.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
