"""
Cleaned-input rerun of the two diameter winners (answers the review point that 14 train / 4 val / 4 test Phase 0 pairs have a black/placeholder latest input).

For each variant and seed, the fixed sweep-winner configuration is retrained from scratch and scored once on test:
  original  pairs_split.parquet as used for all reported diameter results
  clean     the same pairs with every black/placeholder frame (--exclude-list, as in the image track) removed from the input sequences; days / dates / n_input_frames /
            last_input_* are recomputed, a pair whose inputs were ALL black is dropped. Targets, plant split and norm stats are unchanged.
No hyper-parameter search; validation MAE selects the epoch exactly as in the sweeps; test is scored once per run. Seeds 42-44 give a run-to-run reference.
Outputs: outputs/diameter_clean_rerun_results.csv (one row per variant/model/seed) and outputs/diameter_clean_rerun_per_pair.csv.

Usage:
    python src/diameter_clean_rerun.py --pairs-split data/pairs_split.parquet --embeddings-dir data/embeddings --norm-stats data/norm_stats.json \
        --exclude-list outputs/tiny_images_md5.txt --out-dir outputs --checkpoint-dir checkpoints --device cuda
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import cnn_lstm_train_core as L
import cnn_transformer_train_core as T
from cnn_lstm_dataset import collate_fn as lstm_collate
from cnn_lstm_model import CNNLSTM
from cnn_transformer_dataset import collate_fn as tfm_collate
from cnn_transformer_model import CNNTransformer

LSTM_CFG = {"hidden_dim": 256, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "batch_size": 64, "max_epochs": 200, "patience": 15}
TFM_CFG = {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.2, "lr": 3e-4, "dim_feedforward": 128, "batch_size": 64, "max_epochs": 200, "patience": 15}


def clean_pairs(pairs, bad):
    keep_rows, n_changed = [], 0
    for r in pairs.to_dict("records"):
        fps = list(r["input_filepaths"])
        ok = [i for i, f in enumerate(fps) if f not in bad]
        if not ok:
            continue
        if len(ok) != len(fps):
            n_changed += 1
            for k in ("input_filepaths", "input_days_after_planting", "input_acquisition_dates"):
                r[k] = np.asarray(list(r[k]))[ok]
            r["n_input_frames"] = len(ok)
            r["last_input_day"] = int(r["input_days_after_planting"][-1])
            r["last_input_date"] = r["input_acquisition_dates"][-1]
            r["forecast_gap_days"] = int(r["target_day"]) - r["last_input_day"]
        keep_rows.append(r)
    out = pd.DataFrame(keep_rows)
    print(f"Clean variant: {len(pairs)} pairs -> {len(out)} kept ({n_changed} had black frames removed from inputs; {len(pairs) - len(out)} dropped, all inputs black)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--exclude-list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    ns = json.load(open(args.norm_stats))
    tm, ts = ns["target_diameter_mean"], ns["target_diameter_std"]

    pairs = pd.read_parquet(args.pairs_split)
    pairs["pair_id"] = pairs["plant_id"].astype(str) + "::" + pairs["target_day"].astype(str)
    bad = set(pd.read_csv(args.exclude_list, sep=r"\s+", header=None, names=["md5", "fp"])["fp"])
    allfp = set(f for l in pairs["input_filepaths"] for f in l)
    assert len(bad & allfp) > 0, "exclude-list paths do not match input_filepaths"
    last_bad = pairs["input_filepaths"].apply(lambda l: l[-1] in bad)
    print("Phase 0 pairs with a black LATEST input:", pairs[last_bad]["split"].value_counts().to_dict(), flush=True)
    variants = {"original": pairs, "clean": clean_pairs(pairs, bad)}
    affected = set(pairs.loc[pairs["input_filepaths"].apply(lambda l: any(f in bad for f in l)), "pair_id"])
    if args.smoke:
        args.seeds = args.seeds[:1]
        LSTM_CFG["max_epochs"] = TFM_CFG["max_epochs"] = 2

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, "clean_rerun"), exist_ok=True)
    rows, per_pair = [], []
    for vname, vdf in variants.items():
        for mname, core, cfg, coll in (("cnn_lstm", L, LSTM_CFG, lstm_collate), ("cnn_transformer", T, TFM_CFG, tfm_collate)):
            train_ds, val_ds, test_ds, _, _, test_df = core.build_datasets(vdf, args.embeddings_dir, ns)
            for seed in args.seeds:
                ck = os.path.join(args.checkpoint_dir, "clean_rerun", f"{vname}_{mname}_s{seed}.pt")
                vmae, vrmse, hist, _ = core.train_one_config(train_ds, val_ds, cfg, device, ck, tm, ts, seed=seed, verbose=False)
                if mname == "cnn_lstm":
                    model = CNNLSTM(input_dim=513, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(device)
                else:
                    model = CNNTransformer(input_dim=512, d_model=cfg["d_model"], nhead=cfg["nhead"], num_layers=cfg["num_layers"],
                                           dim_feedforward=cfg["dim_feedforward"], dropout=cfg["dropout"]).to(device)
                model.load_state_dict(torch.load(ck, weights_only=True))
                tp = core.predict(model, DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=coll), device, tm, ts)
                tp["abs_error"] = (tp["y_pred"] - tp["y_true"]).abs()
                mae, rmse = core.mae_rmse(tp["y_true"], tp["y_pred"])
                aff = tp[tp["pair_id"].isin(affected)]
                rows.append({"variant": vname, "model": mname, "seed": seed, "val_mae": vmae, "epochs": len(hist), "test_n": len(tp), "test_mae": mae, "test_rmse": rmse,
                             "affected_n": len(aff), "affected_mae": aff["abs_error"].mean() if len(aff) else np.nan})
                per_pair.append(tp.assign(variant=vname, model=mname, seed=seed))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "diameter_clean_rerun_results.csv"), index=False)
    pd.concat(per_pair).to_csv(os.path.join(args.out_dir, "diameter_clean_rerun_per_pair.csv"), index=False)
    res = pd.DataFrame(rows)
    print(res.groupby(["variant", "model"])[["val_mae", "test_mae", "test_rmse", "affected_mae"]].agg(["mean", "std"]).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
