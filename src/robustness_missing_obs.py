"""
Phase 2: Missing-observation robustness test for CNN-LSTM vs CNN-Transformer.

INFERENCE-TIME EVALUATION ONLY. No retraining, no hyperparameter changes.
Loads the two already-trained, frozen sweep-winner checkpoints exactly as
saved and runs inference on synthetically shortened input sequences.

Protocol (as specified):
  - For each test-set pair in the shared evaluation set (same pair_ids
    both frozen checkpoints can already predict, per the existing
    shared-eval-set convention), at each drop level in {0%, 25%, 50%,
    75%} of that pair's available input observations (never the target
    itself, which is not in the input by construction):
      - Randomly remove that fraction of the plant's PRE-TARGET
        observations, keep the remaining ones in original chronological
        order, never drop below 1 remaining observation.
      - Run inference with the frozen model exactly as-is.
  - 5 different random seeds per drop level, report mean +/- std MAE/RMSE.
  - CNN-LSTM and CNN-Transformer see the EXACT SAME dropped-observation
    sets per trial (same seed -> same indices kept, applied identically
    to both models' own input representations of that pair), so the
    comparison is apples-to-apples.

Usage:
    python src/robustness_missing_obs.py \
        --pairs-split data/pairs_split.parquet \
        --embeddings-dir data/embeddings \
        --norm-stats data/norm_stats.json \
        --lstm-checkpoint checkpoints/sweep/h256_lr0.001_L2_d0.2.pt \
        --lstm-hidden-dim 256 --lstm-num-layers 2 --lstm-dropout 0.2 \
        --transformer-checkpoint checkpoints/transformer_sweep/d64_h4_lr0.0003_L2_do0.2.pt \
        --transformer-d-model 64 --transformer-nhead 4 --transformer-num-layers 2 \
        --transformer-dropout 0.2 \
        --lstm-test-predictions outputs/step6_cnn_lstm_sweep_winner_test_predictions.csv \
        --transformer-test-predictions outputs/step_p2_cnn_transformer_test_predictions.csv \
        --out-dir outputs \
        --device cuda
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from cnn_lstm_model import CNNLSTM
from cnn_transformer_model import CNNTransformer

DROP_LEVELS = [0.0, 0.25, 0.5, 0.75]
N_SEEDS = 5


def mae_rmse(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def make_pair_id(df):
    return df["plant_id"].astype(str) + "::" + df["target_day"].astype(str)


def load_embedding_index(embeddings_dir):
    index_df = pd.read_parquet(os.path.join(embeddings_dir, "embedding_index.parquet"))
    return index_df.set_index("image_id")[["shard_file", "offset"]]


def load_all_pair_embeddings(pairs_df, embeddings_dir, fp_to_loc):
    """
    Preload every test pair's full embedding sequence once (as a plain
    dict pair_id -> (embeddings tensor (T,512), days_raw array (T,))),
    so dropping is just index selection at eval time, not re-reading
    shards per trial.
    """
    shard_cache = {}
    data = {}
    for _, row in pairs_df.iterrows():
        filepaths = row["input_filepaths"]
        days_raw = np.asarray(row["input_days_after_planting"], dtype=np.float32)
        embs = []
        for fp in filepaths:
            shard_file, offset = fp_to_loc.loc[fp, ["shard_file", "offset"]]
            if shard_file not in shard_cache:
                shard_cache[shard_file] = torch.load(
                    os.path.join(embeddings_dir, shard_file), weights_only=True
                )
            embs.append(shard_cache[shard_file]["embeddings"][offset])
        embs = torch.stack(embs)  # (T, 512)
        data[row["pair_id"]] = {
            "embeddings": embs,
            "days_raw": days_raw,
            "target_diameter": row["target_diameter"],
        }
    return data


def drop_indices(n_total, drop_frac, rng):
    """
    Randomly select which of n_total chronologically-ordered observations
    to KEEP after dropping drop_frac of them, never dropping below 1
    remaining. Returns sorted indices to keep (preserves original order).
    """
    n_drop = int(round(n_total * drop_frac))
    n_drop = min(n_drop, n_total - 1)  # never drop below 1 remaining
    if n_drop <= 0:
        return list(range(n_total))
    drop_idx = set(rng.choice(n_total, size=n_drop, replace=False).tolist())
    keep_idx = [i for i in range(n_total) if i not in drop_idx]
    return keep_idx


@torch.no_grad()
def predict_lstm(model, embeddings, days_raw, day_mean, day_std, target_mean, target_std, device):
    days_norm = (days_raw - day_mean) / day_std
    days_t = torch.from_numpy(days_norm).unsqueeze(1).float()  # (T,1)
    features = torch.cat([embeddings, days_t], dim=1).unsqueeze(0).to(device)  # (1,T,513)
    lengths = torch.tensor([features.shape[1]], dtype=torch.long)
    pred_norm = model(features, lengths).cpu().item()
    return pred_norm * target_std + target_mean


@torch.no_grad()
def predict_transformer(model, embeddings, days_raw, target_mean, target_std, device):
    embeddings = embeddings.unsqueeze(0).to(device)  # (1,T,512)
    days_t = torch.from_numpy(days_raw).unsqueeze(0).to(device)  # (1,T)
    lengths = torch.tensor([embeddings.shape[1]], dtype=torch.long)
    pred_norm = model(embeddings, days_t, lengths).cpu().item()
    return pred_norm * target_std + target_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--lstm-checkpoint", required=True)
    ap.add_argument("--lstm-hidden-dim", type=int, required=True)
    ap.add_argument("--lstm-num-layers", type=int, required=True)
    ap.add_argument("--lstm-dropout", type=float, default=0.0)
    ap.add_argument("--transformer-checkpoint", required=True)
    ap.add_argument("--transformer-d-model", type=int, required=True)
    ap.add_argument("--transformer-nhead", type=int, required=True)
    ap.add_argument("--transformer-num-layers", type=int, required=True)
    ap.add_argument("--transformer-dropout", type=float, default=0.0)
    ap.add_argument("--lstm-test-predictions", required=True,
                     help="Step 6 sweep winner's test predictions CSV, to define the shared eval pair_id set")
    ap.add_argument("--transformer-test-predictions", required=True,
                     help="Phase 2 transformer sweep winner's test predictions CSV, same purpose")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    with open(args.norm_stats) as f:
        norm_stats = json.load(f)
    target_mean = norm_stats["target_diameter_mean"]
    target_std = norm_stats["target_diameter_std"]
    day_mean = norm_stats["day_after_planting_mean"]
    day_std = norm_stats["day_after_planting_std"]

    pairs_df = pd.read_parquet(args.pairs_split)
    pairs_df["pair_id"] = make_pair_id(pairs_df)
    test_df = pairs_df[pairs_df["split"] == "test"].copy()

    # Restrict to the SAME shared-eval-set pair_ids already used for every
    # other cross-method comparison in this project, so this robustness
    # test is on the identical population, not a new ad hoc subset.
    lstm_preds = pd.read_csv(args.lstm_test_predictions)
    transformer_preds = pd.read_csv(args.transformer_test_predictions)
    shared_ids = set(lstm_preds["pair_id"]) & set(transformer_preds["pair_id"]) & set(test_df["pair_id"])
    test_df = test_df[test_df["pair_id"].isin(shared_ids)].copy()
    print(f"Evaluating on {len(test_df)} shared-eval-set test pairs "
          f"(intersection of LSTM's, Transformer's, and pairs_split's test pair_ids)")

    # --- Load both FROZEN checkpoints, exactly as saved. No retraining. ---
    lstm_model = CNNLSTM(
        input_dim=513, hidden_dim=args.lstm_hidden_dim,
        num_layers=args.lstm_num_layers, dropout=args.lstm_dropout,
    ).to(device)
    lstm_model.load_state_dict(torch.load(args.lstm_checkpoint, map_location=device, weights_only=True))
    lstm_model.eval()
    print(f"Loaded frozen CNN-LSTM checkpoint: {args.lstm_checkpoint}")

    transformer_model = CNNTransformer(
        input_dim=512, d_model=args.transformer_d_model, nhead=args.transformer_nhead,
        num_layers=args.transformer_num_layers, dim_feedforward=args.transformer_d_model * 2,
        dropout=args.transformer_dropout,
    ).to(device)
    transformer_model.load_state_dict(
        torch.load(args.transformer_checkpoint, map_location=device, weights_only=True)
    )
    transformer_model.eval()
    print(f"Loaded frozen CNN-Transformer checkpoint: {args.transformer_checkpoint}")

    # --- Preload all pairs' full embedding sequences once ---
    fp_to_loc = load_embedding_index(args.embeddings_dir)
    pair_data = load_all_pair_embeddings(test_df, args.embeddings_dir, fp_to_loc)

    # --- Run the drop-level x seed grid ---
    results_rows = []
    per_trial_rows = []  # per-pair, per-drop-level, per-seed predictions, for auditing

    for drop_frac in DROP_LEVELS:
        for seed in range(N_SEEDS):
            if drop_frac == 0.0 and seed > 0:
                # 0% drop is deterministic (nothing is dropped, no
                # randomness involved) -- running it 5x would just waste
                # compute reproducing the identical result. Run once,
                # reuse for all 5 "seeds" at this level for a consistent
                # table shape.
                pass

            rng = np.random.RandomState(seed)
            lstm_y_true, lstm_y_pred = [], []
            tfmr_y_true, tfmr_y_pred = [], []

            for pair_id, d in pair_data.items():
                n_total = d["embeddings"].shape[0]
                if drop_frac == 0.0:
                    keep_idx = list(range(n_total))
                else:
                    keep_idx = drop_indices(n_total, drop_frac, rng)

                kept_embs = d["embeddings"][keep_idx]
                kept_days = d["days_raw"][keep_idx]
                target = d["target_diameter"]

                lstm_pred = predict_lstm(
                    lstm_model, kept_embs, kept_days, day_mean, day_std,
                    target_mean, target_std, device,
                )
                tfmr_pred = predict_transformer(
                    transformer_model, kept_embs, kept_days,
                    target_mean, target_std, device,
                )

                lstm_y_true.append(target)
                lstm_y_pred.append(lstm_pred)
                tfmr_y_true.append(target)
                tfmr_y_pred.append(tfmr_pred)

                per_trial_rows.append({
                    "pair_id": pair_id, "drop_frac": drop_frac, "seed": seed,
                    "n_total": n_total, "n_kept": len(keep_idx),
                    "y_true": target, "lstm_pred": lstm_pred, "transformer_pred": tfmr_pred,
                })

            lstm_mae, lstm_rmse = mae_rmse(lstm_y_true, lstm_y_pred)
            tfmr_mae, tfmr_rmse = mae_rmse(tfmr_y_true, tfmr_y_pred)

            results_rows.append({
                "drop_frac": drop_frac, "seed": seed,
                "lstm_mae": lstm_mae, "lstm_rmse": lstm_rmse,
                "transformer_mae": tfmr_mae, "transformer_rmse": tfmr_rmse,
            })
            print(f"drop={drop_frac:.0%} seed={seed}: "
                  f"LSTM MAE={lstm_mae:.3f} RMSE={lstm_rmse:.3f} | "
                  f"Transformer MAE={tfmr_mae:.3f} RMSE={tfmr_rmse:.3f}")

            if drop_frac == 0.0:
                # deterministic -- duplicate this single result across all
                # N_SEEDS rows for a uniform table shape, then stop looping
                # seeds for this drop level.
                for s in range(1, N_SEEDS):
                    results_rows.append({
                        "drop_frac": drop_frac, "seed": s,
                        "lstm_mae": lstm_mae, "lstm_rmse": lstm_rmse,
                        "transformer_mae": tfmr_mae, "transformer_rmse": tfmr_rmse,
                    })
                break

    os.makedirs(args.out_dir, exist_ok=True)
    trials_df = pd.DataFrame(results_rows)
    trials_df.to_csv(os.path.join(args.out_dir, "step_p2_robustness_trials.csv"), index=False)

    per_pair_df = pd.DataFrame(per_trial_rows)
    per_pair_df.to_csv(os.path.join(args.out_dir, "step_p2_robustness_per_pair.csv"), index=False)

    # --- Summarize: mean +/- std per drop level ---
    summary = trials_df.groupby("drop_frac").agg(
        lstm_mae_mean=("lstm_mae", "mean"), lstm_mae_std=("lstm_mae", "std"),
        lstm_rmse_mean=("lstm_rmse", "mean"), lstm_rmse_std=("lstm_rmse", "std"),
        transformer_mae_mean=("transformer_mae", "mean"), transformer_mae_std=("transformer_mae", "std"),
        transformer_rmse_mean=("transformer_rmse", "mean"), transformer_rmse_std=("transformer_rmse", "std"),
    ).reset_index()
    summary.to_csv(os.path.join(args.out_dir, "step_p2_robustness_summary.csv"), index=False)

    print("\n" + "=" * 100)
    print("MISSING-OBSERVATION ROBUSTNESS TABLE (mean +/- std over 5 seeds; drop=0% is deterministic)")
    print("=" * 100)
    for _, row in summary.iterrows():
        print(f"drop={row['drop_frac']:.0%}  "
              f"LSTM MAE={row['lstm_mae_mean']:.3f}+/-{row['lstm_mae_std']:.3f}  "
              f"RMSE={row['lstm_rmse_mean']:.3f}+/-{row['lstm_rmse_std']:.3f}  |  "
              f"Transformer MAE={row['transformer_mae_mean']:.3f}+/-{row['transformer_mae_std']:.3f}  "
              f"RMSE={row['transformer_rmse_mean']:.3f}+/-{row['transformer_rmse_std']:.3f}")

    print(f"\nWrote per-trial results to {args.out_dir}/step_p2_robustness_trials.csv")
    print(f"Wrote per-pair per-trial predictions to {args.out_dir}/step_p2_robustness_per_pair.csv")
    print(f"Wrote summary table to {args.out_dir}/step_p2_robustness_summary.csv")
    print("\nSTOPPING HERE for review before any README conclusions are written, per instructions.")


if __name__ == "__main__":
    main()
