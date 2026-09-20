"""
Phase 2: Hyperparameter sweep for the CNN-Transformer.

Grid: d_model x nhead x num_layers x lr, with dropout swept only for
num_layers=2 (mirrors the CNN-LSTM sweep's treatment of overfitting risk
for deeper models on only 517 training plants). nhead must evenly divide
d_model, so only valid (d_model, nhead) pairs are included.

  d_model:    [64, 128]
  nhead:      [4, 8]   (only where d_model % nhead == 0)
  num_layers: [1, 2]
  lr:         [1e-3, 3e-4]
  dropout:    [0.0] for num_layers=1, [0.0, 0.2] for num_layers=2

MODEL SELECTION DISCIPLINE (identical to sweep_cnn_lstm.py, Phase 1):
every config is trained and compared using VALIDATION MAE ONLY. Test is
never touched during the sweep loop. After the full grid finishes, the
single config with the best (lowest) validation MAE is identified, its
checkpoint is reloaded, and test is evaluated EXACTLY ONCE, for that one
winning config.

The final comparison is 4-way: persistence, single-frame, CNN-LSTM sweep
winner (Phase 1), and this CNN-Transformer sweep winner -- all on the
same shared-evaluation-set convention.

Usage:
    python src/sweep_cnn_transformer.py \
        --pairs-split data/pairs_split.parquet \
        --embeddings-dir data/embeddings \
        --norm-stats data/norm_stats.json \
        --baseline-results outputs/step5_baseline_results.json \
        --lstm-results outputs/step6_sweep_winner_results.json \
        --out-dir outputs \
        --checkpoint-dir checkpoints \
        --device cuda
"""
import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cnn_transformer_dataset import collate_fn
from cnn_transformer_model import CNNTransformer
from cnn_transformer_train_core import (
    mae_rmse, make_pair_id, predict, build_datasets, train_one_config,
)


def build_grid():
    configs = []
    for d_model, nhead, lr in itertools.product([64, 128], [4, 8], [1e-3, 3e-4]):
        if d_model % nhead != 0:
            continue
        dim_feedforward = d_model * 2
        configs.append({
            "d_model": d_model, "nhead": nhead, "lr": lr, "num_layers": 1,
            "dropout": 0.0, "dim_feedforward": dim_feedforward,
        })
        for dropout in [0.0, 0.2]:
            configs.append({
                "d_model": d_model, "nhead": nhead, "lr": lr, "num_layers": 2,
                "dropout": dropout, "dim_feedforward": dim_feedforward,
            })
    return configs


def config_id(config):
    return f"d{config['d_model']}_h{config['nhead']}_lr{config['lr']}_L{config['num_layers']}_do{config['dropout']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--baseline-results", default=None)
    ap.add_argument("--lstm-results", default=None,
                     help="outputs/step6_sweep_winner_results.json, for the 4-way comparison")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-plot-plants", type=int, default=6)
    args = ap.parse_args()

    device = torch.device(args.device)

    pairs_df = pd.read_parquet(args.pairs_split)
    pairs_df["pair_id"] = make_pair_id(pairs_df)
    assert pairs_df["pair_id"].is_unique

    with open(args.norm_stats) as f:
        norm_stats = json.load(f)
    target_mean = norm_stats["target_diameter_mean"]
    target_std = norm_stats["target_diameter_std"]

    train_ds, val_ds, test_ds, train_df, val_df, test_df = build_datasets(
        pairs_df, args.embeddings_dir, norm_stats
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    sweep_ckpt_dir = os.path.join(args.checkpoint_dir, "transformer_sweep")
    os.makedirs(sweep_ckpt_dir, exist_ok=True)

    grid = build_grid()
    print(f"Sweeping {len(grid)} CNN-Transformer configs. Model selection is by VALIDATION MAE only; "
          f"test is touched exactly once, at the end, for the single winning config.\n")

    sweep_results = []
    for i, base_config in enumerate(grid):
        cid = config_id(base_config)
        config = {**base_config, "batch_size": args.batch_size,
                  "max_epochs": args.max_epochs, "patience": args.patience}
        ckpt_path = os.path.join(sweep_ckpt_dir, f"{cid}.pt")

        print(f"[{i+1}/{len(grid)}] {cid}")
        best_val_mae, best_val_rmse, history, _ = train_one_config(
            train_ds, val_ds, config, device, ckpt_path,
            target_mean, target_std, seed=args.seed, verbose=False,
        )
        print(f"    -> best val MAE={best_val_mae:.3f}, val RMSE={best_val_rmse:.3f}, "
              f"epochs_trained={len(history)}")

        sweep_results.append({
            "config_id": cid, **base_config,
            "best_val_mae": best_val_mae, "best_val_rmse": best_val_rmse,
            "epochs_trained": len(history), "checkpoint_path": ckpt_path,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    sweep_df = pd.DataFrame(sweep_results).sort_values("best_val_mae")
    sweep_df.to_csv(os.path.join(args.out_dir, "step_p2_transformer_sweep_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("TRANSFORMER SWEEP RESULTS (sorted by validation MAE, best first)")
    print("=" * 60)
    print(sweep_df[["config_id", "d_model", "nhead", "num_layers", "lr", "dropout",
                     "best_val_mae", "best_val_rmse", "epochs_trained"]].to_string(index=False))

    winner = sweep_df.iloc[0]
    print(f"\nWinning config (lowest val MAE): {winner['config_id']} "
          f"(val MAE={winner['best_val_mae']:.3f})")

    # ============================================================
    # Evaluate TEST exactly once, for the winning config only.
    # ============================================================
    model = CNNTransformer(
        input_dim=512, d_model=int(winner["d_model"]), nhead=int(winner["nhead"]),
        num_layers=int(winner["num_layers"]), dim_feedforward=int(winner["dim_feedforward"]),
        dropout=winner["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(winner["checkpoint_path"], weights_only=True))

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print("\n" + "=" * 60)
    print(f"TEST EVALUATION -- WINNING CONFIG ONLY ({winner['config_id']})")
    print("=" * 60)
    test_preds = predict(model, test_loader, device, target_mean, target_std)
    test_preds["residual"] = test_preds["y_pred"] - test_preds["y_true"]
    test_preds["abs_error"] = test_preds["residual"].abs()

    test_mae, test_rmse = mae_rmse(test_preds["y_true"], test_preds["y_pred"])
    n_test_plants = test_df["plant_id"].nunique()
    print(f"Full test set: N={len(test_preds)} pairs ({n_test_plants} plants), "
          f"MAE={test_mae:.3f}, RMSE={test_rmse:.3f}")

    test_preds.sort_values("abs_error", ascending=False).to_csv(
        os.path.join(args.out_dir, "step_p2_cnn_transformer_test_predictions.csv"), index=False
    )

    results = {
        "sweep_grid_size": len(grid),
        "winning_config": {
            "config_id": winner["config_id"],
            "d_model": int(winner["d_model"]), "nhead": int(winner["nhead"]),
            "num_layers": int(winner["num_layers"]), "lr": winner["lr"], "dropout": winner["dropout"],
            "val_mae": winner["best_val_mae"], "val_rmse": winner["best_val_rmse"],
        },
        "cnn_transformer_sweep_winner": {
            "n_pairs": len(test_preds), "n_plants": n_test_plants,
            "mae": test_mae, "rmse": test_rmse,
        },
    }

    # ============================================================
    # 4-way shared-evaluation-set comparison: persistence, single-frame,
    # CNN-LSTM sweep winner, CNN-Transformer sweep winner.
    # ============================================================
    all_preds = {"cnn_transformer": test_preds}
    if args.baseline_results and os.path.exists(args.baseline_results):
        baseline_pred_dir = os.path.dirname(args.baseline_results)
        persistence_csv = os.path.join(baseline_pred_dir, "step5_persistence_test_predictions.csv")
        single_frame_csv = os.path.join(baseline_pred_dir, "step5_single_frame_test_predictions.csv")
        if os.path.exists(persistence_csv):
            all_preds["persistence"] = pd.read_csv(persistence_csv)
        if os.path.exists(single_frame_csv):
            all_preds["single_frame"] = pd.read_csv(single_frame_csv)

    lstm_pred_csv = os.path.join(args.out_dir, "step6_cnn_lstm_sweep_winner_test_predictions.csv")
    if os.path.exists(lstm_pred_csv):
        all_preds["cnn_lstm"] = pd.read_csv(lstm_pred_csv)

    if len(all_preds) > 1:
        print("\n" + "=" * 60)
        print(f"SHARED EVALUATION SET ({len(all_preds)}-way comparison)")
        print("=" * 60)
        shared_ids = None
        for name, df in all_preds.items():
            ids = set(df["pair_id"])
            shared_ids = ids if shared_ids is None else (shared_ids & ids)
        print(f"Shared test pairs (all {len(all_preds)} methods can predict): {len(shared_ids)}")

        shared_results = {}
        for name, df in all_preds.items():
            sub = df[df["pair_id"].isin(shared_ids)]
            mae, rmse = mae_rmse(sub["y_true"], sub["y_pred"])
            print(f"  {name}: N={len(sub)}, MAE={mae:.3f}, RMSE={rmse:.3f}")
            shared_results[name] = {"n_pairs": len(sub), "mae": mae, "rmse": rmse}

        if "cnn_lstm" in shared_results:
            delta_mae = 100 * (1 - shared_results["cnn_transformer"]["mae"] / shared_results["cnn_lstm"]["mae"])
            delta_rmse = 100 * (1 - shared_results["cnn_transformer"]["rmse"] / shared_results["cnn_lstm"]["rmse"])
            print(f"\nCNN-Transformer vs CNN-LSTM (Phase 1 sweep winner): "
                  f"MAE {delta_mae:+.1f}%, RMSE {delta_rmse:+.1f}%")
            results["cnn_transformer_vs_cnn_lstm_mae_improvement_pct"] = delta_mae
            results["cnn_transformer_vs_cnn_lstm_rmse_improvement_pct"] = delta_rmse

        results["shared_eval_set"] = {"n_pairs": len(shared_ids), "methods": shared_results}

        # Size-quartile bias check, continuing the same trend line from Phase 1.
        print("\nSize-quartile bias check (continuing the Phase 1 trend line):")
        transformer_shared = test_preds[test_preds["pair_id"].isin(shared_ids)]
        corr = transformer_shared["y_true"].corr(transformer_shared["residual"])
        print(f"  Correlation(true_diameter, residual): {corr:.3f} "
              f"(single-frame -0.355, LSTM manual -0.265, LSTM sweep -0.215)")
        results["shared_eval_set"]["size_quartile_bias_correlation"] = corr
    else:
        print("\nNo baseline/LSTM prediction CSVs found -- skipping shared-eval-set comparison.")

    with open(os.path.join(args.out_dir, "step_p2_transformer_sweep_winner_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {args.out_dir}/step_p2_transformer_sweep_winner_results.json")
    print(f"Wrote full sweep table to {args.out_dir}/step_p2_transformer_sweep_results.csv")

    # ============================================================
    # Growth curve plots for the winning config
    # ============================================================
    plot_plant_ids = sorted(test_df["plant_id"].unique())[:args.n_plot_plants]
    for plant_id in plot_plant_ids:
        plant_pairs = test_df[test_df["plant_id"] == plant_id].sort_values("target_day")
        plant_preds = test_preds[test_preds["pair_id"].isin(plant_pairs["pair_id"])]
        plant_preds = plant_preds.merge(plant_pairs[["pair_id", "target_day"]], on="pair_id").sort_values("target_day")
        if len(plant_preds) == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(plant_preds["target_day"], plant_preds["y_true"], "o-", label="Actual", color="tab:green")
        ax.plot(plant_preds["target_day"], plant_preds["y_pred"], "o--",
                label=f"Predicted (CNN-Transformer, {winner['config_id']})", color="tab:purple")
        ax.set_xlabel("Day after planting")
        ax.set_ylabel("Plant diameter (mm)")
        ax.set_title(f"Growth curve: {plant_id} (test)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"step_p2_transformer_growth_curve_{plant_id}.png"), dpi=150)
        plt.close(fig)
    print(f"Wrote {len(plot_plant_ids)} growth curve plots to "
          f"{args.out_dir}/step_p2_transformer_growth_curve_<plant_id>.png")


if __name__ == "__main__":
    main()
