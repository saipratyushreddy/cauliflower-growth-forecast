"""
Step 6: Train ONE CNN-LSTM configuration and evaluate it on held-out test
plants.

For hyperparameter search, use sweep_cnn_lstm.py instead -- it selects the
winning config by VALIDATION MAE only and touches test exactly once, for
that single winner, so the plant-wise test split is never used for model
selection. This script, when run directly with a single hand-picked
config, also only touches test once (there's only one config to begin
with), so it's safe standalone, but is NOT what should be used to report
"the best config found" -- that requires sweep_cnn_lstm.py's discipline.

Reports MAE/RMSE on the same shared-evaluation-set convention established
in Step 5 (src/baselines.py): predictions are dumped per-pair with
pair_id = plant_id::target_day, and the headline test comparison against
the Step 5 baselines is recomputed on the intersection of pair_ids all
three methods can predict.

Also produces predicted-vs-actual growth curve plots for a handful of
individual test plants (saved to outputs/).

Usage:
    python src/train_cnn_lstm.py \
        --pairs-split data/pairs_split.parquet \
        --embeddings-dir data/embeddings \
        --norm-stats data/norm_stats.json \
        --baseline-results outputs/step5_baseline_results.json \
        --out-dir outputs \
        --checkpoint-dir checkpoints \
        --device cuda
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from cnn_lstm_train_core import (
    mae_rmse, make_pair_id, predict, build_datasets, train_one_config,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--baseline-results", default=None,
                     help="outputs/step5_baseline_results.json, for the shared-eval-set comparison")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
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
    ckpt_path = os.path.join(args.checkpoint_dir, "cnn_lstm_best.pt")

    config = {
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
        "dropout": args.dropout, "batch_size": args.batch_size,
        "lr": args.lr, "max_epochs": args.max_epochs, "patience": args.patience,
    }

    print(f"Training on {device}, {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test samples")
    best_val_mae, best_val_rmse, history, _ = train_one_config(
        train_ds, val_ds, config, device, ckpt_path,
        target_mean, target_std, seed=args.seed, verbose=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "step6_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Reload best checkpoint before the SINGLE test evaluation.
    from cnn_lstm_model import CNNLSTM
    model = CNNLSTM(input_dim=513, hidden_dim=args.hidden_dim,
                     num_layers=args.num_layers, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    from torch.utils.data import DataLoader
    from cnn_lstm_dataset import collate_fn
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print("\n" + "=" * 60)
    print("CNN-LSTM TEST EVALUATION")
    print("=" * 60)
    test_preds = predict(model, test_loader, device, target_mean, target_std)
    test_preds["residual"] = test_preds["y_pred"] - test_preds["y_true"]
    test_preds["abs_error"] = test_preds["residual"].abs()

    test_mae, test_rmse = mae_rmse(test_preds["y_true"], test_preds["y_pred"])
    n_test_plants = test_df["plant_id"].nunique()
    print(f"Full test set: N={len(test_preds)} pairs ({n_test_plants} plants), "
          f"MAE={test_mae:.3f}, RMSE={test_rmse:.3f}")

    test_preds.sort_values("abs_error", ascending=False).to_csv(
        os.path.join(args.out_dir, "step6_cnn_lstm_test_predictions.csv"), index=False
    )

    results = {
        "cnn_lstm": {
            "n_pairs": len(test_preds), "n_plants": n_test_plants,
            "mae": test_mae, "rmse": test_rmse,
            "best_val_mae": best_val_mae,
            "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
            "epochs_trained": len(history),
        }
    }

    # ============================================================
    # Shared-evaluation-set comparison against Step 5 baselines
    # (same convention as src/baselines.py)
    # ============================================================
    if args.baseline_results and os.path.exists(args.baseline_results):
        baseline_pred_dir = os.path.dirname(args.baseline_results)
        persistence_csv = os.path.join(baseline_pred_dir, "step5_persistence_test_predictions.csv")
        single_frame_csv = os.path.join(baseline_pred_dir, "step5_single_frame_test_predictions.csv")

        if os.path.exists(persistence_csv) and os.path.exists(single_frame_csv):
            print("\n" + "=" * 60)
            print("SHARED EVALUATION SET (CNN-LSTM vs Step 5 baselines)")
            print("=" * 60)
            persistence_df = pd.read_csv(persistence_csv)
            single_frame_df = pd.read_csv(single_frame_csv)

            all_preds = {
                "persistence": persistence_df,
                "single_frame": single_frame_df,
                "cnn_lstm": test_preds,
            }
            shared_ids = None
            for name, df in all_preds.items():
                ids = set(df["pair_id"])
                shared_ids = ids if shared_ids is None else (shared_ids & ids)
            print(f"Shared test pairs (all 3 methods can predict): {len(shared_ids)}")

            shared_results = {}
            for name, df in all_preds.items():
                sub = df[df["pair_id"].isin(shared_ids)]
                mae, rmse = mae_rmse(sub["y_true"], sub["y_pred"])
                print(f"  {name}: N={len(sub)}, MAE={mae:.3f}, RMSE={rmse:.3f}")
                shared_results[name] = {"n_pairs": len(sub), "mae": mae, "rmse": rmse}

            lstm_vs_sf_mae = 100 * (1 - shared_results["cnn_lstm"]["mae"] / shared_results["single_frame"]["mae"])
            lstm_vs_sf_rmse = 100 * (1 - shared_results["cnn_lstm"]["rmse"] / shared_results["single_frame"]["rmse"])
            lstm_vs_pers_mae = 100 * (1 - shared_results["cnn_lstm"]["mae"] / shared_results["persistence"]["mae"])
            print(f"\nCNN-LSTM vs single-frame: MAE {lstm_vs_sf_mae:+.1f}%, RMSE {lstm_vs_sf_rmse:+.1f}%")
            print(f"CNN-LSTM vs persistence: MAE {lstm_vs_pers_mae:+.1f}%")

            results["shared_eval_set"] = {
                "n_pairs": len(shared_ids),
                "methods": shared_results,
                "cnn_lstm_vs_single_frame_mae_improvement_pct": lstm_vs_sf_mae,
                "cnn_lstm_vs_single_frame_rmse_improvement_pct": lstm_vs_sf_rmse,
                "cnn_lstm_vs_persistence_mae_improvement_pct": lstm_vs_pers_mae,
            }

            print("\nSize-quartile bias check (same analysis as Step 5's single-frame residuals):")
            lstm_shared = test_preds[test_preds["pair_id"].isin(shared_ids)]
            corr = lstm_shared["y_true"].corr(lstm_shared["residual"])
            print(f"  Correlation(true_diameter, residual): {corr:.3f} (Step 5 single-frame was -0.355)")
            q25, q75 = lstm_shared["y_true"].quantile([0.25, 0.75])
            for label, sub in [
                ("bottom quartile", lstm_shared[lstm_shared["y_true"] <= q25]),
                ("middle 50%", lstm_shared[(lstm_shared["y_true"] > q25) & (lstm_shared["y_true"] < q75)]),
                ("top quartile", lstm_shared[lstm_shared["y_true"] >= q75]),
            ]:
                print(f"  {label}: N={len(sub)}, MAE={sub['abs_error'].mean():.2f}, "
                      f"mean residual={sub['residual'].mean():.2f}")
            results["shared_eval_set"]["size_quartile_bias_correlation"] = corr
        else:
            print("\nStep 5 prediction CSVs not found -- skipping shared-eval-set comparison "
                  "(run src/baselines.py first to produce them).")

    with open(os.path.join(args.out_dir, "step6_cnn_lstm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {args.out_dir}/step6_cnn_lstm_results.json")

    # ============================================================
    # Predicted-vs-actual growth curve plots for a handful of test plants
    # ============================================================
    plot_plant_ids = sorted(test_df["plant_id"].unique())[:args.n_plot_plants]
    for plant_id in plot_plant_ids:
        plant_pairs = test_df[test_df["plant_id"] == plant_id].sort_values("target_day")
        plant_preds = test_preds[test_preds["pair_id"].isin(plant_pairs["pair_id"])]
        plant_preds = plant_preds.merge(
            plant_pairs[["pair_id", "target_day"]], on="pair_id"
        ).sort_values("target_day")

        if len(plant_preds) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(plant_preds["target_day"], plant_preds["y_true"], "o-", label="Actual", color="tab:green")
        ax.plot(plant_preds["target_day"], plant_preds["y_pred"], "o--", label="Predicted (CNN-LSTM)", color="tab:orange")
        ax.set_xlabel("Day after planting")
        ax.set_ylabel("Plant diameter (mm)")
        ax.set_title(f"Growth curve: {plant_id} (test)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"step6_growth_curve_{plant_id}.png"), dpi=150)
        plt.close(fig)

    print(f"Wrote {len(plot_plant_ids)} growth curve plots to {args.out_dir}/step6_growth_curve_<plant_id>.png")


if __name__ == "__main__":
    main()
