"""
Step 6 hyperparameter sweep for the CNN-LSTM.

Grid: hidden_dim x lr x num_layers, with dropout swept only for
num_layers=2 configs (dropout has no effect on a 1-layer nn.LSTM and
PyTorch warns/ignores it there, so 1-layer configs are only run once
per hidden_dim/lr rather than duplicated per dropout value).

  hidden_dim: [64, 128, 256]
  lr:         [1e-3, 3e-4]
  num_layers: [1, 2]
  dropout:    [0.0] for num_layers=1, [0.0, 0.2] for num_layers=2

-> 3 * 2 * 1 (num_layers=1, dropout=0.0 only) = 6 configs
 + 3 * 2 * 2 (num_layers=2, dropout in [0.0, 0.2]) = 12 configs
 = 18 configs total.

MODEL SELECTION DISCIPLINE (the whole point of this script existing
separately from train_cnn_lstm.py): every config is trained and compared
using VALIDATION MAE ONLY. Test is never touched during the sweep loop.
After the full grid finishes, the single config with the best (lowest)
validation MAE is identified, its best checkpoint (already saved during
its own training run) is reloaded, and test is evaluated EXACTLY ONCE,
for that one winning config. This mirrors the discipline in
src/baselines.py's ridge alpha sweep (also selected on val, evaluated on
test once for the winner) -- the same principle applied to the LSTM's
larger hyperparameter space.

Usage:
    python src/sweep_cnn_lstm.py \
        --pairs-split data/pairs_split.parquet \
        --embeddings-dir data/embeddings \
        --norm-stats data/norm_stats.json \
        --baseline-results outputs/step5_baseline_results.json \
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

from cnn_lstm_dataset import collate_fn
from cnn_lstm_model import CNNLSTM
from cnn_lstm_train_core import (
    mae_rmse, make_pair_id, predict, build_datasets, train_one_config,
)


def build_grid():
    configs = []
    for hidden_dim, lr in itertools.product([64, 128, 256], [1e-3, 3e-4]):
        # num_layers=1: dropout is a no-op in nn.LSTM, so only one entry.
        configs.append({"hidden_dim": hidden_dim, "lr": lr, "num_layers": 1, "dropout": 0.0})
        # num_layers=2: sweep dropout since overfitting risk is real with
        # only 517 training plants and a deeper model.
        for dropout in [0.0, 0.2]:
            configs.append({"hidden_dim": hidden_dim, "lr": lr, "num_layers": 2, "dropout": dropout})
    return configs


def config_id(config):
    return f"h{config['hidden_dim']}_lr{config['lr']}_L{config['num_layers']}_d{config['dropout']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-split", required=True)
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--norm-stats", required=True)
    ap.add_argument("--baseline-results", default=None)
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
    sweep_ckpt_dir = os.path.join(args.checkpoint_dir, "sweep")
    os.makedirs(sweep_ckpt_dir, exist_ok=True)

    grid = build_grid()
    print(f"Sweeping {len(grid)} configs. Model selection is by VALIDATION MAE only; "
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
    sweep_df.to_csv(os.path.join(args.out_dir, "step6_sweep_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("SWEEP RESULTS (sorted by validation MAE, best first)")
    print("=" * 60)
    print(sweep_df[["config_id", "hidden_dim", "lr", "num_layers", "dropout",
                     "best_val_mae", "best_val_rmse", "epochs_trained"]].to_string(index=False))

    winner = sweep_df.iloc[0]
    print(f"\nWinning config (lowest val MAE): {winner['config_id']} "
          f"(val MAE={winner['best_val_mae']:.3f})")

    # ============================================================
    # Evaluate TEST exactly once, for the winning config only.
    # ============================================================
    winner_config = {
        "hidden_dim": int(winner["hidden_dim"]), "lr": winner["lr"],
        "num_layers": int(winner["num_layers"]), "dropout": winner["dropout"],
    }
    model = CNNLSTM(input_dim=513, hidden_dim=winner_config["hidden_dim"],
                     num_layers=winner_config["num_layers"], dropout=winner_config["dropout"]).to(device)
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
        os.path.join(args.out_dir, "step6_cnn_lstm_sweep_winner_test_predictions.csv"), index=False
    )

    results = {
        "sweep_grid_size": len(grid),
        "winning_config": {
            "config_id": winner["config_id"], **winner_config,
            "val_mae": winner["best_val_mae"], "val_rmse": winner["best_val_rmse"],
        },
        "cnn_lstm_sweep_winner": {
            "n_pairs": len(test_preds), "n_plants": n_test_plants,
            "mae": test_mae, "rmse": test_rmse,
        },
    }

    # ============================================================
    # Shared-evaluation-set comparison against Step 5 baselines
    # ============================================================
    if args.baseline_results and os.path.exists(args.baseline_results):
        baseline_pred_dir = os.path.dirname(args.baseline_results)
        persistence_csv = os.path.join(baseline_pred_dir, "step5_persistence_test_predictions.csv")
        single_frame_csv = os.path.join(baseline_pred_dir, "step5_single_frame_test_predictions.csv")

        if os.path.exists(persistence_csv) and os.path.exists(single_frame_csv):
            print("\n" + "=" * 60)
            print("SHARED EVALUATION SET (sweep winner vs Step 5 baselines)")
            print("=" * 60)
            persistence_df = pd.read_csv(persistence_csv)
            single_frame_df = pd.read_csv(single_frame_csv)

            all_preds = {"persistence": persistence_df, "single_frame": single_frame_df, "cnn_lstm": test_preds}
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
            print(f"\nSweep winner vs single-frame: MAE {lstm_vs_sf_mae:+.1f}%, RMSE {lstm_vs_sf_rmse:+.1f}%")
            print(f"Sweep winner vs persistence: MAE {lstm_vs_pers_mae:+.1f}%")

            results["shared_eval_set"] = {
                "n_pairs": len(shared_ids), "methods": shared_results,
                "sweep_winner_vs_single_frame_mae_improvement_pct": lstm_vs_sf_mae,
                "sweep_winner_vs_single_frame_rmse_improvement_pct": lstm_vs_sf_rmse,
                "sweep_winner_vs_persistence_mae_improvement_pct": lstm_vs_pers_mae,
            }

            print("\nSize-quartile bias check:")
            lstm_shared = test_preds[test_preds["pair_id"].isin(shared_ids)]
            corr = lstm_shared["y_true"].corr(lstm_shared["residual"])
            print(f"  Correlation(true_diameter, residual): {corr:.3f} "
                  f"(single-frame was -0.355, single-config CNN-LSTM was -0.265)")
            q25, q75 = lstm_shared["y_true"].quantile([0.25, 0.75])
            for label, sub in [
                ("bottom quartile", lstm_shared[lstm_shared["y_true"] <= q25]),
                ("middle 50%", lstm_shared[(lstm_shared["y_true"] > q25) & (lstm_shared["y_true"] < q75)]),
                ("top quartile", lstm_shared[lstm_shared["y_true"] >= q75]),
            ]:
                print(f"  {label}: N={len(sub)}, MAE={sub['abs_error'].mean():.2f}, "
                      f"mean residual={sub['residual'].mean():.2f}")
            results["shared_eval_set"]["size_quartile_bias_correlation"] = corr

    with open(os.path.join(args.out_dir, "step6_sweep_winner_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {args.out_dir}/step6_sweep_winner_results.json")
    print(f"Wrote full sweep table to {args.out_dir}/step6_sweep_results.csv")

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
                label=f"Predicted (CNN-LSTM, {winner['config_id']})", color="tab:orange")
        ax.set_xlabel("Day after planting")
        ax.set_ylabel("Plant diameter (mm)")
        ax.set_title(f"Growth curve: {plant_id} (test)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"step6_sweep_growth_curve_{plant_id}.png"), dpi=150)
        plt.close(fig)
    print(f"Wrote {len(plot_plant_ids)} growth curve plots to "
          f"{args.out_dir}/step6_sweep_growth_curve_<plant_id>.png")


if __name__ == "__main__":
    main()
