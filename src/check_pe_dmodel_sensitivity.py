"""
Targeted follow-up: does increasing d_model resolve the sinusoidal
positional encoding's aliasing behavior at small d_model (see the PE
sanity check in the conversation/commit log -- day=91 was found more
cosine-similar to day=15 than to the closer day=22, at d_model=64),
and if so, does the CNN-Transformer's test performance improve?

This is a NARROW, hypothesis-driven check, not a full re-sweep: it
retrains a small set of configs at d_model=256 (double the original
sweep's max of 128) holding other hyperparameters close to the original
winner (d64_h4_lr0.0003_L2_do0.2), to see whether d_model is the
bottleneck specifically, rather than re-running the entire 24-config grid.

Uses the SAME model-selection discipline as every other sweep in this
project: configs are compared by VALIDATION MAE only; test is evaluated
EXACTLY ONCE, for the winner of this small follow-up grid.

Usage:
    python src/check_pe_dmodel_sensitivity.py \
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

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cnn_transformer_dataset import collate_fn
from cnn_transformer_model import CNNTransformer
from cnn_transformer_train_core import (
    mae_rmse, make_pair_id, predict, build_datasets, train_one_config,
)


def build_grid():
    """
    Small grid varying ONLY d_model (64, 128, 256, 512) with nhead scaled
    proportionally to keep head_dim roughly constant (~16), other
    hyperparameters fixed near the original 24-config sweep's winner
    (num_layers=2, dropout=0.2) so this isolates d_model's effect rather
    than re-exploring the full space.
    """
    configs = []
    for d_model, nhead in [(64, 4), (128, 8), (256, 16), (512, 32)]:
        configs.append({
            "d_model": d_model, "nhead": nhead, "lr": 3e-4, "num_layers": 2,
            "dropout": 0.2, "dim_feedforward": d_model * 2,
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
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
    ckpt_dir = os.path.join(args.checkpoint_dir, "pe_dmodel_check")
    os.makedirs(ckpt_dir, exist_ok=True)

    grid = build_grid()
    print(f"Checking {len(grid)} d_model values: "
          f"{[c['d_model'] for c in grid]} (nhead scaled to keep head_dim ~16). "
          f"Model selection is VALIDATION MAE only; test touched exactly once, for the winner.\n")

    results_rows = []
    for i, base_config in enumerate(grid):
        cid = config_id(base_config)
        config = {**base_config, "batch_size": args.batch_size,
                  "max_epochs": args.max_epochs, "patience": args.patience}
        ckpt_path = os.path.join(ckpt_dir, f"{cid}.pt")

        print(f"[{i+1}/{len(grid)}] {cid}")
        best_val_mae, best_val_rmse, history, _ = train_one_config(
            train_ds, val_ds, config, device, ckpt_path,
            target_mean, target_std, seed=args.seed, verbose=False,
        )
        print(f"    -> best val MAE={best_val_mae:.3f}, val RMSE={best_val_rmse:.3f}, "
              f"epochs_trained={len(history)}")

        results_rows.append({
            "config_id": cid, **base_config,
            "best_val_mae": best_val_mae, "best_val_rmse": best_val_rmse,
            "epochs_trained": len(history), "checkpoint_path": ckpt_path,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    results_df = pd.DataFrame(results_rows).sort_values("best_val_mae")
    results_df.to_csv(os.path.join(args.out_dir, "step_p2_pe_dmodel_check_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("d_model SENSITIVITY CHECK (sorted by validation MAE)")
    print("=" * 60)
    print(results_df[["config_id", "d_model", "nhead", "best_val_mae", "best_val_rmse", "epochs_trained"]]
          .to_string(index=False))

    winner = results_df.iloc[0]
    print(f"\nBest d_model in this check: {int(winner['d_model'])} (val MAE={winner['best_val_mae']:.3f})")
    print(f"Original 24-config sweep winner was d_model=64 (val MAE=5.097) -- "
          f"{'CONFIRMS' if winner['d_model'] == 64 else 'CONTRADICTS'} that d_model=64 was already near-optimal.")

    # Evaluate TEST exactly once, for the winner of THIS grid.
    model = CNNTransformer(
        input_dim=512, d_model=int(winner["d_model"]), nhead=int(winner["nhead"]),
        num_layers=int(winner["num_layers"]), dim_feedforward=int(winner["dim_feedforward"]),
        dropout=winner["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(winner["checkpoint_path"], weights_only=True))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print("\n" + "=" * 60)
    print(f"TEST EVALUATION -- WINNER OF THIS CHECK ONLY ({winner['config_id']})")
    print("=" * 60)
    test_preds = predict(model, test_loader, device, target_mean, target_std)
    test_mae, test_rmse = mae_rmse(test_preds["y_true"], test_preds["y_pred"])
    print(f"Full test set: N={len(test_preds)} pairs, MAE={test_mae:.3f}, RMSE={test_rmse:.3f}")

    output = {
        "grid_checked": [c["d_model"] for c in grid],
        "winner": {"config_id": winner["config_id"], "d_model": int(winner["d_model"]),
                   "val_mae": winner["best_val_mae"], "val_rmse": winner["best_val_rmse"]},
        "test_mae": test_mae, "test_rmse": test_rmse,
        "original_sweep_winner_d_model": 64,
        "original_sweep_winner_val_mae": 5.097365847157531,
        "original_sweep_winner_test_mae_shared_set": 5.527928918064374,
        "original_sweep_winner_test_rmse_shared_set": 8.66778764902442,
    }

    if args.baseline_results and os.path.exists(args.baseline_results):
        baseline_pred_dir = os.path.dirname(args.baseline_results)
        lstm_csv = os.path.join(baseline_pred_dir, "step6_cnn_lstm_sweep_winner_test_predictions.csv")
        if os.path.exists(lstm_csv):
            lstm_df = pd.read_csv(lstm_csv)
            shared_ids = set(lstm_df["pair_id"]) & set(test_preds["pair_id"])
            lstm_shared = lstm_df[lstm_df["pair_id"].isin(shared_ids)]
            new_shared = test_preds[test_preds["pair_id"].isin(shared_ids)]
            lstm_mae, lstm_rmse = mae_rmse(lstm_shared["y_true"], lstm_shared["y_pred"])
            new_mae, new_rmse = mae_rmse(new_shared["y_true"], new_shared["y_pred"])
            print(f"\nOn shared set (N={len(shared_ids)}): "
                  f"this d_model={int(winner['d_model'])} config MAE={new_mae:.3f}/RMSE={new_rmse:.3f} "
                  f"vs CNN-LSTM MAE={lstm_mae:.3f}/RMSE={lstm_rmse:.3f}")
            output["shared_set_n"] = len(shared_ids)
            output["shared_set_mae_this_config"] = new_mae
            output["shared_set_rmse_this_config"] = new_rmse
            output["shared_set_mae_cnn_lstm"] = lstm_mae
            output["shared_set_rmse_cnn_lstm"] = lstm_rmse

    with open(os.path.join(args.out_dir, "step_p2_pe_dmodel_check_winner.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote results to {args.out_dir}/step_p2_pe_dmodel_check_winner.json")


if __name__ == "__main__":
    main()
