"""
Shared training core for the CNN-LSTM, used by both the single-run script
(train_cnn_lstm.py) and the hyperparameter sweep (sweep_cnn_lstm.py).

IMPORTANT model-selection boundary: `train_one_config()` trains on TRAIN
and selects its own best checkpoint using VAL MAE only. It never touches
the test split -- test predictions are computed by a separate call
(`predict()`) that the CALLER invokes explicitly, and the sweep script is
responsible for calling it exactly once, only for the single config
selected by validation performance. This function does not decide whether
or when test is evaluated; that decision, and the "touch test exactly
once" discipline, lives in the caller.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from cnn_lstm_dataset import CNNLSTMDataset, collate_fn
from cnn_lstm_model import CNNLSTM


def mae_rmse(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def make_pair_id(df):
    return df["plant_id"].astype(str) + "::" + df["target_day"].astype(str)


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_samples = 0
    loss_fn = torch.nn.MSELoss()

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"]
            target_norm = batch["target_norm"].to(device)

            pred_norm = model(features, lengths)
            loss = loss_fn(pred_norm, target_norm)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * features.shape[0]
            n_samples += features.shape[0]

    return total_loss / n_samples


@torch.no_grad()
def predict(model, loader, device, target_mean, target_std):
    model.eval()
    pair_ids, y_true, y_pred = [], [], []
    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"]
        pred_norm = model(features, lengths).cpu().numpy()
        pred_raw = pred_norm * target_std + target_mean

        pair_ids.extend(batch["pair_id"])
        y_true.extend(batch["target_raw"].numpy().tolist())
        y_pred.extend(pred_raw.tolist())
    return pd.DataFrame({"pair_id": pair_ids, "y_true": y_true, "y_pred": y_pred})


def build_datasets(pairs_df, embeddings_dir, norm_stats):
    target_mean = norm_stats["target_diameter_mean"]
    target_std = norm_stats["target_diameter_std"]
    day_mean = norm_stats["day_after_planting_mean"]
    day_std = norm_stats["day_after_planting_std"]

    train_df = pairs_df[pairs_df["split"] == "train"]
    val_df = pairs_df[pairs_df["split"] == "val"]
    test_df = pairs_df[pairs_df["split"] == "test"]

    common_kwargs = dict(
        embeddings_dir=embeddings_dir,
        day_mean=day_mean, day_std=day_std,
        target_mean=target_mean, target_std=target_std,
    )
    train_ds = CNNLSTMDataset(train_df, **common_kwargs)
    val_ds = CNNLSTMDataset(val_df, **common_kwargs)
    test_ds = CNNLSTMDataset(test_df, **common_kwargs)
    return train_ds, val_ds, test_ds, train_df, val_df, test_df


def train_one_config(
    train_ds, val_ds, config, device, checkpoint_path,
    target_mean, target_std, seed=42, verbose=True,
):
    """
    Train a single CNN-LSTM configuration with early stopping on VAL MAE.
    Returns (best_val_mae, best_val_rmse, history, checkpoint_path).
    Does NOT touch test data at all -- caller decides if/when to evaluate
    test, and for which config.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = CNNLSTM(
        input_dim=513, hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"], dropout=config.get("dropout", 0.0),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_val_mae = np.inf
    best_val_rmse = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_preds = predict(model, val_loader, device, target_mean, target_std)
        val_mae, val_rmse = mae_rmse(val_preds["y_true"], val_preds["y_pred"])

        history.append({"epoch": epoch, "train_loss_norm_mse": train_loss, "val_mae": val_mae, "val_rmse": val_rmse})
        if verbose:
            print(f"  Epoch {epoch:3d}  train_loss(norm MSE)={train_loss:.4f}  "
                  f"val_MAE={val_mae:.3f}  val_RMSE={val_rmse:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no val improvement for {config['patience']} epochs). "
                          f"Best val MAE={best_val_mae:.3f}")
                break

    return best_val_mae, best_val_rmse, history, checkpoint_path
