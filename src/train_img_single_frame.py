"""
Step C: single-frame learned image forecaster (first trained image model, NOT temporal).

Input per pair: ONLY the most recent input image + scalar day offset
(target_day - input_last_day). Earlier history frames are never read. Output: generated
256x256 RGB image. Architecture in img_film_unet.py (CNN encoder -> FiLM at bottleneck ->
U-Net decoder). Loss = w_l1 * L1 + w_ssim * (1 - SSIM) with a differentiable Gaussian SSIM;
no adversarial term and NO colour-match / oracle logic anywhere in the loss or model.

Model-selection discipline (same as every earlier step): TRAIN on train; the best epoch
is chosen by VALIDATION combined loss only (val L1, val SSIM, val PSNR are also logged);
TEST is evaluated exactly once, after training, for that single selected checkpoint, with
the same skimage RGB SSIM/PSNR + structure-SSIM used for the copy-forward baseline.
--smoke never touches the test split (it exercises the final-evaluation code on a few
VAL pairs instead and writes smoke_* files).

Augmentation (train only): random flips / 90-degree rotations applied identically to input
and target (nadir imagery has no preferred orientation; no photometric augmentation).

Usage:
    python src/train_img_single_frame.py --image-pairs data/image_pairs.parquet \
        --cache-dir data --out-dir outputs --checkpoint-dir checkpoints --device cuda
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from image_copy_forward_baseline import structure_ssim
from img_film_unet import FiLMUNet, gaussian_window, ssim_torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

EXPECTED_COUNTS = {"train": 5823, "val": 1330, "test": 1233}


def load_rows(pairs_path, cache_dir):
    pairs = pd.read_parquet(pairs_path)
    idx = pd.read_parquet(os.path.join(cache_dir, "images256_index.parquet")).set_index("filepath")["row"]
    pairs = pairs.copy()
    pairs["row_in"] = pairs["input_last_filepath"].map(idx)   # ONLY the last input frame is used
    pairs["row_tg"] = pairs["target_filepath"].map(idx)
    assert pairs["row_in"].notna().all() and pairs["row_tg"].notna().all(), "image missing from cache"
    pairs["offset"] = (pairs["target_day"] - pairs["input_last_day"]).astype(float)
    return pairs


def to_gpu(arr, rows, device):
    return torch.from_numpy(arr[rows]).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def augment(x, y):
    """Identical random flip/rot90 per sample for input and target (GPU)."""
    for i in range(x.shape[0]):
        k = int(torch.randint(0, 4, (1,)))
        if k:
            x[i] = torch.rot90(x[i], k, (1, 2))
            y[i] = torch.rot90(y[i], k, (1, 2))
        if torch.rand(1) < 0.5:
            x[i] = x[i].flip(2)
            y[i] = y[i].flip(2)
    return x, y


def psnr_torch(p, t):
    mse = ((p - t) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean()


def batches(n, bs, shuffle, rng):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def run_epoch(model, arr, df, norm, args, device, win, optimizer=None, rng=None):
    train = optimizer is not None
    model.train(train)
    tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "psnr": 0.0}
    n = 0
    offs = ((df["offset"].values - norm["mean"]) / norm["std"]).astype(np.float32)
    for b in batches(len(df), args.batch_size, train, rng):
        x = to_gpu(arr, df["row_in"].values[b], device)
        y = to_gpu(arr, df["row_tg"].values[b], device)
        d = torch.from_numpy(offs[b]).to(device)
        if train:
            x, y = augment(x, y)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.amp)):
                p = model(x, d)
            p = p.float()
            l1 = (p - y).abs().mean()
            ss = ssim_torch(p, y, win)
            loss = args.w_l1 * l1 + args.w_ssim * (1 - ss)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        m = len(b)
        n += m
        tot["loss"] += loss.item() * m
        tot["l1"] += l1.item() * m
        tot["ssim"] += ss.item() * m
        tot["psnr"] += psnr_torch(p.detach(), y).item() * m
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def predict_uint8(model, arr, df, norm, device, bs=32):
    model.eval()
    offs = ((df["offset"].values - norm["mean"]) / norm["std"]).astype(np.float32)
    out = []
    for b in batches(len(df), bs, False, None):
        x = to_gpu(arr, df["row_in"].values[b], device)
        p = model(x, torch.from_numpy(offs[b]).to(device)).float().clamp(0, 1)
        out.append((p * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)


def score_pairs(preds, arr, df):
    """Per-pair metrics identical to the copy-forward baseline (skimage RGB SSIM/PSNR + structure SSIM)."""
    rows = []
    for p, r in zip(preds, df.itertuples(index=False)):
        t = arr[r.row_tg]
        rows.append({"pair_id": r.pair_id, "plant_id": r.plant_id, "field": r.field, "split": r.split,
                     "gap_days": r.gap_days, "n_input_frames": r.n_input_frames,
                     "ssim": structural_similarity(p, t, channel_axis=2, data_range=255),
                     "psnr": peak_signal_noise_ratio(p, t, data_range=255),
                     "structure_ssim": structure_ssim(p, t)})
    res = pd.DataFrame(rows)
    res["psnr"] = res["psnr"].replace(np.inf, np.nan)
    return res


def figure(preds, arr, df, path, k_per_field=3, seed=0):
    rng = np.random.RandomState(seed)
    sel = []
    for f in sorted(df["field"].unique()):
        ix = np.where(df["field"].values == f)[0]
        sel += list(rng.choice(ix, min(k_per_field, len(ix)), replace=False))
    fig, axes = plt.subplots(len(sel), 4, figsize=(10, 2.6 * len(sel)))
    for ax_row, i in zip(np.atleast_2d(axes), sel):
        r = df.iloc[i]
        a, t, p = arr[int(r.row_in)], arr[int(r.row_tg)], preds[i]
        d = np.abs(p.astype(int) - t.astype(int)).astype(np.uint8)
        for ax, im, ttl in zip(ax_row, [a, p, t, d], [f"input d{int(r.input_last_day)}", "prediction",
                                                      f"target d{int(r.target_day)}", "|pred-target|"]):
            ax.imshow(im); ax.axis("off"); ax.set_title(ttl, fontsize=7)
        ax_row[1].set_title(f"prediction  {r.plant_id}", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def plot_history(hist, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    ep = [h["epoch"] for h in hist]
    for ax, key, title in zip(axes, ["loss", "l1", "ssim"], ["combined loss", "L1 term", "SSIM (higher better)"]):
        ax.plot(ep, [h["train"][key] for h in hist], label="train")
        ax.plot(ep, [h["val"][key] for h in hist], label="val")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=1.0)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="tiny run, never reads the test split")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tag = "smoke_" if args.smoke else ""

    pairs = load_rows(args.image_pairs, args.cache_dir)
    arr = np.load(os.path.join(args.cache_dir, "images256.npy"))
    counts = pairs["split"].value_counts().to_dict()
    print(f"Pairs used: {counts}  | images cached: {arr.shape}", flush=True)
    if not args.smoke:
        assert counts == EXPECTED_COUNTS, f"unexpected pair counts {counts}"
    tr = pairs[pairs.split == "train"].reset_index(drop=True)
    va = pairs[pairs.split == "val"].reset_index(drop=True)
    if args.smoke:
        tr, va = tr.sample(min(48, len(tr)), random_state=0).reset_index(drop=True), \
                 va.sample(min(16, len(va)), random_state=0).reset_index(drop=True)
        args.max_epochs = 2
    norm = {"mean": float(tr["offset"].mean()), "std": float(tr["offset"].std() or 1.0)}  # train only
    print(f"Day-offset normalization (train only): {norm}", flush=True)

    model = FiLMUNet().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | device {device} | "
          f"loss = {args.w_l1}*L1 + {args.w_ssim}*(1-SSIM)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    win = gaussian_window(device=device)

    ckpt = os.path.join(args.checkpoint_dir, f"{tag}img_single_frame_best.pt")
    hist, best, bad = [], float("inf"), 0
    for ep in range(1, args.max_epochs + 1):
        t0 = time.time()
        trm = run_epoch(model, arr, tr, norm, args, device, win, optimizer=opt, rng=rng)
        vam = run_epoch(model, arr, va, norm, args, device, win)
        sched.step(vam["loss"])
        hist.append({"epoch": ep, "train": trm, "val": vam, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0})
        improved = vam["loss"] < best
        if improved:
            best, bad = vam["loss"], 0
            torch.save({"model": model.state_dict(), "norm": norm, "epoch": ep, "args": vars(args)}, ckpt)
        else:
            bad += 1
        print(f"ep {ep:3d} | train loss {trm['loss']:.4f} (L1 {trm['l1']:.4f}, SSIM {trm['ssim']:.4f}) | "
              f"val loss {vam['loss']:.4f} (L1 {vam['l1']:.4f}, SSIM {vam['ssim']:.4f}, PSNR {vam['psnr']:.2f}) | "
              f"lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s {'*' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"Early stopping at epoch {ep} (no val-loss improvement for {args.patience} epochs)", flush=True)
            break

    json.dump(hist, open(os.path.join(args.out_dir, f"{tag}img_single_frame_history.json"), "w"), indent=1)
    plot_history(hist, os.path.join(args.out_dir, f"{tag}img_single_frame_curves.png"))
    best_ep = min(hist, key=lambda h: h["val"]["loss"])["epoch"]
    print(f"Selected checkpoint: epoch {best_ep} (best val combined loss {best:.4f}) -- chosen on VAL only", flush=True)

    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd["model"])
    if args.smoke:
        ev, label = va, "SMOKE (val pairs; test not touched)"
    else:
        ev, label = pairs[pairs.split == "test"].reset_index(drop=True), "TEST EVALUATION"
    print(f"=== {label}: {len(ev)} pairs ===", flush=True)
    preds = predict_uint8(model, arr, ev, norm, device)
    res = score_pairs(preds, arr, ev)
    res.to_csv(os.path.join(args.out_dir, f"{tag}img_single_frame_per_pair.csv"), index=False)
    figure(preds, arr, ev, os.path.join(args.out_dir, f"{tag}img_single_frame_examples.png"))
    print(res[["ssim", "psnr", "structure_ssim"]].agg(["mean", "median"]).round(4).to_string(), flush=True)
    json.dump({"best_epoch": best_ep, "best_val_loss": best, "norm": norm, "args": vars(args),
               "n_pairs_by_split": counts}, open(os.path.join(args.out_dir, f"{tag}img_single_frame_run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
