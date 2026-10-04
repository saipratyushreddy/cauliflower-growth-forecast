"""
History-aware ConvLSTM image forecaster (see img_convlstm.py). Two runs differ ONLY in --k:
K=4 (primary) and K=1 (reference: the same architecture with no history, isolating the
architecture effect from the history effect).

Same pair set, same L1 + SSIM loss, optimizer, augmentation, early stopping and model-selection
discipline as Step C (train_img_single_frame.py): best epoch by VALIDATION combined loss only; TEST
evaluated exactly once at the end for that checkpoint; --smoke never reads the test split. History comes
from the (post-scrub) `input_filepaths` / `input_days`. Augmentation is applied identically to every frame
and the target.

Usage:
    python src/train_img_convlstm.py --image-pairs data/image_pairs.parquet --cache-dir data \
        --out-dir outputs --checkpoint-dir checkpoints --device cuda --k 4
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from img_convlstm import TemporalFiLMUNet
from img_film_unet import gaussian_window, ssim_torch
from train_img_single_frame import (EXPECTED_COUNTS, batches, figure, load_rows, plot_history, psnr_torch,
                                    score_pairs)


def build_sequences(pairs, cache_idx, K, norm):
    """Per pair: cache rows of the last K input frames (left-padded by repeating the first real frame),
    normalized gap to the next frame (last = target offset), and a real-frame mask."""
    N = len(pairs)
    rows = np.zeros((N, K), dtype=np.int64)
    gaps = np.zeros((N, K), dtype=np.float32)
    mask = np.zeros((N, K), dtype=np.float32)
    n_hist = np.zeros(N, dtype=np.int64)
    for i, r in enumerate(pairs.itertuples(index=False)):
        fps = list(r.input_filepaths)[-K:]
        days = list(r.input_days)[-K:]
        assert fps[-1] == r.input_last_filepath and days[-1] == r.input_last_day
        n = len(fps)
        n_hist[i] = n
        g = [days[j + 1] - days[j] for j in range(n - 1)] + [r.target_day - days[-1]]
        pad = K - n
        rr = [cache_idx[f] for f in fps]
        rows[i] = [rr[0]] * pad + rr
        gaps[i] = [0.0] * pad + g
        mask[i] = [0.0] * pad + [1.0] * n
    gaps = (gaps - norm["mean"]) / norm["std"]
    return rows, gaps.astype(np.float32), mask, n_hist


def gpu_batch(arr, rows, device):
    """rows [b,K] -> frames [b,K,3,H,W] float in [0,1]."""
    x = torch.from_numpy(arr[rows.reshape(-1)]).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    return x.view(rows.shape[0], rows.shape[1], *x.shape[1:])


def augment_seq(x, y):
    for i in range(x.shape[0]):
        k = int(torch.randint(0, 4, (1,)))
        if k:
            x[i] = torch.rot90(x[i], k, (2, 3))
            y[i] = torch.rot90(y[i], k, (1, 2))
        if torch.rand(1) < 0.5:
            x[i] = x[i].flip(3)
            y[i] = y[i].flip(2)
    return x, y


def run_epoch(model, arr, df, S, norm, args, device, win, optimizer=None, rng=None):
    train = optimizer is not None
    model.train(train)
    tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "psnr": 0.0}
    n = 0
    for b in batches(len(df), args.batch_size, train, rng):
        x = gpu_batch(arr, S["rows"][b], device)
        y = torch.from_numpy(arr[df["row_tg"].values[b]]).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        g = torch.from_numpy(S["gaps"][b]).to(device)
        m = torch.from_numpy(S["mask"][b]).to(device)
        if train:
            x, y = augment_seq(x, y)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.amp)):
                p = model(x, g, m, g[:, -1])
            p = p.float()
            l1 = (p - y).abs().mean()
            ss = ssim_torch(p, y, win)
            loss = args.w_l1 * l1 + args.w_ssim * (1 - ss)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        k_ = len(b)
        n += k_
        tot["loss"] += loss.item() * k_
        tot["l1"] += l1.item() * k_
        tot["ssim"] += ss.item() * k_
        tot["psnr"] += psnr_torch(p.detach(), y).item() * k_
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def predict_uint8(model, arr, df, S, device, bs=16):
    model.eval()
    out = []
    for b in batches(len(df), bs, False, None):
        x = gpu_batch(arr, S["rows"][b], device)
        g = torch.from_numpy(S["gaps"][b]).to(device)
        m = torch.from_numpy(S["mask"][b]).to(device)
        p = model(x, g, m, g[:, -1]).float().clamp(0, 1)
        out.append((p * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--k", type=int, required=True, help="number of most recent input frames used")
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
    tag = f"{'smoke_' if args.smoke else ''}img_convlstm_k{args.k}"

    pairs = load_rows(args.image_pairs, args.cache_dir)
    arr = np.load(os.path.join(args.cache_dir, "images256.npy"))
    cache_idx = pd.read_parquet(os.path.join(args.cache_dir, "images256_index.parquet")).set_index("filepath")["row"].to_dict()
    counts = pairs["split"].value_counts().to_dict()
    print(f"K = {args.k} | Pairs used: {counts} | images cached: {arr.shape}", flush=True)
    if not args.smoke:
        assert counts == EXPECTED_COUNTS, f"unexpected pair counts {counts}"
    tr = pairs[pairs.split == "train"].reset_index(drop=True)
    va = pairs[pairs.split == "val"].reset_index(drop=True)
    if args.smoke:
        tr, va = tr.sample(min(24, len(tr)), random_state=0).reset_index(drop=True), \
                 va.sample(min(8, len(va)), random_state=0).reset_index(drop=True)
        args.max_epochs = 2
    norm = {"mean": float(tr["offset"].mean()), "std": float(tr["offset"].std() or 1.0)}  # train only, as Step C
    Str, Sva = (build_sequences(d, cache_idx, args.k, norm) for d in (tr, va))
    Str, Sva = ({"rows": a[0], "gaps": a[1], "mask": a[2]} for a in (Str, Sva))
    nh = build_sequences(tr, cache_idx, args.k, norm)[3]
    print(f"Train pairs by frames used under K={args.k}: "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(nh, return_counts=True))} } (padded where < K)", flush=True)
    print(f"Gap normalization (train only): {norm}", flush=True)

    model = TemporalFiLMUNet().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | device {device} | "
          f"loss = {args.w_l1}*L1 + {args.w_ssim}*(1-SSIM)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    win = gaussian_window(device=device)

    ckpt = os.path.join(args.checkpoint_dir, f"{tag}_best.pt")
    hist, best, bad = [], float("inf"), 0
    for ep in range(1, args.max_epochs + 1):
        t0 = time.time()
        trm = run_epoch(model, arr, tr, Str, norm, args, device, win, optimizer=opt, rng=rng)
        vam = run_epoch(model, arr, va, Sva, norm, args, device, win)
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

    json.dump(hist, open(os.path.join(args.out_dir, f"{tag}_history.json"), "w"), indent=1)
    plot_history(hist, os.path.join(args.out_dir, f"{tag}_curves.png"))
    best_ep = min(hist, key=lambda h: h["val"]["loss"])["epoch"]
    print(f"Selected checkpoint: epoch {best_ep} (best val combined loss {best:.4f}) -- chosen on VAL only", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    if args.smoke:
        ev, label = va, "SMOKE (val pairs; test not touched)"
        Sev = Sva
    else:
        ev, label = pairs[pairs.split == "test"].reset_index(drop=True), "TEST EVALUATION"
        Sev = build_sequences(ev, cache_idx, args.k, norm)
        Sev = {"rows": Sev[0], "gaps": Sev[1], "mask": Sev[2]}
    print(f"=== {label} (K={args.k}): {len(ev)} pairs ===", flush=True)
    preds = predict_uint8(model, arr, ev, Sev, device)
    res = score_pairs(preds, arr, ev)
    res.to_csv(os.path.join(args.out_dir, f"{tag}_per_pair.csv"), index=False)
    figure(preds, arr, ev, os.path.join(args.out_dir, f"{tag}_examples.png"))
    print(res[["ssim", "psnr", "structure_ssim"]].agg(["mean", "median"]).round(4).to_string(), flush=True)
    json.dump({"k": args.k, "best_epoch": best_ep, "best_val_loss": best, "norm": norm, "args": vars(args),
               "n_pairs_by_split": counts}, open(os.path.join(args.out_dir, f"{tag}_run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
