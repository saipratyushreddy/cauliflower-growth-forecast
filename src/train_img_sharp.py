"""
Sharpness ablation on the single-frame model (Step C architecture), one ingredient at a time. Flags (all default OFF = exactly Step C's recipe):

  --residual     ResFiLMUNet: predict the change from the last photo (starts at copy-forward)
  --w-perc W     add W * VGG16 feature L1 distance (perceptual loss) to L1 + (1-SSIM)
  --any-gap      train on ANY earlier photo -> any later photo of the same plant (input = one history frame, offset = target_day - that frame's day) instead of only
                 last photo -> next photo; built from the cleaned pair table (history lists post-scrub, so black / blurry frames never enter). Each epoch samples
                 --epoch-size such pairs (default = size of the standard train set, so epochs cost the same).

Everything else is Step C: same cleaned pairs and plant-wise split, L1 + (1-SSIM) loss, augmentation, optimizer, early stopping, model selection on the STANDARD
validation pairs (last photo -> next photo; the deployed task) by validation combined loss only, TEST evaluated exactly once. Test outputs: the usual per-pair CSV
(skimage structure SSIM / RGB SSIM / PSNR) plus SHARPNESS columns (see img_sharp.py) and a VGG feature distance; with --eval-any-gap also the test plants' any-gap pairs
(every earlier photo -> every later photo) scored in the same way, for the gap-stratified view. --smoke never reads the test split.

Usage:
    python src/train_img_sharp.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs --checkpoint-dir checkpoints \
        --device cuda --name residual_perc --residual --w-perc 1.0
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from img_film_unet import FiLMUNet, gaussian_window, ssim_torch
from img_sharp import ResFiLMUNet, VGGPerceptual, sharpness_table
from train_img_single_frame import (EXPECTED_COUNTS, augment, batches, figure, load_rows, plot_history, predict_uint8, psnr_torch,
                                    score_pairs, to_gpu)


def expand_any_gap(pairs, cache_idx):
    """Every (earlier history frame -> target) combination of the cleaned pairs, as a table with row_in / row_tg / offset."""
    rows = []
    for r in pairs.itertuples(index=False):
        for fp, d in zip(r.input_filepaths, r.input_days):
            rows.append((r.pair_id, r.plant_id, r.field, r.split, int(r.target_day) - int(d), cache_idx[fp], int(r.row_tg), int(d), int(r.target_day)))
    df = pd.DataFrame(rows, columns=["base_pair_id", "plant_id", "field", "split", "gap_days", "row_in", "row_tg", "input_last_day", "target_day"])
    df["pair_id"] = df["base_pair_id"] + "::in" + df["input_last_day"].astype(str)
    df["offset"] = df["gap_days"].astype(float)
    df["n_input_frames"] = 1
    return df


def run_epoch(model, vgg, arr, df, norm, args, device, win, optimizer=None, rng=None, n_samples=None):
    train = optimizer is not None
    model.train(train)
    offs = ((df["offset"].values - norm["mean"]) / norm["std"]).astype(np.float32)
    idx = rng.choice(len(df), size=n_samples, replace=False) if (train and n_samples) else np.arange(len(df))
    tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "psnr": 0.0, "perc": 0.0}
    n = 0
    for b0 in batches(len(idx), args.batch_size, train, rng):
        b = idx[b0]
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
            perc = vgg(p, y) if vgg is not None else torch.zeros((), device=device)
            loss = args.w_l1 * l1 + args.w_ssim * (1 - ss) + args.w_perc * perc
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        m = len(b)
        n += m
        for k, v in (("loss", loss), ("l1", l1), ("ssim", ss), ("perc", perc)):
            tot[k] += float(v) * m
        tot["psnr"] += psnr_torch(p.detach(), y).item() * m
    return {k: v / n for k, v in tot.items()}


def evaluate(model, vgg, arr, df, norm, device, tag_rows=True):
    """Predictions -> standard per-pair metrics + sharpness + VGG distance. Returns (per-pair DataFrame, sharpness summary)."""
    preds = predict_uint8(model, arr, df, norm, device)
    res = score_pairs(preds, arr, df)
    tg = np.stack([np.asarray(arr[int(r)]) for r in df["row_tg"]])
    sh = sharpness_table(preds, tg)
    res["grad_pred"], res["grad_target"], res["lap_pred"], res["lap_target"] = sh["grad_pred"], sh["grad_target"], sh["lap_pred"], sh["lap_target"]
    if vgg is not None:
        d = []
        for i in range(0, len(df), 32):
            p = torch.from_numpy(preds[i:i + 32]).to(device).permute(0, 3, 1, 2).float() / 255
            t = torch.from_numpy(tg[i:i + 32]).to(device).permute(0, 3, 1, 2).float() / 255
            with torch.no_grad():
                d.append(vgg(p, t, reduce=False).cpu().numpy())
        res["vgg_dist"] = np.concatenate(d)
    res["gap_days"] = df["gap_days"].values
    return res, preds, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--name", required=True, help="run name, used in every output file name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--w-perc", type=float, default=0.0)
    ap.add_argument("--any-gap", action="store_true")
    ap.add_argument("--epoch-size", type=int, default=0, help="pairs sampled per epoch with --any-gap (0 = size of the standard train set)")
    ap.add_argument("--eval-any-gap", action="store_true", help="also score the test plants' any-gap pairs (gap-stratified view)")
    ap.add_argument("--no-pretrained-vgg", dest="pretrained_vgg", action="store_false", help="smoke tests only")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-ssim", type=float, default=1.0)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tag = ("smoke_" if args.smoke else "") + f"img_sharp_{args.name}" + ("" if args.seed == 42 else f"_seed{args.seed}")

    pairs = load_rows(args.image_pairs, args.cache_dir)
    arr = np.load(os.path.join(args.cache_dir, "images256.npy"), mmap_mode="r")
    cache_idx = pd.read_parquet(os.path.join(args.cache_dir, "images256_index.parquet")).set_index("filepath")["row"].to_dict()
    counts = pairs["split"].value_counts().to_dict()
    print(f"Run '{args.name}': residual={args.residual} w_perc={args.w_perc} any_gap={args.any_gap} | pairs {counts}", flush=True)
    if not args.smoke:
        assert counts == EXPECTED_COUNTS, f"unexpected pair counts {counts}"
    tr_std = pairs[pairs.split == "train"].reset_index(drop=True)
    va = pairs[pairs.split == "val"].reset_index(drop=True)
    if args.smoke:
        tr_std, va = tr_std.sample(min(48, len(tr_std)), random_state=0).reset_index(drop=True), va.sample(min(16, len(va)), random_state=0).reset_index(drop=True)
        args.max_epochs = 2
    if args.any_gap:
        tr = expand_any_gap(pairs[pairs.split == "train"].loc[lambda d: d.pair_id.isin(tr_std.pair_id)] if args.smoke else pairs[pairs.split == "train"], cache_idx)
        n_epoch = args.epoch_size or len(tr_std)
        print(f"Any-gap training set: {len(tr)} (earlier photo -> later photo) pairs from {len(tr_std)} standard pairs; gaps {tr.gap_days.min()}-{tr.gap_days.max()} days "
              f"(median {int(tr.gap_days.median())}); {n_epoch} sampled per epoch", flush=True)
    else:
        tr, n_epoch = tr_std, None
    norm = {"mean": float(tr["offset"].mean()), "std": float(tr["offset"].std() or 1.0)}
    print(f"Offset normalization (train only): {norm}", flush=True)

    model = (ResFiLMUNet() if args.residual else FiLMUNet()).to(device)
    vgg = VGGPerceptual(pretrained=args.pretrained_vgg).to(device) if (args.w_perc > 0 or True) else None
    win = gaussian_window(device=device)
    vgg_train = vgg if args.w_perc > 0 else None
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | loss = {args.w_l1}*L1 + {args.w_ssim}*(1-SSIM) + {args.w_perc}*VGG", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    ckpt = os.path.join(args.checkpoint_dir, f"{tag}_best.pt")
    hist, best, bad = [], float("inf"), 0
    for ep in range(1, args.max_epochs + 1):
        t0 = time.time()
        trm = run_epoch(model, vgg_train, arr, tr, norm, args, device, win, optimizer=opt, rng=rng, n_samples=n_epoch)
        vam = run_epoch(model, vgg_train, arr, va, norm, args, device, win)
        sched.step(vam["loss"])
        hist.append({"epoch": ep, "train": trm, "val": vam, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0})
        improved = vam["loss"] < best
        if improved:
            best, bad = vam["loss"], 0
            torch.save({"model": model.state_dict(), "norm": norm, "epoch": ep, "args": vars(args)}, ckpt)
        else:
            bad += 1
        print(f"ep {ep:3d} | train loss {trm['loss']:.4f} (L1 {trm['l1']:.4f}, SSIM {trm['ssim']:.4f}, VGG {trm['perc']:.3f}) | "
              f"val loss {vam['loss']:.4f} (L1 {vam['l1']:.4f}, SSIM {vam['ssim']:.4f}, PSNR {vam['psnr']:.2f}, VGG {vam['perc']:.3f}) | "
              f"lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s {'*' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"Early stopping at epoch {ep}", flush=True)
            break
    json.dump(hist, open(os.path.join(args.out_dir, f"{tag}_history.json"), "w"), indent=1)
    plot_history(hist, os.path.join(args.out_dir, f"{tag}_curves.png"))
    best_ep = min(hist, key=lambda h: h["val"]["loss"])["epoch"]
    print(f"Selected checkpoint: epoch {best_ep} (best val combined loss {best:.4f}) -- chosen on VAL only", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    if args.smoke:
        ev, label = va, "SMOKE (val pairs; test not touched)"
    else:
        ev, label = pairs[pairs.split == "test"].reset_index(drop=True), "TEST EVALUATION"
    print(f"=== {label}: {len(ev)} pairs ===", flush=True)
    res, preds, sh = evaluate(model, vgg, arr, ev, norm, device)
    res.to_csv(os.path.join(args.out_dir, f"{tag}_per_pair.csv"), index=False)
    figure(preds, arr, ev, os.path.join(args.out_dir, f"{tag}_examples.png"))
    print(res[["ssim", "psnr", "structure_ssim", "vgg_dist"]].agg(["mean", "median"]).round(4).to_string(), flush=True)
    print(f"sharpness (prediction / real target): gradient-magnitude ratio {sh['grad_ratio']:.3f}, Laplacian-variance ratio {sh['lap_ratio']:.3f}", flush=True)
    if args.eval_any_gap and not args.smoke:
        te_all = pairs[pairs.split == "test"]
        eg = expand_any_gap(te_all, cache_idx)
        print(f"=== TEST any-gap evaluation: {len(eg)} (earlier photo -> later photo) pairs ===", flush=True)
        for c in ("n_input_frames",):
            eg[c] = 1
        res2, _, sh2 = evaluate(model, vgg, arr, eg, norm, device)
        res2["base_pair_id"] = eg["base_pair_id"].values
        res2.to_csv(os.path.join(args.out_dir, f"{tag}_anygap_per_pair.csv"), index=False)
        print(f"any-gap pooled: structure SSIM {res2.structure_ssim.mean():.4f}; sharpness ratios {sh2['grad_ratio']:.3f} / {sh2['lap_ratio']:.3f}", flush=True)
    json.dump({"best_epoch": best_ep, "best_val_loss": best, "norm": norm, "args": vars(args), "n_pairs_by_split": counts},
              open(os.path.join(args.out_dir, f"{tag}_run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
