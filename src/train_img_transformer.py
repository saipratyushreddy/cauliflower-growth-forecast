"""
Image-track CNN-Transformer (see img_transformer.py): one token per frame from the shared bottleneck encoder, continuous
sinusoidal encoding of (target_day - input_day), full available history (up to 14 frames, variable length, padding mask),
attention-pooled bottleneck AND skips for the decoder.

Same pair set, L1 + SSIM loss, optimizer, augmentation (applied identically to every frame and the target), early stopping and
model-selection discipline as Step C / ConvLSTM: best epoch by VALIDATION combined loss only; TEST evaluated exactly once at the
end for that checkpoint; --smoke never reads the test split. Also writes the attention weights per evaluated pair
(<tag>_attn.csv) to show how much weight falls on the last frame vs earlier history.

Usage:
    python src/train_img_transformer.py --image-pairs data/image_pairs.parquet --cache-dir data \
        --out-dir outputs --checkpoint-dir checkpoints --device cuda
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from img_film_unet import gaussian_window, ssim_torch
from img_transformer import TemporalTransformerUNet
from train_img_single_frame import (EXPECTED_COUNTS, batches, figure, load_rows, plot_history, psnr_torch,
                                    score_pairs)


def build_histories(pairs, cache_idx):
    """Full (post-scrub) history per pair: cache rows, and offsets = target_day - input_day for each frame."""
    rows, offs = [], []
    for r in pairs.itertuples(index=False):
        fps, days = list(r.input_filepaths), list(r.input_days)
        assert fps[-1] == r.input_last_filepath and days[-1] == r.input_last_day
        rows.append(np.array([cache_idx[f] for f in fps], dtype=np.int64))
        offs.append(np.array([r.target_day - d for d in days], dtype=np.float32))
    return rows, offs


def make_batch(arr, rows, offs, idx, device):
    counts = np.array([len(rows[i]) for i in idx])
    flat = np.concatenate([rows[i] for i in idx])
    x = torch.from_numpy(arr[flat]).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    b_idx = torch.from_numpy(np.repeat(np.arange(len(idx)), counts)).to(device)
    t_idx = torch.from_numpy(np.concatenate([np.arange(c) for c in counts])).to(device)
    off = torch.from_numpy(np.concatenate([offs[i] for i in idx])).to(device)
    return x, b_idx, t_idx, off, len(idx), int(counts.max()), counts


def augment_flat(x, y, counts):
    s = 0
    for i, c in enumerate(counts):
        k = int(torch.randint(0, 4, (1,)))
        flip = bool(torch.rand(1) < 0.5)
        seg = x[s:s + c]
        if k:
            seg = torch.rot90(seg, k, (2, 3))
            y[i] = torch.rot90(y[i], k, (1, 2))
        if flip:
            seg = seg.flip(3)
            y[i] = y[i].flip(2)
        x[s:s + c] = seg
        s += c
    return x, y


def run_epoch(model, arr, df, H, args, device, win, optimizer=None, rng=None):
    train = optimizer is not None
    model.train(train)
    tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "psnr": 0.0}
    n = 0
    for b in batches(len(df), args.batch_size, train, rng):
        x, b_idx, t_idx, off, B, Kmax, counts = make_batch(arr, H[0], H[1], b, device)
        y = torch.from_numpy(arr[df["row_tg"].values[b]]).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        if train:
            x, y = augment_flat(x, y, counts)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.amp)):
                p = model(x, b_idx, t_idx, off, B, Kmax)
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
def predict(model, arr, df, H, device, bs=16):
    model.eval()
    out, wts = [], []
    for b in batches(len(df), bs, False, None):
        x, b_idx, t_idx, off, B, Kmax, counts = make_batch(arr, H[0], H[1], b, device)
        p, w = model(x, b_idx, t_idx, off, B, Kmax, return_weights=True)
        out.append((p.float().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        for i, c in enumerate(counts):
            wts.append(w[i, :c].float().cpu().numpy())
    return np.concatenate(out), wts


def attn_table(df, wts):
    rows = []
    for r, w in zip(df.itertuples(index=False), wts):
        n = len(w)
        ent = float(-(w * np.log(np.maximum(w, 1e-12))).sum())
        rows.append({"pair_id": r.pair_id, "field": r.field, "n_frames": n, "w_last": float(w[-1]),
                     "w_first": float(w[0]), "w_max": float(w.max()), "eff_frames": float(np.exp(ent)),
                     "uniform_w": 1.0 / n})
    return pd.DataFrame(rows)


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
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dim-ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="tiny run, never reads the test split")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tag = f"{'smoke_' if args.smoke else ''}img_transformer" + ("" if args.seed == 42 else f"_seed{args.seed}")

    pairs = load_rows(args.image_pairs, args.cache_dir)
    arr = np.load(os.path.join(args.cache_dir, "images256.npy"))
    cache_idx = pd.read_parquet(os.path.join(args.cache_dir, "images256_index.parquet")).set_index("filepath")["row"].to_dict()
    counts = pairs["split"].value_counts().to_dict()
    print(f"Pairs used: {counts} | images cached: {arr.shape} | history length min/median/max = "
          f"{pairs.n_input_frames.min()}/{int(pairs.n_input_frames.median())}/{pairs.n_input_frames.max()}", flush=True)
    if not args.smoke:
        assert counts == EXPECTED_COUNTS, f"unexpected pair counts {counts}"
    tr = pairs[pairs.split == "train"].reset_index(drop=True)
    va = pairs[pairs.split == "val"].reset_index(drop=True)
    if args.smoke:
        # smoke subset deliberately spans short (1-2 frames) and long (10+ frames) histories
        def span(d, k):
            n = d.n_input_frames
            parts = [d[n == 1].head(k), d[n == 2].head(k), d[(n >= 4) & (n <= 6)].head(k), d[n >= 10].head(k)]
            return pd.concat(parts).reset_index(drop=True)
        tr, va = span(tr, 3), span(va, 1)
        args.max_epochs = 2
    Htr, Hva = build_histories(tr, cache_idx), build_histories(va, cache_idx)
    lens = np.array([len(r) for r in Htr[0]])
    print(f"Train pairs: history length distribution {dict(zip(*[x.tolist() for x in np.unique(lens, return_counts=True)]))}", flush=True)

    model = TemporalTransformerUNet(d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                                    dim_ff=args.dim_ff, dropout=args.dropout, grad_checkpoint=args.grad_checkpoint).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | device {device} | "
          f"d_model {args.d_model}, heads {args.nhead}, layers {args.num_layers}, dropout {args.dropout} | "
          f"loss = {args.w_l1}*L1 + {args.w_ssim}*(1-SSIM)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    win = gaussian_window(device=device)

    ckpt = os.path.join(args.checkpoint_dir, f"{tag}_best.pt")
    hist, best, bad = [], float("inf"), 0
    for ep in range(1, args.max_epochs + 1):
        t0 = time.time()
        trm = run_epoch(model, arr, tr, Htr, args, device, win, optimizer=opt, rng=rng)
        vam = run_epoch(model, arr, va, Hva, args, device, win)
        sched.step(vam["loss"])
        hist.append({"epoch": ep, "train": trm, "val": vam, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0})
        improved = vam["loss"] < best
        if improved:
            best, bad = vam["loss"], 0
            torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(args)}, ckpt)
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
        ev, Hev, label = va, Hva, "SMOKE (val pairs; test not touched)"
    else:
        ev, label = pairs[pairs.split == "test"].reset_index(drop=True), "TEST EVALUATION"
        Hev = build_histories(ev, cache_idx)
    print(f"=== {label}: {len(ev)} pairs ===", flush=True)
    preds, wts = predict(model, arr, ev, Hev, device)
    res = score_pairs(preds, arr, ev)
    res.to_csv(os.path.join(args.out_dir, f"{tag}_per_pair.csv"), index=False)
    at = attn_table(ev, wts)
    at.to_csv(os.path.join(args.out_dir, f"{tag}_attn.csv"), index=False)
    figure(preds, arr, ev, os.path.join(args.out_dir, f"{tag}_examples.png"))
    print(res[["ssim", "psnr", "structure_ssim"]].agg(["mean", "median"]).round(4).to_string(), flush=True)
    g = at.groupby(pd.cut(at.n_frames, [0, 1, 2, 3, 5, 9, 14], labels=["1", "2", "3", "4-5", "6-9", "10+"]), observed=True)
    print("Attention on the LAST frame by history length (uniform weight would be 1/n):")
    print(g.agg(pairs=("pair_id", "size"), w_last=("w_last", "mean"), uniform=("uniform_w", "mean"),
                eff_frames=("eff_frames", "mean")).round(3).to_string(), flush=True)
    json.dump({"best_epoch": best_ep, "best_val_loss": best, "args": vars(args), "n_pairs_by_split": counts},
              open(os.path.join(args.out_dir, f"{tag}_run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
