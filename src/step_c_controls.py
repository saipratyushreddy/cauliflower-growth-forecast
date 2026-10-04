"""
Step C controls: closed-form, no training. Do the single-frame model's gains come from anything
beyond (a) blur and (b) learned per-flight exposure? Each control is evaluated on TEST exactly once;
sigma for control 1 is chosen on VALIDATION only.

  CONTROL 1  blurred copy-forward: Gaussian blur (spatial axes) of the last input image. Sigma = the
             value maximizing mean VALIDATION RGB SSIM (skimage, the evaluation metric) over a grid.
             Maximizing (not matching the model's val SSIM) is used because the model's val SSIM comes
             from a different (Gaussian-window, torch) SSIM, so it is not a clean target, and the
             maximum gives the control its best shot (conservative for any claim that the model
             beats a blur). The full val curve over sigma (RGB SSIM, PSNR, structure SSIM) is printed so
             the sensitivity of the DECIDING metric (structure SSIM) to this choice is visible without
             a second test evaluation.
  CONTROL 2  flight-aware colour copy-forward. For each (field, input_day, target_day) bucket, from
             TRAIN pairs only: M_t / S_t = mean over the bucket's training pairs of the TARGET image's
             per-channel mean / std. A test pair's copy-forward image x is standardized by its OWN
             (observable) per-channel mean/std and re-scaled to the bucket's (M_t, S_t):
             y = (x - m_x) / s_x * S_t + M_t. No test-target statistics are used (contrast with the
             oracle). Buckets with <10 training pairs fall back to (field, target_day); fallback pairs are
             flagged in the per-pair CSV (bucket_n, used_fallback).
  CONTROL 3  control 2's colour transform, then control 1's blur (sigma from control 1).

All outputs are rounded to uint8 and scored with the same skimage RGB SSIM/PSNR + structure SSIM as
everything else (train_img_single_frame.score_pairs).

Usage:
    python src/step_c_controls.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs \
        [--model-checkpoint checkpoints/img_single_frame_best.pt --model-per-pair-csv outputs/img_single_frame_per_pair.csv \
         --copy-forward-per-pair-csv outputs/img_stepB_copy_forward_per_pair.csv]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from image_copy_forward_baseline import structure_ssim
from summarize_baseline import HONEST, stats, strata
from train_img_single_frame import load_rows, score_pairs

SIGMAS = [0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
MIN_BUCKET = 10  # overridden by --min-bucket (smoke only)


def blur(img_u8, sigma):
    if sigma <= 0:
        return img_u8
    out = gaussian_filter(img_u8.astype(np.float32), sigma=(sigma, sigma, 0), mode="reflect")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def chan_stats(img_u8):
    f = img_u8.astype(np.float64)
    return f.mean((0, 1)), f.std((0, 1))


def fit_buckets(arr, train):
    """Per-bucket mean of TARGET per-channel mean/std, from training pairs only."""
    tm, ts = [], []
    for r in train["row_tg"].values:
        m, s = chan_stats(arr[int(r)])
        tm.append(m), ts.append(s)
    t = train[["field", "input_last_day", "target_day"]].copy()
    t[["mr", "mg", "mb"]] = np.array(tm)
    t[["sr", "sg", "sb"]] = np.array(ts)

    def agg(keys):
        g = t.groupby(keys)
        return {k: (np.array([v[["mr", "mg", "mb"]].mean().values, v[["sr", "sg", "sb"]].mean().values]), len(v))
                for k, v in g}
    return agg(["field", "input_last_day", "target_day"]), agg(["field", "target_day"])


def colour_transform(x_u8, M, S):
    m, s = chan_stats(x_u8)
    y = (x_u8.astype(np.float64) - m) / np.maximum(s, 1e-6) * S + M
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def control2_pred(arr, r, buckets, fallback):
    key = (r.field, r.input_last_day, r.target_day)
    st, n = buckets.get(key, (None, 0))
    used_fb = n < MIN_BUCKET
    if used_fb:
        st, n = fallback.get((r.field, r.target_day), (None, 0))
        assert st is not None and n >= MIN_BUCKET, f"no usable bucket for {key} even at (field,target_day)"
    return colour_transform(arr[int(r.row_in)], st[0], st[1]), n, used_fb


def val_curve(arr, va):
    rows = []
    for s in [0] + SIGMAS:
        ss, ps, st = [], [], []
        for r in va.itertuples(index=False):
            p, t = blur(arr[int(r.row_in)], s), arr[int(r.row_tg)]
            ss.append(structural_similarity(p, t, channel_axis=2, data_range=255))
            ps.append(peak_signal_noise_ratio(p, t, data_range=255))
            st.append(structure_ssim(p, t))
        rows.append({"sigma": s, "val_ssim": np.mean(ss), "val_psnr": np.mean(ps), "val_structure_ssim": np.mean(st)})
        print(f"  sigma {s:5.1f} | val SSIM {rows[-1]['val_ssim']:.4f} | PSNR {rows[-1]['val_psnr']:.2f} | "
              f"structure SSIM {rows[-1]['val_structure_ssim']:.4f}", flush=True)
    return pd.DataFrame(rows)


def master_table(frames, test_pairs):
    """All methods x strata, mean (median), for the three metrics."""
    lines = []
    extra = test_pairs[["pair_id", "input_last_day", "target_day", "d28"]]
    for metric, nd in [("structure_ssim", 3), ("ssim", 3), ("psnr", 2)]:
        title = {"structure_ssim": "STRUCTURE SSIM (the deciding metric)", "ssim": "RGB SSIM", "psnr": "RGB PSNR dB"}[metric]
        names = list(frames)
        lines.append(f"\n{title}, test mean (median)")
        lines.append("| Subset | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        per = {}
        for n, df in frames.items():
            per[n] = strata(df.drop(columns=[c for c in ("input_last_day", "target_day", "d28") if c in df])
                              .merge(extra, on="pair_id"))
        for sub in per[names[0]]:
            lines.append(f"| {sub.strip()} | " + " | ".join(
                f"{per[n][sub][metric].mean():.{nd}f} ({per[n][sub][metric].median():.{nd}f})" for n in names) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-checkpoint", default=None)
    ap.add_argument("--model-per-pair-csv", default=None)
    ap.add_argument("--copy-forward-per-pair-csv", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--min-bucket", type=int, default=10)
    args = ap.parse_args()
    global MIN_BUCKET
    MIN_BUCKET = args.min_bucket
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "smoke_" if args.smoke else ""

    pairs = load_rows(args.image_pairs, args.cache_dir)
    arr = np.load(os.path.join(args.cache_dir, "images256.npy"))
    tr = pairs[pairs.split == "train"].reset_index(drop=True)
    va = pairs[pairs.split == "val"].reset_index(drop=True)
    te = pairs[pairs.split == "test"].reset_index(drop=True)
    if args.smoke:
        va = va.head(12).reset_index(drop=True)
    print(f"Pairs: train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    # ---- Control 1: pick sigma on VAL only ----
    print("Control 1: validation sweep over blur sigma", flush=True)
    curve = val_curve(arr, va)
    curve.to_csv(os.path.join(args.out_dir, f"{tag}img_ctrl1_val_sigma_curve.csv"), index=False)
    best = curve.loc[curve[curve.sigma > 0]["val_ssim"].idxmax()]
    sigma = float(best["sigma"])
    sbest = float(curve.loc[curve["val_structure_ssim"].idxmax(), "sigma"])
    if sigma == max(SIGMAS):
        print("WARNING: val SSIM is still rising at the largest sigma in the grid; the optimum may be beyond it.", flush=True)
    print(f"Control 1 sigma = {sigma} (max val RGB SSIM {best['val_ssim']:.4f}). "
          f"[Sensitivity: val structure SSIM is maximal at sigma = {sbest}; not evaluated on test.]", flush=True)

    # ---- Control 2 fit (train only) ----
    buckets, fallback = fit_buckets(arr, tr)
    print(f"Control 2: {len(buckets)} (field,input_day,target_day) buckets from train; "
          f"sizes min {min(n for _, n in buckets.values())} median {int(np.median([n for _, n in buckets.values()]))}", flush=True)

    # ---- Single test evaluation per control ----
    preds = {"ctrl1": [], "ctrl2": [], "ctrl3": []}
    meta2 = []
    for r in te.itertuples(index=False):
        x = arr[int(r.row_in)]
        preds["ctrl1"].append(blur(x, sigma))
        p2, n, fb = control2_pred(arr, r, buckets, fallback)
        preds["ctrl2"].append(p2)
        preds["ctrl3"].append(blur(p2, sigma))
        meta2.append((n, fb))
    out = {"sigma": sigma, "sigma_selected_on": "validation RGB SSIM (skimage), grid " + str(SIGMAS),
           "val_structure_ssim_argmax_sigma": sbest}
    res = {}
    for k, label in [("ctrl1", "CONTROL 1 blurred copy-forward"), ("ctrl2", "CONTROL 2 flight-aware colour copy-forward"),
                     ("ctrl3", "CONTROL 3 colour + blur")]:
        print(f"=== TEST EVALUATION ({label}): {len(te)} pairs ===", flush=True)
        df = score_pairs(np.stack(preds[k]), arr, te)
        if k != "ctrl1":
            df["bucket_n"] = [m[0] for m in meta2]
            df["used_fallback"] = [m[1] for m in meta2]
        df.to_csv(os.path.join(args.out_dir, f"{tag}img_{k}_per_pair.csv"), index=False)
        res[k] = df
        print(df[["ssim", "psnr", "structure_ssim"]].agg(["mean", "median"]).round(4).to_string(), flush=True)
    m2 = res["ctrl2"]
    fb = m2[m2.used_fallback]
    out["ctrl2_bucket_n_per_test_pair"] = {"min": int(m2.bucket_n.min()), "median": float(m2.bucket_n.median()),
                                           "max": int(m2.bucket_n.max())}
    out["ctrl2_test_pairs_using_fallback"] = int(len(fb))
    print(f"Control 2 bucket training-pair count per test pair: min {m2.bucket_n.min()}, median "
          f"{m2.bucket_n.median():.0f}, max {m2.bucket_n.max()}; fallback to (field,target_day) used for "
          f"{len(fb)} of {len(m2)} test pairs", flush=True)
    if len(fb):
        print(fb.merge(te[["pair_id", "input_last_day", "target_day"]], on="pair_id")
                .groupby(["field", "input_last_day", "target_day"]).size().rename("n_test_pairs").to_string(), flush=True)
    json.dump(out, open(os.path.join(args.out_dir, f"{tag}img_ctrl_run.json"), "w"), indent=1)

    # ---- Master comparison with copy-forward and the trained model ----
    te_d = te[["pair_id", "input_last_day", "target_day", "field"]].copy()
    te_d["d28"] = (te_d.field == "Field1") & ((te_d.input_last_day == 28) | (te_d.target_day == 28))
    frames = {}
    if args.copy_forward_per_pair_csv:
        cf = pd.read_csv(args.copy_forward_per_pair_csv)
        frames["copy-fwd"] = cf[cf.pair_id.isin(te.pair_id)].reset_index(drop=True)
    frames.update({f"C1 blur s={sigma:g}": res["ctrl1"], "C2 flight-colour": res["ctrl2"], "C3 colour+blur": res["ctrl3"]})
    model = None
    if args.model_per_pair_csv:
        frames["MODEL (Step C)"] = pd.read_csv(args.model_per_pair_csv)
    print(master_table(frames, te_d))

    if "MODEL (Step C)" in frames:
        print("\nPAIRED: model - control, structure SSIM (the deciding comparison), test; win = share of pairs the model is better")
        print("| Subset | Pairs | vs copy-fwd | vs C1 | vs C2 | vs C3 |\n|---|---|---|---|---|---|")
        mdl = frames["MODEL (Step C)"].set_index("pair_id")
        per = {}
        for n in ["copy-fwd", "C1 blur s=%g" % sigma, "C2 flight-colour", "C3 colour+blur"]:
            if n not in frames:
                continue
            f = frames[n].set_index("pair_id")
            d = (mdl.loc[te.pair_id, "structure_ssim"] - f.loc[te.pair_id, "structure_ssim"]).reset_index()
            d.columns = ["pair_id", "d"]
            per[n] = d.merge(te_d, on="pair_id")
        def sub_strata(d):
            return strata(d.assign(plant_id=d.pair_id.str.split("::").str[0]))
        subs = sub_strata(per[list(per)[0]])
        for sub in subs:
            cells = []
            for n, d in per.items():
                g = sub_strata(d)[sub]
                cells.append(f"{g['d'].mean():+.3f} ({g['d'].median():+.3f}), win {(g['d'] > 0).mean():.0%}")
            print(f"| {sub.strip()} | {len(subs[sub]):,} | " + " | ".join(cells) + " |")

    # ---- Visual grid ----
    rng = np.random.RandomState(0)
    pick = []

    def choose(mask, fallback=None):
        ix = [i for i in np.where(mask)[0] if i not in pick]
        if not ix and fallback is not None:
            ix = [i for i in np.where(fallback)[0] if i not in pick]
        pick.append(int(rng.choice(ix)))

    d28t = ((te.field == "Field1") & (te.target_day == 28)).values            # day-28 TARGET pair (required)
    d28i = ((te.field == "Field1") & (te.input_last_day == 28)).values
    f1 = ((te.field == "Field1") & ~d28t & ~d28i).values
    f2 = (te.field == "Field2").values
    choose(d28t)
    choose(f1)
    choose(f2 & (te.target_day.values < 40), f2)    # early-season Field2
    choose(f2 & (te.target_day.values >= 65), f2)   # late-season Field2
    cols = ["input", "target", "copy-fwd", f"C1 blur s={sigma:g}", "C2 flight-colour", "C3 colour+blur"] + (["MODEL"] if args.model_checkpoint else [])
    model_preds = None
    if args.model_checkpoint:
        import torch
        from img_film_unet import FiLMUNet
        from train_img_single_frame import predict_uint8
        sd = torch.load(args.model_checkpoint, map_location="cpu")
        mdl_net = FiLMUNet(); mdl_net.load_state_dict(sd["model"])
        sub = te.iloc[pick].reset_index(drop=True)
        model_preds = predict_uint8(mdl_net, arr, sub, sd["norm"], torch.device("cpu"))
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(2.2 * len(cols), 2.5 * len(pick)))
    for i, ix in enumerate(pick):
        r = te.iloc[ix]
        row = [arr[int(r.row_in)], arr[int(r.row_tg)], arr[int(r.row_in)], preds["ctrl1"][ix], preds["ctrl2"][ix], preds["ctrl3"][ix]]
        if model_preds is not None:
            row.append(model_preds[i])
        for j, (ax, im) in enumerate(zip(axes[i], row)):
            ax.imshow(im); ax.axis("off")
            if i == 0:
                ax.set_title(cols[j], fontsize=8)
        axes[i][0].set_title(f"{cols[0] if i == 0 else ''}\n{r.plant_id.replace('_Ref_', '_')} d{int(r.input_last_day)}->d{int(r.target_day)}",
                             fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{tag}img_ctrl_grid.png"), dpi=90)
    print("Wrote grid; pairs:", [te.pair_id.iloc[i] for i in pick])


if __name__ == "__main__":
    main()
