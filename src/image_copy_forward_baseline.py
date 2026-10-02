"""
Image-prediction track, Step B: naive copy-forward baseline (honest), plus two
clearly separated reference points.

HONEST BASELINE ("copy_forward"): prediction for a pair is the plant's most
recent input image, unchanged (y_hat(t+1) = image(t)). Metrics: RGB SSIM, RGB
PSNR, and structure-only SSIM.

STRUCTURE-ONLY SSIM ("structure_ssim"): both images -> grayscale (luma) ->
local contrast normalization (subtract Gaussian-weighted local mean, divide by
local std + eps, sigma=LCN_SIGMA px, values clipped to +-3) -> SSIM with
data_range=6. Removes global exposure/colour shifts and local shading so the
score reflects spatial structure. eps stops flat regions from being amplified
into noise. Only scikit-image/scipy are used.

ORACLE ("oracle_colormatch"): NOT a baseline and NOT achievable at inference. Each
channel of the copy-forward prediction is shifted/scaled to the TARGET's
per-channel mean and std, i.e. it USES TARGET STATISTICS. It only estimates how
much raw copy-forward error is global exposure/colour change between flights.
It is stored under its own JSON key and its own CSV columns and must always be
shown with that caveat, never alongside baselines/trained models as if comparable.

All images are resized to 256x256 (Lanczos) BEFORE scoring. Field1/2020 images
are natively 490x490 and are DOWNSAMPLED; Field2/2021 images are natively
256x256 and untouched. Downsampling rather than upsampling Field2 means no
detail is synthesized that was not in the source images, which would make
SSIM/PSNR comparisons misleading. Every later model must use the same
preprocessing.

No training and no model selection happens here, so val and test are both
reported; test is the headline. Pooled and per-field are co-equal results.

Usage:
    python src/image_copy_forward_baseline.py --image-pairs data/image_pairs.parquet \
        --images-root $WORK/cauliflower-growth-forecast/data/images --out-dir outputs
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SIZE = 256
LCN_SIGMA = 7.0
LCN_EPS = 0.05  # on luma in [0,1]
ORACLE_CAVEAT = ("ORACLE: uses the TARGET image's per-channel mean/std. Not achievable at "
                 "inference; not comparable to the honest baseline or any trained model.")


def load(images_root, fp, native_sizes):
    with Image.open(os.path.join(images_root, fp)) as im:
        im = im.convert("RGB")
        native_sizes[im.size] += 1
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.LANCZOS)
        return np.asarray(im)


def color_match(a, b):
    """Per-channel mean/std match of a to b (oracle)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ma, sa = a.mean((0, 1)), a.std((0, 1))
    mb, sb = b.mean((0, 1)), b.std((0, 1))
    scale = np.where(sa > 1e-6, sb / np.maximum(sa, 1e-6), 1.0)
    return np.clip((a - ma) * scale + mb, 0, 255).round().astype(np.uint8)


def lcn_gray(img):
    g = np.asarray(Image.fromarray(img).convert("L"), dtype=np.float64) / 255.0
    mu = gaussian_filter(g, LCN_SIGMA)
    var = gaussian_filter((g - mu) ** 2, LCN_SIGMA)
    return np.clip((g - mu) / (np.sqrt(var) + LCN_EPS), -3, 3)


def structure_ssim(a, b):
    return structural_similarity(lcn_gray(a), lcn_gray(b), data_range=6.0)


def _stats(df, cols):
    out = {"n_pairs": int(len(df)), "n_plants": int(df["plant_id"].nunique())}
    for k, col in cols.items():
        out[f"{k}_mean"] = float(df[col].mean())
        out[f"{k}_median"] = float(df[col].median())
    return out


HONEST = {"ssim": "ssim", "psnr": "psnr", "structure_ssim": "structure_ssim"}
ORACLE = {"oracle_colormatch_ssim": "oracle_colormatch_ssim",
          "oracle_colormatch_psnr": "oracle_colormatch_psnr"}


def breakdown(res, cols):
    test = res[res.split == "test"]
    gap = pd.cut(test["gap_days"], [0, 4, 7, 10, 100])
    return {"by_split": {s: _stats(res[res.split == s], cols) for s in ["train", "val", "test"]},
            "by_split_field": {s: {f: _stats(g, cols) for f, g in res[res.split == s].groupby("field")}
                               for s in ["train", "val", "test"]},
            "test_by_gap": {str(k): _stats(g, cols) for k, g in test.groupby(gap, observed=True)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)
    native_sizes = Counter()
    cache = {}  # tiny cache: consecutive pairs share an image
    rows = []
    for i, r in enumerate(pairs.itertuples(index=False)):
        try:
            a = cache.pop(r.input_last_filepath, None)
            if a is None:
                a = load(args.images_root, r.input_last_filepath, native_sizes)
            b = load(args.images_root, r.target_filepath, native_sizes)
        except Exception as e:
            print(f"FAILED {r.pair_id}: {e}", flush=True)
            continue
        cache = {r.target_filepath: b}
        c = color_match(a, b)  # oracle: uses target statistics
        rows.append({"pair_id": r.pair_id, "plant_id": r.plant_id, "field": r.field,
                     "split": r.split, "gap_days": r.gap_days, "n_input_frames": r.n_input_frames,
                     "ssim": structural_similarity(a, b, channel_axis=2, data_range=255),
                     "psnr": peak_signal_noise_ratio(a, b, data_range=255),
                     "structure_ssim": structure_ssim(a, b),
                     "oracle_colormatch_ssim": structural_similarity(c, b, channel_axis=2, data_range=255),
                     "oracle_colormatch_psnr": peak_signal_noise_ratio(c, b, data_range=255)})
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(pairs)}", flush=True)

    res = pd.DataFrame(rows)
    for k in ["psnr", "oracle_colormatch_psnr"]:
        res[k] = res[k].replace(np.inf, np.nan)  # identical images
    print(f"Scored {len(res)}/{len(pairs)} pairs. Native image sizes seen: {dict(native_sizes)}", flush=True)

    out = {"resize": f"{SIZE}x{SIZE} Lanczos (Field1 490->256 downsampled; Field2 native)",
           "n_failed": int(len(pairs) - len(res)),
           "native_sizes_loaded": {str(k): v for k, v in native_sizes.items()},
           "structure_ssim_definition": f"luma, local contrast norm (gaussian sigma={LCN_SIGMA}, "
                                        f"eps={LCN_EPS}, clip +-3), SSIM data_range=6",
           "copy_forward": breakdown(res, HONEST),
           "oracle_colormatch": {"CAVEAT": ORACLE_CAVEAT, **breakdown(res, ORACLE)}}
    os.makedirs(args.out_dir, exist_ok=True)
    res.to_csv(os.path.join(args.out_dir, "img_stepB_copy_forward_per_pair.csv"), index=False)
    with open(os.path.join(args.out_dir, "img_stepB_copy_forward_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
