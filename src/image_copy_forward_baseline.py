"""
Image-prediction track, Step B: naive copy-forward baseline, plus a
colour-matched variant.

Prediction for a pair is the plant's most recent input image, unchanged
(y_hat(t+1) = image(t)). Scored against the true next image with SSIM and PSNR.

Colour-matched copy-forward (cm_*) is an ORACLE reference, not a deployable
predictor: each channel of the input is shifted/scaled to the TARGET's per-channel
mean and std (it peeks at the target's statistics). It estimates how much of the
raw copy-forward error is global exposure/colour change between flights rather
than growth or misregistration; a future model should be judged against both.

All images are resized to 256x256 (Lanczos) BEFORE scoring. Field1/2020 images
are natively 490x490 and are DOWNSAMPLED; Field2/2021 images are natively
256x256 and are untouched. Downsampling rather than upsampling Field2 means no
detail is synthesized that was not in the source images, which would
make SSIM/PSNR comparisons misleading. The same preprocessing must be used by
every later model so metrics are comparable.

No training and no model selection happens here, so val and test are both
reported; test is the headline.

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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SIZE = 256


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


def summarize(df):
    out = {"n_pairs": int(len(df)), "n_plants": int(df["plant_id"].nunique())}
    for k in ["ssim", "psnr", "cm_ssim", "cm_psnr"]:
        out[f"{k}_mean"] = float(df[k].mean())
        out[f"{k}_median"] = float(df[k].median())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)
    native_sizes = Counter()
    cache = {}  # tiny per-plant cache: consecutive pairs share an image
    rows = []
    for i, r in enumerate(pairs.itertuples(index=False)):
        try:
            a = cache.pop(r.input_last_filepath, None)
            if a is None:
                a = load(args.images_root, r.input_last_filepath, native_sizes)
            b = load(args.images_root, r.target_filepath, native_sizes)
        except Exception as e:
            print(f"FAILED {r.pair_id}: {e}")
            continue
        cache = {r.target_filepath: b}
        c = color_match(a, b)
        rows.append({"pair_id": r.pair_id, "plant_id": r.plant_id, "field": r.field,
                     "split": r.split, "gap_days": r.gap_days, "n_input_frames": r.n_input_frames,
                     "ssim": structural_similarity(a, b, channel_axis=2, data_range=255),
                     "psnr": peak_signal_noise_ratio(a, b, data_range=255),
                     "cm_ssim": structural_similarity(c, b, channel_axis=2, data_range=255),
                     "cm_psnr": peak_signal_noise_ratio(c, b, data_range=255)})
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(pairs)}")

    res = pd.DataFrame(rows)
    for k in ["psnr", "cm_psnr"]:
        res[k] = res[k].replace(np.inf, np.nan)  # identical images
    print(f"Scored {len(res)}/{len(pairs)} pairs. Native image sizes seen: {dict(native_sizes)}")

    out = {"resize": f"{SIZE}x{SIZE} Lanczos (Field1 490->256 downsampled; Field2 native)",
           "n_failed": int(len(pairs) - len(res)), "native_sizes_loaded": {str(k): v for k, v in native_sizes.items()},
           "by_split": {s: summarize(res[res.split == s]) for s in ["train", "val", "test"]},
           "test_by_field": {f: summarize(g) for f, g in res[res.split == "test"].groupby("field")},
           "test_by_gap": {str(k): summarize(g) for k, g in
                           res[res.split == "test"].groupby(pd.cut(res[res.split == "test"].gap_days, [0, 4, 7, 10, 100]),
                                                           observed=True)}}
    os.makedirs(args.out_dir, exist_ok=True)
    res.to_csv(os.path.join(args.out_dir, "img_stepB_copy_forward_per_pair.csv"), index=False)
    with open(os.path.join(args.out_dir, "img_stepB_copy_forward_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
