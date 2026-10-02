"""
Cache every image referenced by the CLEANED pair set (inputs incl. history, and targets)
as one uint8 array at 256x256 (Lanczos downsample for Field1 490->256, Field2 untouched --
the same preprocessing as the Step B baseline), so training epochs never touch Lustre.

Outputs:
    <out-dir>/images256.npy         uint8 [N, 256, 256, 3]
    <out-dir>/images256_index.parquet   filepath -> row

Idempotent: rebuilds only if the set of filepaths differs from the existing index.

Usage:
    python src/cache_images_256.py --image-pairs data/image_pairs.parquet \
        --images-root $WORK/cauliflower-growth-forecast/data/images --out-dir data
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

SIZE = 256


def load(images_root, fp):
    with Image.open(os.path.join(images_root, fp)) as im:
        im = im.convert("RGB")
        if im.size != (SIZE, SIZE):
            im = im.resize((SIZE, SIZE), Image.LANCZOS)
        return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pairs = pd.read_parquet(args.image_pairs)
    fps = set(pairs["target_filepath"]) | set(pairs["input_last_filepath"])
    for l in pairs["input_filepaths"]:
        fps.update(l)
    fps = sorted(fps)
    npy = os.path.join(args.out_dir, "images256.npy")
    idx_path = os.path.join(args.out_dir, "images256_index.parquet")
    if os.path.exists(npy) and os.path.exists(idx_path):
        old = pd.read_parquet(idx_path)
        if list(old["filepath"]) == fps:
            print(f"Cache already up to date ({len(fps)} images); nothing to do.")
            return
    print(f"Caching {len(fps)} images -> {npy} ({len(fps) * SIZE * SIZE * 3 / 1e9:.2f} GB)", flush=True)
    arr = np.lib.format.open_memmap(npy + ".tmp", mode="w+", dtype=np.uint8, shape=(len(fps), SIZE, SIZE, 3))
    for i, fp in enumerate(fps):
        arr[i] = load(args.images_root, fp)
        if (i + 1) % 1000 == 0:
            print(f"  ...{i + 1}/{len(fps)}", flush=True)
    arr.flush()
    del arr
    os.replace(npy + ".tmp", npy)
    pd.DataFrame({"filepath": fps, "row": np.arange(len(fps))}).to_parquet(idx_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
