"""
Run ON SWAN (in the ORIGINAL project directory, the one whose checkpoints produced the README numbers) to build the self-contained bundle the
demo UI needs on a laptop. Reads only; writes <out>/ :

    image_pairs_test.parquet                 the cleaned TEST pairs (1,233 rows; same columns as data/image_pairs.parquet)
    cache/images256.npy, images256_index.parquet   only the 256x256 uint8 images those pairs use (inputs incl. history + targets), a subset of the
                                             verified data/images256.npy, rows checked byte-for-byte against it
    checkpoints/img_transformer_best.pt      frozen full-history Transformer (README numbers)
    checkpoints/img_convlstm_k4_best.pt      frozen ConvLSTM K=4 seed 42 (README numbers)
    MANIFEST.json                            sha256 of every file, counts, code commit

Usage (from the project directory on Swan):
    python scripts/export_demo_bundle.py --out demo_bundle
Then copy the folder to the laptop, e.g.:  rsync -av swan:<project dir>/demo_bundle/ ~/projects/cauliflower-growth-forecast/demo_bundle/
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=".")
    ap.add_argument("--out", default="demo_bundle")
    args = ap.parse_args()
    P, O = args.project_dir, args.out
    os.makedirs(os.path.join(O, "cache"), exist_ok=True)
    os.makedirs(os.path.join(O, "checkpoints"), exist_ok=True)

    pairs = pd.read_parquet(os.path.join(P, "data/image_pairs.parquet"))
    test = pairs[pairs.split == "test"].reset_index(drop=True)
    assert len(test) == 1233 and test.plant_id.nunique() == 110, f"unexpected test split: {len(test)} pairs, {test.plant_id.nunique()} plants"
    fps = set(test.target_filepath) | set(test.input_last_filepath)
    for l in test.input_filepaths:
        fps.update(l)
    fps = sorted(fps)
    full_idx = pd.read_parquet(os.path.join(P, "data/images256_index.parquet")).set_index("filepath")["row"]
    full = np.load(os.path.join(P, "data/images256.npy"), mmap_mode="r")
    sub = np.lib.format.open_memmap(os.path.join(O, "cache/images256.npy"), mode="w+", dtype=np.uint8, shape=(len(fps), 256, 256, 3))
    for i, fp in enumerate(fps):
        sub[i] = full[int(full_idx[fp])]
    sub.flush()
    chk = np.load(os.path.join(O, "cache/images256.npy"), mmap_mode="r")
    assert all(np.array_equal(chk[i], full[int(full_idx[fp])]) for i, fp in enumerate(fps)), "subset differs from the verified cache"
    pd.DataFrame({"filepath": fps, "row": np.arange(len(fps))}).to_parquet(os.path.join(O, "cache/images256_index.parquet"), index=False)
    test.to_parquet(os.path.join(O, "image_pairs_test.parquet"), index=False)
    for name in ("img_transformer_best.pt", "img_convlstm_k4_best.pt"):
        shutil.copy2(os.path.join(P, "checkpoints", name), os.path.join(O, "checkpoints", name))
    files = {}
    for dp, _, fn in os.walk(O):
        for f in fn:
            if f != "MANIFEST.json":
                p = os.path.join(dp, f)
                files[os.path.relpath(p, O)] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=P, capture_output=True, text=True).stdout.strip()
    json.dump({"test_pairs": len(test), "test_plants": int(test.plant_id.nunique()), "images": len(fps), "code_commit": commit, "files": files},
              open(os.path.join(O, "MANIFEST.json"), "w"), indent=1)
    tot = sum(v["bytes"] for v in files.values()) / 1e6
    print(f"Bundle written to {O}: {len(test)} test pairs, {test.plant_id.nunique()} plants, {len(fps)} images, {tot:.0f} MB total")
    for k, v in files.items():
        print(f"  {k}: {v['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
