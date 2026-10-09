"""
DEVELOPMENT ONLY. Fabricates a demo bundle with SYNTHETIC pixels and UNTRAINED (random-weight) checkpoints, using the real test-pair metadata, so the
app / backend code can be exercised on a machine that does not have the real images and checkpoints. Nothing produced from this bundle means anything.

Usage: python scripts/demo_dev_bundle.py --out <dir> [--plants-per-field 4]
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.ndimage as nd
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from img_convlstm import TemporalFiLMUNet                       # noqa: E402
from img_transformer import TemporalTransformerUNet             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="data/image_pairs.parquet")
    ap.add_argument("--plants-per-field", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "cache"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
    p = pd.read_parquet(a.pairs)
    te = p[p.split == "test"]
    keep_plants = []
    for f in ("Field1", "Field2"):
        keep_plants += sorted(te[te.field == f].plant_id.unique())[:a.plants_per_field]
    te = te[te.plant_id.isin(keep_plants)].reset_index(drop=True)
    fps = set(te.target_filepath) | set(te.input_last_filepath)
    for l in te.input_filepaths:
        fps.update(l)
    fps = sorted(fps)
    rng = np.random.RandomState(0)
    arr = np.zeros((len(fps), 256, 256, 3), np.uint8)
    for i in range(len(fps)):
        x = nd.gaussian_filter(rng.rand(256, 256, 3), (6, 6, 0))
        arr[i] = ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
    np.save(os.path.join(a.out, "cache/images256.npy"), arr)
    pd.DataFrame({"filepath": fps, "row": np.arange(len(fps))}).to_parquet(os.path.join(a.out, "cache/images256_index.parquet"), index=False)
    te.to_parquet(os.path.join(a.out, "image_pairs_test.parquet"), index=False)
    torch.manual_seed(0)
    tf = TemporalTransformerUNet(grad_checkpoint=False)
    torch.save({"model": tf.state_dict(), "epoch": 0, "args": {"d_model": 128, "nhead": 4, "num_layers": 2, "dim_ff": 256, "dropout": 0.1, "max_history": 0}},
               os.path.join(a.out, "checkpoints/img_transformer_best.pt"))
    cl = TemporalFiLMUNet()
    torch.save({"model": cl.state_dict(), "norm": {"mean": 6.8, "std": 4.4}, "epoch": 0, "args": {"k": 4}}, os.path.join(a.out, "checkpoints/img_convlstm_k4_best.pt"))
    print(f"SYNTHETIC dev bundle: {len(te)} pairs, {te.plant_id.nunique()} plants, {len(fps)} images -> {a.out}")


if __name__ == "__main__":
    main()
