"""
Aliasing check for the continuous time encoding of (target_day - input_day), run BEFORE training the image
CNN-Transformer (same spirit as the Phase 0 check; the Phase 0 frequency base is NOT assumed to transfer).

Computes the encoding for the real range of (target_day - input_day) over every (pair, history frame) of the
cleaned pair set, and reports, for several encodings: how many anchor values have a cosine-similarity profile that is
non-monotone in distance, the largest rise, ordering violations among a representative spread of real values
(nearer value less similar than a farther one), and mean cosine at several distances.

Result used by the model (img_transformer.py): geometric frequency set, 128 dims (64 sin + 64 cos), wavelengths 16-1000
days. Worst non-monotone rise 0.0009 (Phase 0 setting: 0.078), 2 ordering violations of ~0.001-0.002 cosine.

Usage: python src/pe_aliasing_check.py --image-pairs data/image_pairs.parquet
"""
import argparse

import numpy as np
import pandas as pd


def pe_std(x, d, base):
    i = np.arange(d // 2)
    a = np.outer(x, 1.0 / (base ** (2 * i / d)))
    o = np.empty((len(x), d))
    o[:, 0::2], o[:, 1::2] = np.sin(a), np.cos(a)
    return o


def pe_geo(x, m=64, lmin=16.0, lmax=1000.0):
    lam = lmin * (lmax / lmin) ** (np.arange(m) / (m - 1))
    a = np.outer(x, 2 * np.pi / lam)
    return np.concatenate([np.sin(a), np.cos(a)], 1)


def cos(a):
    n = a / np.linalg.norm(a, axis=1, keepdims=True)
    return n @ n.T


def report(name, f, xs, spread):
    C = cos(f(xs))
    rises = [np.max(np.diff(C[a, a:])) if len(xs) - a > 1 else 0 for a in range(len(xs))]
    Cs = cos(f(spread))
    viol = sum(1 for i in range(len(spread)) for j in range(len(spread)) for k in range(len(spread))
               if abs(spread[i] - spread[j]) < abs(spread[i] - spread[k]) and Cs[i, j] < Cs[i, k] - 1e-9)
    prof = [np.mean(np.diag(C, k)) for k in (1, 3, 7, 20, 50, 80)]
    print(f"{name:36s} non-monotone anchors {sum(r > 1e-9 for r in rises):2d}/{len(xs)}  max rise {max(rises):.4f}  "
          f"spread violations {viol:3d} | mean cos at |dx|=1,3,7,20,50,80: " + " ".join(f"{x:.2f}" for x in prof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    args = ap.parse_args()
    p = pd.read_parquet(args.image_pairs)
    D = np.array([r.target_day - d for r in p.itertuples() for d in r.input_days])
    print(f"history frames per pair: min {p.n_input_frames.min()} median {int(p.n_input_frames.median())} max {p.n_input_frames.max()}")
    print(f"target_day - input_day over all (pair, frame): n={len(D)} min {D.min()} median {np.median(D):.0f} max {D.max()}, "
          f"{len(np.unique(D))} distinct values")
    xs = np.arange(D.min(), D.max() + 1).astype(float)
    spread = np.array([2, 5, 8, 15, 22, 36, 50, 64, 84], float)
    report("standard d=64 base=10000 (Phase 0)", lambda x: pe_std(x, 64, 10000), xs, spread)
    report("standard d=128 base=10000", lambda x: pe_std(x, 128, 10000), xs, spread)
    for lmin, lmax in [(10, 400), (16, 400), (24, 400), (16, 1000)]:
        report(f"geometric d=128 wavelengths {lmin}-{lmax}", lambda x, a=lmin, b=lmax: pe_geo(x, 64, a, b), xs, spread)


if __name__ == "__main__":
    main()
