"""
Compare the generic-date control with every other method on the early Field2 subset (target days 16, 22; n=146) and on the rest.

Table A: all methods on the subset, structure SSIM / RGB SSIM / PSNR, mean (median).
Table B: paired differences on the subset (mean (median) [95% plant-bootstrap CI], % first better): generic-date minus Transformer K=1,
         Transformer minus Transformer K=1 (the +0.016 gap), Transformer minus generic-date, plus the share of the Transformer-vs-K=1 gap
         reproduced by the generic-date control: (generic - K1) / (Transformer - K1), with a plant-bootstrap interval.
Table C: the same quantities on the other 1,087 test pairs and on all 1,233 (context).

Usage: python src/compare_generic_date.py --image-pairs data/image_pairs.parquet --generic outputs/img_ctrl_genericdate_per_pair.csv \
         --copy-forward ... --c1 ... --c2 ... --c3 ... --step-c ... --k1 ... --k4 ... --k4-seed2 ... --tf ... --tf-k1 ...
"""
import argparse

import numpy as np
import pandas as pd

from compare_temporal import boot_ci, load


def ratio_ci(a, b, plants, n_boot=2000, seed=0):
    """Plant-bootstrap interval of mean(a)/mean(b) (a, b per-pair differences)."""
    df = pd.DataFrame({"a": a, "b": b, "p": plants}).groupby("p").agg(sa=("a", "sum"), sb=("b", "sum"), n=("a", "size"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(df), size=(n_boot, len(df)))
    sa, sb = df.sa.values[idx].sum(1), df.sb.values[idx].sum(1)
    r = sa / sb
    return np.percentile(r, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-pairs", required=True)
    names = [("generic", "GEN"), ("copy-forward", "CF"), ("c1", "C1"), ("c2", "C2"), ("c3", "C3"), ("step-c", "SC"), ("k1", "K1"),
             ("k4", "K4"), ("k4-seed2", "K4b"), ("tf", "TF"), ("tf-k1", "TFK1")]
    for a, _ in names:
        ap.add_argument(f"--{a}", required=True)
    args = ap.parse_args()

    p = pd.read_parquet(args.image_pairs)
    te = p[p.split == "test"][["pair_id", "plant_id", "field", "target_day"]].sort_values("pair_id").reset_index(drop=True)
    ids = set(te.pair_id)
    M = {k: load(getattr(args, a.replace("-", "_")), ids) for a, k in names}
    lab = {"GEN": "Generic-date mean image", "CF": "Copy-forward", "C1": "C1 blur", "C2": "C2 flight-colour", "C3": "C3 colour+blur",
           "SC": "Step C", "K1": "ConvLSTM K=1", "K4": "ConvLSTM K=4 (s42)", "K4b": "ConvLSTM K=4 (s43)", "TFK1": "Transformer K=1",
           "TF": "Transformer (full)"}
    early = ((te.field == "Field2") & te.target_day.isin([16, 22])).values
    print(f"early Field2 subset: {early.sum()} of {len(te)} test pairs; training images behind each mean image: "
          f"{sorted(set(M['GEN']['n_train_images'].astype(int)))[:6]} (all groups)\n")
    fr = lambda k, m: M[k][m].values  # noqa: E731

    print("TABLE A -- early Field2 subset (targets days 16, 22; n=%d), test, mean (median)" % early.sum())
    print("| Method | Structure SSIM | RGB SSIM | RGB PSNR dB |\n|---|---|---|---|")
    for k in ["CF", "C1", "C2", "C3", "GEN", "SC", "K1", "K4", "K4b", "TFK1", "TF"]:
        cells = []
        for m, nd in [("structure_ssim", 4), ("ssim", 3), ("psnr", 2)]:
            v = fr(k, m)[early]
            cells.append(f"{v.mean():.{nd}f} ({np.median(v):.{nd}f})")
        print(f"| {lab[k]} | " + " | ".join(cells) + " |")

    def line(title, a, b, mask):
        d = fr(a, "structure_ssim") - fr(b, "structure_ssim")
        lo, hi = boot_ci(d[mask], te.plant_id.values[mask])
        return f"| {title} | {d[mask].mean():+.4f} ({np.median(d[mask]):+.4f}) [{lo:+.4f}, {hi:+.4f}], {(d[mask] > 0).mean():.0%} |"

    for name, mask in [("early Field2 subset (n=%d)" % early.sum(), early), ("all OTHER test pairs (n=%d)" % (~early).sum(), ~early),
                       ("all test pairs (n=%d)" % len(te), np.ones(len(te), bool))]:
        print(f"\nTABLE B/C -- structure SSIM differences on {name}")
        print("| Comparison | mean (median) [95% plant-bootstrap CI], % first better |\n|---|---|")
        for title, a, b in [("Generic-date minus Transformer K=1", "GEN", "TFK1"), ("Transformer minus Transformer K=1", "TF", "TFK1"),
                            ("Transformer minus Generic-date", "TF", "GEN"), ("Generic-date minus ConvLSTM K=4 (s42)", "GEN", "K4"),
                            ("Generic-date minus Step C", "GEN", "SC"), ("Generic-date minus C1 (blur)", "GEN", "C1"),
                            ("Generic-date minus C3 (colour+blur)", "GEN", "C3"), ("Generic-date minus copy-forward", "GEN", "CF")]:
            print(line(title, a, b, mask))
        gap = fr("TF", "structure_ssim") - fr("TFK1", "structure_ssim")
        gen = fr("GEN", "structure_ssim") - fr("TFK1", "structure_ssim")
        lo, hi = ratio_ci(gen[mask], gap[mask], te.plant_id.values[mask])
        print(f"| SHARE of the (Transformer - Transformer K=1) gap reproduced by the generic-date control | "
              f"{gen[mask].mean() / gap[mask].mean():.2f}  [{lo:.2f}, {hi:.2f}]  (generic-K1 = {gen[mask].mean():+.4f}; Transformer-K1 = {gap[mask].mean():+.4f}) |")


if __name__ == "__main__":
    main()
