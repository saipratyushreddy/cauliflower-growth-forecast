"""
Reproducibility pass, STEP A: verify the cached / underlying data BEFORE trusting it. Recomputes, from the raw images and the code in
--code-dir, and compares with (i) the numbers stated in the README, (ii) the actual file lists / tables on disk (not just counts).
Exit code 0 only if every check passes. Writes a JSON report incl. fingerprints of the key data files and the software environment.

Checks
  1  raw images: count (9,377) and exact set equality with metadata.parquet filepaths; no extra jpgs
  2  black / placeholder frames: recomputed from file size (<= 9,216 bytes, = `find -size -10k`) and md5, compared FILE BY FILE with the
     saved list outputs/tiny_images_md5.txt (234 files; 232 share one md5, 2 near-black: Plot2_A9-8, Plot2_C9-8)
  3  blurry exclusions: code constant vs an independent copy of the README list (9 images); each file exists; for the 9 files the variance of
     Laplacian is recomputed and compared with the stored blur scan (same file CONTENT, not just same name)
  4  pair table: rebuilt into a scratch file with the code under test and compared EXACTLY (all columns) with data/image_pairs.parquet; counts
     per split and per field, black-frame removals (178/16/40), blurry removals (18 = 14/0/4), scrubbed pairs (25), unfiltered 8,638
  5  split: the 738 Phase 0 plants keep their assignment; the 739th plant is in train; Phase 0 counts 517/111/110; image track 518/111/110
  6  image cache: images256.npy rebuilt from the raw images and compared byte-for-byte; index parquet identical
  7  fingerprints (sha256) of the data files + environment (python / library versions, pip freeze)

Usage:
    python src/repro_verify_data.py --project-dir $WORK/cauliflower-growth-forecast --code-dir . --scratch-dir <dir> --report <json>
    (--skip-images for a machine without the raw images: checks 1, 2, 3 (VoL part) and 6 are then skipped)
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# README-stated reference values (independent of the code under test)
EXPECT = {
    "n_raw_images": 9377, "n_black": 234, "n_black_same_md5": 232, "black_md5": "1fe4b7f0a4c8f7f36474bd46c2c12408",
    "black_odd": {"data/data_5/2020_Ref_Plot2_A9-8.jpg", "data/data_5/2020_Ref_Plot2_C9-8.jpg"},
    "unfiltered_pairs": 8638, "black_pairs_removed": {"train": 178, "val": 16, "test": 40},
    "blurry_target_removed": {"train": 7, "val": 0, "test": 2}, "blurry_total_removed": {"train": 14, "val": 0, "test": 4},
    "blurry_total_pairs": 18, "scrubbed_pairs": 25,
    "pairs": {"train": 5823, "val": 1330, "test": 1233},
    "plants": {"train": 518, "val": 111, "test": 110},
    "field_pairs": {("Field1", "train"): 1021, ("Field1", "val"): 168, ("Field1", "test"): 215,
                    ("Field2", "train"): 4802, ("Field2", "val"): 1162, ("Field2", "test"): 1018},
    "phase0_plants": {"train": 517, "val": 111, "test": 110},
    "blurry": [(f"2021_Ref_Plot5_{p}", "2021-08-11") for p in ("A16", "A17", "A18", "B17")]
              + [(f"2021_Ref_Plot1_{p}", "2021-08-30") for p in ("A4", "A5", "A6", "B6", "B7")],
}
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def norm_pairs(df):
    d = df.copy()
    for c in d.columns:
        if d[c].map(lambda v: isinstance(v, (list, np.ndarray))).any():
            d[c] = d[c].map(lambda v: tuple(np.asarray(v).tolist()))
    return d.sort_values("pair_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True, help="directory holding the data/ and outputs/ being verified")
    ap.add_argument("--code-dir", required=True, help="fresh clone whose code is tested")
    ap.add_argument("--scratch-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    P, C, S = args.project_dir, args.code_dir, args.scratch_dir
    os.makedirs(S, exist_ok=True)
    sys.path.insert(0, os.path.join(C, "src"))
    meta = pd.read_parquet(os.path.join(P, "data/metadata.parquet"))
    images_root = os.path.join(P, "data/images")
    fps = sorted(meta["filepath"].unique())
    check("metadata: 9,377 rows / 739 plants / 9,377 unique files",
          len(meta) == 9377 and meta.plant_id.nunique() == 739 and len(fps) == 9377, f"{len(meta)} rows, {meta.plant_id.nunique()} plants, {len(fps)} files")

    # 1-2: raw images and black frames
    ref_black = pd.read_csv(os.path.join(P, "outputs/tiny_images_md5.txt"), sep=r"\s+", header=None, names=["md5", "fp"])
    if not args.skip_images:
        found = set()
        for dp, _, fn in os.walk(images_root):
            for f in fn:
                if f.lower().endswith(".jpg"):
                    found.add(os.path.relpath(os.path.join(dp, f), images_root))
        check("1 raw images: count 9,377 and set == metadata filepaths", len(found) == EXPECT["n_raw_images"] and found == set(fps),
              f"{len(found)} jpgs; missing {len(set(fps) - found)}, extra {len(found - set(fps))}")
        sizes = {fp: os.path.getsize(os.path.join(images_root, fp)) for fp in fps if os.path.exists(os.path.join(images_root, fp))}
        small = sorted(fp for fp, s in sizes.items() if s <= 9216)
        rec = pd.DataFrame({"fp": small, "md5": [md5(os.path.join(images_root, fp)) for fp in small]})
        same = set(rec.fp) == set(ref_black.fp) and len(rec) == EXPECT["n_black"]
        check("2a black frames: recomputed set (size <= 9,216 B) == saved list file-by-file; 234 files", same,
              f"recomputed {len(rec)}, saved {len(ref_black)}, only-recomputed {sorted(set(rec.fp) - set(ref_black.fp))[:3]}, only-saved {sorted(set(ref_black.fp) - set(rec.fp))[:3]}")
        m = rec.merge(ref_black, on="fp", suffixes=("", "_saved"))
        check("2b black frames: md5 of every file equals the saved md5", len(m) == len(rec) and (m.md5 == m.md5_saved).all(),
              f"{int((m.md5 != m.md5_saved).sum())} md5 mismatches")
        n_same = int((rec.md5 == EXPECT["black_md5"]).sum())
        odd = set(rec[rec.md5 != EXPECT["black_md5"]].fp)
        check("2c black frames: 232 share md5 1fe4b7f0..., the other 2 are Plot2_A9-8 and Plot2_C9-8", n_same == 232 and odd == EXPECT["black_odd"],
              f"{n_same} with the shared md5; others {sorted(odd)}")
    # 3: blurry exclusions
    from build_image_pairs import BLURRY_EXCLUDE
    check("3a blurry list: code constant == independent README list (9 images)", sorted(BLURRY_EXCLUDE) == sorted(EXPECT["blurry"]),
          f"{len(BLURRY_EXCLUDE)} entries")
    bl = meta.merge(pd.DataFrame(EXPECT["blurry"], columns=["plant_id", "acquisition_date"]), on=["plant_id", "acquisition_date"])
    check("3b blurry list: all 9 resolve to exactly one metadata file each", len(bl) == 9 and bl.filepath.nunique() == 9, f"{len(bl)} matched")
    scan_path = os.path.join(P, "outputs/img_blur_scan_all.csv")
    have9 = all(os.path.exists(os.path.join(images_root, fp)) for fp in bl.filepath)
    if not args.skip_images and os.path.exists(scan_path) and have9:
        from blur_scan import vol
        scan = pd.read_csv(scan_path).set_index("filepath")
        diffs = [abs(vol(images_root, fp) - scan.loc[fp, "vol"]) for fp in bl.filepath]
        check("3c blurry list: recomputed variance of Laplacian == stored blur scan (same file CONTENT)", max(diffs) < 1e-6, f"max |diff| {max(diffs):.2e}")
        rec_md5 = {fp: md5(os.path.join(images_root, fp)) for fp in bl.filepath}
    else:
        rec_md5 = {}
        print("[SKIP] 3c variance-of-Laplacian recompute (no images or no stored scan)")

    old = norm_pairs(pd.read_parquet(os.path.join(P, "data/image_pairs.parquet")))
    # 4: pair table rebuilt with the code under test and compared exactly
    try:
        out_pq = os.path.join(S, "image_pairs_rebuild.parquet")
        cmd = [sys.executable, os.path.join(C, "src/build_image_pairs.py"), "--metadata", os.path.join(P, "data/metadata.parquet"),
               "--phase0-split", os.path.join(P, "data/pairs_split.parquet"), "--exclude-list", os.path.join(P, "outputs/tiny_images_md5.txt"),
               "--out", out_pq]
        cp = subprocess.run(cmd, cwd=C, capture_output=True, text=True)
        open(os.path.join(S, "build_image_pairs.log"), "w").write(cp.stdout + cp.stderr)
        check("4a rebuild of the pair table ran", cp.returncode == 0, cp.stderr.strip().splitlines()[-1] if cp.returncode else "")
        log = cp.stdout

        def removed(label):
            mm = re.search(rf"{label}: \d+ pairs? removed -> (\{{[^}}]*\}})", log)
            return eval(mm.group(1)) if mm else None
        new = norm_pairs(pd.read_parquet(out_pq))
        same_tbl = old.shape == new.shape and list(old.columns) == list(new.columns) and old.equals(new)
        check("4b rebuilt pair table == data/image_pairs.parquet EXACTLY (every column, every row)", same_tbl, f"shapes {old.shape} vs {new.shape}")
        check("4c unfiltered pairs 8,638", f"Unfiltered: {EXPECT['unfiltered_pairs']} pairs" in log)
        check("4d black-frame pairs removed 178/16/40 (234)", removed("black frames") == EXPECT["black_pairs_removed"], str(removed("black frames")))
        check("4e blurry pairs removed: as target 7/0/2 (9), total 14/0/4 (18)",
              removed("blurry as target") == EXPECT["blurry_target_removed"] and removed("blurry total") == EXPECT["blurry_total_removed"],
              f"target {removed('blurry as target')}, total {removed('blurry total')}")
        mm = re.search(r"(\d+) retained pairs had a blurry frame scrubbed", log)
        check("4f scrubbed pairs == 25", mm is not None and int(mm.group(1)) == EXPECT["scrubbed_pairs"], mm.group(0) if mm else "line not found")
        cnt = old.split.value_counts().to_dict()
        check("4g cleaned pairs 5,823 / 1,330 / 1,233", cnt == EXPECT["pairs"], str(cnt))
        check("4h plants per split 518 / 111 / 110", old.groupby("split").plant_id.nunique().to_dict() == EXPECT["plants"],
              str(old.groupby("split").plant_id.nunique().to_dict()))
        fp_ = {k: int(v) for k, v in old.groupby(["field", "split"]).size().items()}
        check("4i Field1/Field2 pairs per split match the README table", fp_ == EXPECT["field_pairs"], str(fp_))
        # 5: split
        p0 = pd.read_parquet(os.path.join(P, "data/pairs_split.parquet")).drop_duplicates("plant_id").set_index("plant_id")["split"]
        cur = old.drop_duplicates("plant_id").set_index("plant_id")["split"]
        check("5a Phase 0 split 517 / 111 / 110 plants", p0.value_counts().to_dict() == EXPECT["phase0_plants"], str(p0.value_counts().to_dict()))
        check("5b all 738 Phase 0 plants keep their assignment; the 739th (2021_Ref_Plot2_B13) is in train",
              (cur.loc[p0.index] == p0).all() and len(cur) == 739 and cur.loc["2021_Ref_Plot2_B13"] == "train",
              f"{int((cur.loc[p0.index] != p0).sum())} changed")

    except Exception as e:  # a crash in this section must not hide the later checks
        check("4/5 pair-table and split checks completed without error", False, repr(e)[:300])
    # 6: image cache
    try:
        cache_npy, cache_idx = os.path.join(P, "data/images256.npy"), os.path.join(P, "data/images256_index.parquet")
        if not args.skip_images and not (os.path.exists(cache_npy) and os.path.exists(cache_idx)):
            check("6 image cache files exist (images256.npy + index)", False, f"missing under {os.path.join(P, 'data')}")
        elif not args.skip_images:
            from cache_images_256 import load as load256
            idx = pd.read_parquet(cache_idx)
            arr = np.load(cache_npy, mmap_mode="r")
            want = set(old.target_filepath) | set(old.input_last_filepath)
            for l in old.input_filepaths:
                want.update(l)
            check("6a cache index == sorted union of files referenced by the cleaned pairs", list(idx.filepath) == sorted(want) and arr.shape[0] == len(idx),
                  f"{len(idx)} indexed, {len(want)} referenced, array {arr.shape}")
            bad = [i for i, fp in enumerate(idx.filepath)
                   if not os.path.exists(os.path.join(images_root, fp)) or not np.array_equal(arr[i], load256(images_root, fp))]
            check("6b images256.npy == a fresh rebuild from the raw images, byte for byte", not bad, f"{len(bad)} of {len(idx)} rows differ")
    except Exception as e:
        check("6 image-cache checks completed without error", False, repr(e)[:300])
    # 7: fingerprints / environment
    fing = {}
    for rel in ["data/image_pairs.parquet", "data/metadata.parquet", "data/pairs_split.parquet", "data/norm_stats.json",
                "outputs/tiny_images_md5.txt", "data/images256_index.parquet"] + ([] if args.skip_images else ["data/images256.npy"]):
        if os.path.exists(os.path.join(P, rel)):
            fing[rel] = sha256(os.path.join(P, rel))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C, capture_output=True, text=True).stdout.strip()
    env = {"python": platform.python_version(), "platform": platform.platform(), "code_commit": git, "pip_freeze": freeze}
    for mod in ["torch", "numpy", "pandas", "scipy", "skimage", "PIL", "matplotlib", "pyarrow"]:
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    print("\nSoftware: " + ", ".join(f"{k} {env[k]}" for k in ["python", "torch", "numpy", "pandas", "scipy", "skimage", "PIL"] if k in env), flush=True)
    print(f"Code commit under test: {git}")
    n_fail = sum(not r["pass"] for r in RESULTS)
    rep = {"results": RESULTS, "fingerprints_sha256": fing, "blurry_file_md5": rec_md5, "environment": env,
           "elapsed_sec": time.time() - t0, "all_pass": n_fail == 0}
    json.dump(rep, open(args.report, "w"), indent=1)
    print(f"\n==== STEP A: {len(RESULTS) - n_fail} passed, {n_fail} FAILED ({time.time() - t0:.0f}s) ====")
    if n_fail:
        print("STOP: do not proceed to Step B; report these discrepancies:")
        for r in RESULTS:
            if not r["pass"]:
                print(f"  - {r['check']}: {r['detail']}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
