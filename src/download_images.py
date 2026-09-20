"""
Pre-Step-4 prep: download the reference-task GrowliFlower images (the only
images Phase 1 needs) from Hugging Face Hub directly onto the target machine
(intended for Swan's $WORK filesystem, not $HOME).

Resumable: hf_hub_download skips re-downloading a file that's already present
and unchanged in the target local_dir, so this can simply be rerun if the
job is killed or times out partway through.

Usage:
    python src/download_images.py --metadata data/metadata.parquet --images-root $WORK/cauliflower-growth-forecast/data/images
"""
import argparse

import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--images-root", required=True)
    args = ap.parse_args()

    meta = pd.read_parquet(args.metadata)
    filepaths = sorted(meta["filepath"].unique())
    print(f"Downloading {len(filepaths)} unique reference-task images to {args.images_root}")

    n_ok = 0
    n_failed = 0
    for i, fp in enumerate(filepaths):
        try:
            hf_hub_download(
                repo_id="Voxel51/GrowliFlower",
                repo_type="dataset",
                filename=fp,
                local_dir=args.images_root,
            )
            n_ok += 1
        except Exception as e:
            print(f"FAILED: {fp}: {e}")
            n_failed += 1
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(filepaths)} processed ({n_ok} ok, {n_failed} failed)")

    print(f"\nDone. {n_ok} downloaded/verified, {n_failed} failed.")
    if n_failed > 0:
        print("Rerun this same command to retry the failed files.")


if __name__ == "__main__":
    main()
