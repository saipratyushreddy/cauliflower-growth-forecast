"""
Step 4: Cache frozen ResNet18 (ImageNet weights) embeddings for every image
in the reference metadata table.

Sharding scheme:
  - One shard file per plant_id: checkpoints-style tensor file at
    <out_dir>/shards/<plant_id>.pt containing a dict:
        {"filepaths": [str, ...], "embeddings": FloatTensor[N, 512]}
    ordered by day_after_planting (matches metadata.parquet's per-plant order).
  - A single Parquet index at <out_dir>/embedding_index.parquet mapping
    image_id (== filepath, which is already unique per image) -> (shard_file,
    offset, plant_id, embedding_dim). Downstream code loads a plant's whole
    shard once and indexes by offset, rather than opening one file per image.

Resumability:
  - Before computing anything, the index is loaded (if it exists) and any
    filepath already present there is skipped. A plant's shard is only
    rewritten if it has at least one missing image; otherwise it's left
    untouched. This means a killed/requeued SLURM job can be restarted with
    the exact same command and will only do the remaining work.

Model:
  - torchvision resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), frozen
    (eval mode, no_grad, no gradient updates -- never finetuned here), with
    its final fc layer replaced by Identity() so the model outputs the
    512-dim penultimate (avgpool) feature vector directly.
  - Standard ImageNet preprocessing (resize/crop + ImageNet mean/std) is a
    fixed constant from the pretraining source, not fit on GrowliFlower, so
    using it introduces no leakage (see src/fit_normalization.py docstring).

Usage:
    python src/cache_embeddings.py \
        --metadata data/metadata.parquet \
        --images-root <local dir OR HF cache dir containing the data/data_N/*.jpg tree> \
        --out-dir data/embeddings \
        --device cpu   # or cuda on Swan
"""
import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights


def build_model(device):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Identity()  # -> 512-dim penultimate features
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    preprocess = weights.transforms()
    return model, preprocess


def load_index(index_path):
    if os.path.exists(index_path):
        return pd.read_parquet(index_path)
    return pd.DataFrame(columns=["image_id", "shard_file", "offset", "plant_id", "embedding_dim"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--images-root", required=True,
                     help="Root directory containing the dataset's data/data_N/*.jpg tree "
                          "(filepath column in metadata.parquet is relative to this root)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shards_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    index_path = os.path.join(args.out_dir, "embedding_index.parquet")

    meta = pd.read_parquet(args.metadata)
    meta = meta.sort_values(["plant_id", "day_after_planting"]).reset_index(drop=True)

    index_df = load_index(index_path)
    already_done = set(index_df["image_id"])
    print(f"Existing index: {len(already_done)} images already cached")

    device = torch.device(args.device)
    model, preprocess = build_model(device)

    new_index_rows = []
    n_plants_total = meta["plant_id"].nunique()
    n_plants_processed = 0
    n_images_computed = 0
    n_images_skipped = 0
    n_images_failed = 0

    for plant_id, g in meta.groupby("plant_id", sort=False):
        g = g.sort_values("day_after_planting").reset_index(drop=True)
        filepaths = g["filepath"].tolist()

        missing = [fp for fp in filepaths if fp not in already_done]
        if not missing:
            n_images_skipped += len(filepaths)
            continue  # this plant's shard is already fully cached

        shard_path = os.path.join(shards_dir, f"{plant_id}.pt")

        # Load existing shard content if present, so we only add what's missing
        # rather than recompute embeddings this plant already has cached.
        if os.path.exists(shard_path):
            existing = torch.load(shard_path)
            existing_fp_to_emb = dict(zip(existing["filepaths"], existing["embeddings"]))
        else:
            existing_fp_to_emb = {}

        embeddings_ordered = []
        for start in range(0, len(missing), args.batch_size):
            batch_fps = missing[start:start + args.batch_size]
            imgs = []
            valid_fps = []
            for fp in batch_fps:
                full_path = os.path.join(args.images_root, fp)
                try:
                    img = Image.open(full_path).convert("RGB")
                    imgs.append(preprocess(img))
                    valid_fps.append(fp)
                except (FileNotFoundError, OSError) as e:
                    print(f"WARNING: failed to load {full_path}: {e}")
                    n_images_failed += 1
            if not imgs:
                continue
            batch_tensor = torch.stack(imgs).to(device)
            with torch.no_grad():
                feats = model(batch_tensor).cpu()
            for fp, feat in zip(valid_fps, feats):
                existing_fp_to_emb[fp] = feat
            n_images_computed += len(valid_fps)

        # Rebuild the shard in the metadata-defined per-plant order, keeping
        # only images that belong to this plant and that we could load.
        final_fps = [fp for fp in filepaths if fp in existing_fp_to_emb]
        final_embs = torch.stack([existing_fp_to_emb[fp] for fp in final_fps])
        torch.save({"filepaths": final_fps, "embeddings": final_embs}, shard_path)

        for offset, fp in enumerate(final_fps):
            new_index_rows.append({
                "image_id": fp,
                "shard_file": os.path.relpath(shard_path, args.out_dir),
                "offset": offset,
                "plant_id": plant_id,
                "embedding_dim": final_embs.shape[1],
            })

        n_plants_processed += 1
        if n_plants_processed % 50 == 0:
            print(f"  ...processed {n_plants_processed} plants so far")

    if new_index_rows:
        new_df = pd.DataFrame(new_index_rows)
        # Drop any stale rows for plants we just rewrote, then append fresh ones.
        touched_plants = set(new_df["plant_id"])
        index_df = index_df[~index_df["plant_id"].isin(touched_plants)]
        index_df = pd.concat([index_df, new_df], ignore_index=True)
        index_df.to_parquet(index_path, index=False)

    print(f"\nDone.")
    print(f"Plants with at least one new image computed: {n_plants_processed} / {n_plants_total}")
    print(f"Images newly computed: {n_images_computed}")
    print(f"Images already cached (skipped, whole-plant shortcut): {n_images_skipped}")
    print(f"Images failed to load: {n_images_failed}")
    print(f"Total images now in index: {len(index_df)}")
    print(f"Index written to {index_path}")


if __name__ == "__main__":
    main()
