#!/bin/bash
#SBATCH --job-name=img-copyfwd
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_DIR="$WORK/cauliflower-growth-forecast"
cd "$PROJECT_DIR"
source .venv/bin/activate

# CPU-only: no model, just image loading + SSIM/PSNR.
python -u src/build_image_pairs.py --metadata data/metadata.parquet \
    --phase0-split data/pairs_split.parquet --exclude-list outputs/tiny_images_md5.txt \
    --out data/image_pairs.parquet
python -u src/image_copy_forward_baseline.py --image-pairs data/image_pairs.parquet \
    --images-root "$PROJECT_DIR/data/images" --out-dir outputs
