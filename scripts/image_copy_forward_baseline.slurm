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

PROJECT_DIR="${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}"
cd "$PROJECT_DIR"
source .venv/bin/activate

# CPU-only: no model, just image loading + SSIM/PSNR.
# SKIP_PAIR_BUILD=1 (reproducibility pass): reuse the existing, verified data/image_pairs.parquet instead of rebuilding it.
if [ "${SKIP_PAIR_BUILD:-0}" != "1" ]; then
python -u src/build_image_pairs.py --metadata data/metadata.parquet \
    --phase0-split data/pairs_split.parquet --exclude-list outputs/tiny_images_md5.txt \
    --out data/image_pairs.parquet
fi
python -u src/image_copy_forward_baseline.py --image-pairs data/image_pairs.parquet \
    --images-root "$PROJECT_DIR/data/images" --out-dir outputs
python -u src/summarize_baseline.py --per-pair-csv outputs/img_stepB_copy_forward_per_pair.csv \
    --image-pairs data/image_pairs.parquet --out outputs/img_stepB_final_protocol.json
