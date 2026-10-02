#!/bin/bash
#SBATCH --job-name=img-blurscan
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

# CPU-only. Requires outputs/tiny_images_md5.txt (the known black-frame list).
python -u src/blur_scan.py --metadata data/metadata.parquet \
    --images-root "$PROJECT_DIR/data/images" \
    --exclude-list outputs/tiny_images_md5.txt --out-dir outputs
