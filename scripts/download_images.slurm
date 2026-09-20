#!/bin/bash
#SBATCH --job-name=growli-download
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ---- EDIT these two paths if your layout differs ----
PROJECT_DIR="$WORK/cauliflower-growth-forecast"
IMAGES_ROOT="$PROJECT_DIR/data/images"
# -------------------------------------------------------

module purge
module load anaconda   # adjust to whatever Swan's actual module name is; confirm with `module spider anaconda` or `module avail python`

cd "$PROJECT_DIR"
source .venv/bin/activate   # created via: python -m venv .venv && pip install -r requirements.txt

mkdir -p "$IMAGES_ROOT"

python src/download_images.py \
    --metadata data/metadata.parquet \
    --images-root "$IMAGES_ROOT"
