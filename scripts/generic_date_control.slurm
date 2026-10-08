#!/bin/bash
#SBATCH --job-name=img-genericdate
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# CPU only: the generic-date structural control (needs data/images256.npy and the cleaned pair table, no checkpoints).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}"
cd "$PROJECT_DIR"
source .venv/bin/activate
python -u src/generic_date_control.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs
