#!/bin/bash
#SBATCH --job-name=growli-embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ---- EDIT these two paths if your layout differs ----
PROJECT_DIR="$WORK/cauliflower-growth-forecast"
IMAGES_ROOT="$PROJECT_DIR/data/images"
# -------------------------------------------------------

cd "$PROJECT_DIR"
source .venv/bin/activate   # created via: python -m venv .venv && pip install -r requirements.txt

# Sanity check: confirm the requested GPU is actually visible before doing
# real work, so a silent CPU fallback doesn't burn the whole time allocation.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

python src/cache_embeddings.py \
    --metadata data/metadata.parquet \
    --images-root "$IMAGES_ROOT" \
    --out-dir data/embeddings \
    --device cuda \
    --batch-size 64

# This job is resumable: if it times out or is killed partway through, submit
# it again unchanged -- cache_embeddings.py checks the existing index and
# only computes images/plants that are still missing.
