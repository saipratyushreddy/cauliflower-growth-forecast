#!/bin/bash
#SBATCH --job-name=img-transformer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Optional: SEED (default 42; a non-default seed adds a _seed<N> suffix to every output file):
#   sbatch --export=ALL,SEED=43 --job-name=img-transformer-s43 scripts/train_img_transformer.slurm
# Optional: MAXHIST (default 0 = full history; 1 = single-frame control with the identical architecture; outputs get a _k1 tag):
#   sbatch --export=ALL,MAXHIST=1 --job-name=img-transformer-k1 --time=03:00:00 scripts/train_img_transformer.slurm
set -euo pipefail
SEED="${SEED:-42}"
MAXHIST="${MAXHIST:-0}"

PROJECT_DIR="$WORK/cauliflower-growth-forecast"
cd "$PROJECT_DIR"
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

# 1. Smoke test on REAL images: 2 epochs on a subset spanning 1-14 history frames; never reads the test split.
python -u src/train_img_transformer.py --image-pairs data/image_pairs.parquet --cache-dir data \
    --out-dir outputs --checkpoint-dir checkpoints --device cuda --seed "$SEED" --max-history "$MAXHIST" --smoke

# 2. Full run: best epoch by VAL loss only; test evaluated exactly once at the end.
python -u src/train_img_transformer.py --image-pairs data/image_pairs.parquet --cache-dir data \
    --out-dir outputs --checkpoint-dir checkpoints --device cuda --seed "$SEED" --max-history "$MAXHIST"
