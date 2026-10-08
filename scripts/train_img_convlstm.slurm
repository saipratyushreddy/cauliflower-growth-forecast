#!/bin/bash
#SBATCH --job-name=img-convlstm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Optional SEED (default 42; a non-default seed adds a _seed<N> suffix to every output file):
#   sbatch --export=ALL,K=4,SEED=43 --job-name=img-convlstm-k4-s43 scripts/train_img_convlstm.slurm
# Submit once per K (same architecture and hyperparameters, differing only in K):
#   sbatch --export=ALL,K=4 --job-name=img-convlstm-k4 scripts/train_img_convlstm.slurm
#   sbatch --export=ALL,K=1 --job-name=img-convlstm-k1 scripts/train_img_convlstm.slurm
set -euo pipefail
: "${K:?set K via --export=ALL,K=<n>}"

SEED="${SEED:-42}"
PROJECT_DIR="${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}"
cd "$PROJECT_DIR"
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

# 1. Smoke test on REAL images (2 epochs, 24 train / 8 val pairs; never reads the test split).
python -u src/train_img_convlstm.py --image-pairs data/image_pairs.parquet --cache-dir data \
    --out-dir outputs --checkpoint-dir checkpoints --device cuda --k "$K" --seed "$SEED" --smoke

# 2. Full run: best epoch by VAL loss only; test evaluated exactly once at the end.
python -u src/train_img_convlstm.py --image-pairs data/image_pairs.parquet --cache-dir data \
    --out-dir outputs --checkpoint-dir checkpoints --device cuda --k "$K" --seed "$SEED"
