#!/bin/bash
#SBATCH --job-name=pe-abl
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

# Positional-encoding ablation of the diameter CNN-Transformer; see src/pe_ablation.py.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
python -u src/pe_ablation.py --pairs-split data/pairs_split.parquet --embeddings-dir data/embeddings --norm-stats data/norm_stats.json \
    --out-dir outputs --checkpoint-dir checkpoints --device cuda
