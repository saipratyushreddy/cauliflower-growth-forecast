#!/bin/bash
#SBATCH --job-name=growli-lstm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_DIR="$WORK/cauliflower-growth-forecast"
cd "$PROJECT_DIR"
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

python src/train_cnn_lstm.py \
    --pairs-split data/pairs_split.parquet \
    --embeddings-dir data/embeddings \
    --norm-stats data/norm_stats.json \
    --baseline-results outputs/step5_baseline_results.json \
    --out-dir outputs \
    --checkpoint-dir checkpoints \
    --device cuda \
    --hidden-dim 128 \
    --num-layers 1 \
    --batch-size 64 \
    --lr 1e-3 \
    --max-epochs 200 \
    --patience 15 \
    --seed 42 \
    --n-plot-plants 6
