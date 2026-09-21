#!/bin/bash
#SBATCH --job-name=growli-pe-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_DIR="$WORK/cauliflower-growth-forecast"
cd "$PROJECT_DIR"
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

# Narrow, hypothesis-driven check: does increasing d_model (64->128->256->512)
# resolve the sinusoidal positional encoding's aliasing behavior at small
# d_model and improve CNN-Transformer performance? Val-MAE-only model
# selection; test touched exactly once, for the winner of this small grid.
python src/check_pe_dmodel_sensitivity.py \
    --pairs-split data/pairs_split.parquet \
    --embeddings-dir data/embeddings \
    --norm-stats data/norm_stats.json \
    --baseline-results outputs/step5_baseline_results.json \
    --out-dir outputs \
    --checkpoint-dir checkpoints \
    --device cuda \
    --batch-size 64 \
    --max-epochs 200 \
    --patience 15 \
    --seed 42
