#!/bin/bash
#SBATCH --job-name=growli-sweep
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

PROJECT_DIR="$WORK/cauliflower-growth-forecast"
cd "$PROJECT_DIR"
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

# 18-config grid (hidden_dim x lr x num_layers, dropout for 2-layer only).
# Model selection is by VALIDATION MAE only; test is evaluated exactly
# once, for the single winning config, at the end of the sweep -- see
# src/sweep_cnn_lstm.py docstring for the full rationale.
python src/sweep_cnn_lstm.py \
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
    --seed 42 \
    --n-plot-plants 6

# This job is resumable in spirit (each config's checkpoint is saved to
# checkpoints/sweep/<config_id>.pt as it finishes) but the sweep loop
# itself is not currently checkpointed/resumable across configs -- if
# this times out partway through, rerun from scratch. At ~29 epochs per
# config observed for a comparable single run, 18 configs should comfortably
# finish within the 4-hour limit; increase --time if the cluster is busy.
