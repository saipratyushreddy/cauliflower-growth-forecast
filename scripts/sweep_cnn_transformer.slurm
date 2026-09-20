#!/bin/bash
#SBATCH --job-name=growli-tfmr-sweep
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

# 24-config grid (d_model x nhead x num_layers x lr, dropout for 2-layer
# only). Model selection is by VALIDATION MAE only; test is evaluated
# exactly once, for the single winning config -- see
# src/sweep_cnn_transformer.py docstring. Also produces a 4-way
# shared-eval-set comparison against Phase 1's persistence/single-frame/
# CNN-LSTM results (requires outputs/step5_baseline_results.json and
# outputs/step6_cnn_lstm_sweep_winner_test_predictions.csv to already
# exist from Phase 1 -- both should already be on Swan from the earlier
# sweep_cnn_lstm.slurm run).
python src/sweep_cnn_transformer.py \
    --pairs-split data/pairs_split.parquet \
    --embeddings-dir data/embeddings \
    --norm-stats data/norm_stats.json \
    --baseline-results outputs/step5_baseline_results.json \
    --lstm-results outputs/step6_sweep_winner_results.json \
    --out-dir outputs \
    --checkpoint-dir checkpoints \
    --device cuda \
    --batch-size 64 \
    --max-epochs 200 \
    --patience 15 \
    --seed 42 \
    --n-plot-plants 6

# Not resumable across configs (same as sweep_cnn_lstm.slurm) -- rerun
# from scratch if this times out. 24 configs at a similar per-config cost
# to the CNN-LSTM sweep (18 configs finished well within 4 hours) should
# also comfortably fit; increase --time if the cluster is busy or configs
# run slower than expected (Transformer attention cost scales with
# sequence length^2, but sequences here are short -- max 14 frames).
