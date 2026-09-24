#!/bin/bash
#SBATCH --job-name=growli-robust
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

# INFERENCE ONLY -- loads the two already-trained, frozen sweep-winner
# checkpoints exactly as saved. No training, no hyperparameter changes.
# CNN-LSTM sweep winner: h256_lr0.001_L2_d0.2 (hidden=256, 2 layers, dropout=0.2)
# CNN-Transformer sweep winner: d64_h4_lr0.0003_L2_do0.2 (d_model=64, 4 heads, 2 layers, dropout=0.2)
python src/robustness_missing_obs.py \
    --pairs-split data/pairs_split.parquet \
    --embeddings-dir data/embeddings \
    --norm-stats data/norm_stats.json \
    --lstm-checkpoint checkpoints/sweep/h256_lr0.001_L2_d0.2.pt \
    --lstm-hidden-dim 256 --lstm-num-layers 2 --lstm-dropout 0.2 \
    --transformer-checkpoint checkpoints/transformer_sweep/d64_h4_lr0.0003_L2_do0.2.pt \
    --transformer-d-model 64 --transformer-nhead 4 --transformer-num-layers 2 --transformer-dropout 0.2 \
    --lstm-test-predictions outputs/step6_cnn_lstm_sweep_winner_test_predictions.csv \
    --transformer-test-predictions outputs/step_p2_cnn_transformer_test_predictions.csv \
    --out-dir outputs \
    --device cuda
