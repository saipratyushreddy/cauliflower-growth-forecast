#!/bin/bash
#SBATCH --job-name=img-swap-hist
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Inference only, saved (already selected) checkpoints; no training, no model selection.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}"
cd "$PROJECT_DIR"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

python -u src/swapped_history_test.py --image-pairs data/image_pairs.parquet --metadata data/metadata.parquet --cache-dir data \
    --out-dir outputs --tf-ckpt checkpoints/img_transformer_best.pt --tf-k1-ckpt checkpoints/img_transformer_k1_best.pt \
    --tf-csv outputs/img_transformer_per_pair.csv --tf-k1-csv outputs/img_transformer_k1_per_pair.csv --device cuda --draws 5
