#!/bin/bash
#SBATCH --job-name=img-single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s|gpu_a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_DIR="$WORK/cauliflower-growth-forecast"
cd "$PROJECT_DIR"
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"

# 1. Cache all cleaned-pair images at 256x256 as one uint8 array (idempotent; reused by later steps).
python -u src/cache_images_256.py --image-pairs data/image_pairs.parquet \
    --images-root "$PROJECT_DIR/data/images" --out-dir data

# 2. Smoke test on REAL images: 2 epochs on 48 train / 16 val pairs. Never reads the test split.
#    set -e aborts the job here if anything is broken, before the long run.
python -u src/train_img_single_frame.py --image-pairs data/image_pairs.parquet --cache-dir data \
    --out-dir outputs --checkpoint-dir checkpoints --device cuda --smoke

# 3. Full run. Best epoch chosen on VAL loss only; test evaluated exactly once at the end.
python -u src/train_img_single_frame.py --image-pairs data/image_pairs.parquet --cache-dir data \
    --out-dir outputs --checkpoint-dir checkpoints --device cuda

# 4. Final protocol table + paired comparison against the copy-forward baseline (same test pairs).
python -u src/summarize_baseline.py --per-pair-csv outputs/img_single_frame_per_pair.csv \
    --image-pairs data/image_pairs.parquet --no-oracle --label "single-frame FiLM-UNet" \
    --baseline-per-pair-csv outputs/img_stepB_copy_forward_per_pair.csv \
    --out outputs/img_single_frame_final_protocol.json
