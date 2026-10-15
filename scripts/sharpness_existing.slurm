#!/bin/bash
#SBATCH --job-name=img-sharp-existing
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

# Inference only, ORIGINAL project directory. Needs torchvision's VGG16 weights in the torch hub cache (see scripts/prefetch_vgg.sh, run once on the login node).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}"
cd "$PROJECT_DIR"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
python -u src/sharpness_existing.py --image-pairs data/image_pairs.parquet --cache-dir data --out-dir outputs \
    --sc-ckpt checkpoints/img_single_frame_best.pt --cl-ckpt checkpoints/img_convlstm_k4_best.pt --tf-ckpt checkpoints/img_transformer_best.pt --device cuda
