#!/bin/bash
#SBATCH --job-name=rp-nd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Nondeterminism test: the Step C training run (single-frame model, seed 42, default hyperparameters) repeated with IDENTICAL settings in the SAME
# environment, writing to separate output folders so the earlier fresh run is not overwritten. Pinned to L40S (the GPU model of the original and the
# fresh Step C run), so all four runs of this configuration share the GPU model.
#   sbatch --export=ALL,RUN=1 --job-name=rp-nd1 scripts/repro/nondeterminism_stepc.slurm
#   sbatch --export=ALL,RUN=2 --job-name=rp-nd2 scripts/repro/nondeterminism_stepc.slurm
set -euo pipefail
: "${RUN:?set RUN=1 or RUN=2}"
PROJECT_DIR="${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}"
cd "$PROJECT_DIR"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('GPU:', torch.cuda.get_device_name(0))"
echo "node: $(hostname)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| cudnn', torch.backends.cudnn.version())"
python -u src/train_img_single_frame.py --image-pairs data/image_pairs.parquet --cache-dir data \
    --out-dir "outputs_nd${RUN}" --checkpoint-dir "checkpoints_nd${RUN}" --device cuda
