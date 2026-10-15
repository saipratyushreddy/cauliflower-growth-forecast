#!/usr/bin/env bash
# Run ONCE on the Swan login node (which has internet) before submitting jobs that use the VGG perceptual loss / distance:
# downloads torchvision's ImageNet VGG16 weights (~528 MB) into the torch hub cache ($HOME/.cache/torch), so GPU jobs do not need internet.
set -euo pipefail
source "${PROJECT_DIR:-$WORK/cauliflower-growth-forecast}/.venv/bin/activate"
python - <<'PY'
import torchvision, torch
m = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
print("VGG16 weights cached at", torch.hub.get_dir())
PY
ls -la "$(python -c 'import torch; print(torch.hub.get_dir())')/checkpoints" | grep -i vgg
