#!/bin/bash
# Main env on a Linux GPU machine (used on NVIDIA Brev, L40S): API models, evaluators, data.
# Run from the repository root copied to ~/a3:  ./experiment/setup_gpu_main_env.sh
set -euo pipefail
cd ~/a3
echo "HOME=$HOME"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone -q https://github.com/open-compass/VLMEvalKit.git third_party/VLMEvalKit
git -C third_party/VLMEvalKit checkout -q f57c3d90154010ba176fbafc39bb79ee521fbf70
git -C third_party/VLMEvalKit rev-parse HEAD
uv venv -q --python 3.11 .venv
source .venv/bin/activate
uv pip install -q -e third_party/VLMEvalKit -r experiment/requirements-extra.txt
python -m playwright install --with-deps chromium 2>&1 | tail -3
python -c "import torch, transformers, vlmeval; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), 'transformers', transformers.__version__)"
export LMUData=$HOME/a3/LMUData
python experiment/download_data.py
echo SETUP_DONE
