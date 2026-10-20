#!/bin/bash
# Env for models whose HF remote code needs an older transformers: Phi-3.5-Vision (< 4.47,
# DynamicCache.seen_tokens) and Ovis2-8B (4.46.2 per VLMEvalKit; newer versions register 'aimv2')
set -euo pipefail
cd ~/a3 && export PATH="$HOME/.local/bin:$PATH"
rm -rf .venv-legacy
uv venv -q --python 3.11 .venv-legacy
source .venv-legacy/bin/activate
uv pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q -e third_party/VLMEvalKit -r experiment/requirements-extra.txt "torch==2.8.0" "torchvision==0.23.0" "transformers==4.46.2" "datasets>=3" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
python -c "import torch, transformers, flash_attn, vlmeval; from vlmeval.config import supported_VLM; print('torch', torch.__version__, 'transformers', transformers.__version__, 'models ok', all(k in supported_VLM for k in ('Phi-3.5-Vision', 'Ovis2-8B')))"
echo LEGACY_ENV_DONE
