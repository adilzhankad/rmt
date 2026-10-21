#!/bin/bash
# Env for the open-weight models: transformers 4.x (InternVL3's remote code breaks on 5.x) and
# torch 2.8 (the last torch with a prebuilt flash-attn; VLMEvalKit loads Qwen2.5-VL with flash_attention_2)
set -euo pipefail
cd ~/a3 && export PATH="$HOME/.local/bin:$PATH"
rm -rf .venv-open
uv venv -q --python 3.11 .venv-open
source .venv-open/bin/activate
uv pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q -e third_party/VLMEvalKit -r experiment/requirements-extra.txt "torch==2.8.0" "torchvision==0.23.0" "transformers>=4.49,<5" "datasets>=3" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
python -m playwright install chromium
python -c "import torch, transformers, flash_attn, vlmeval; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), 'transformers', transformers.__version__, 'flash_attn', flash_attn.__version__)"
echo OPEN_ENV_DONE
