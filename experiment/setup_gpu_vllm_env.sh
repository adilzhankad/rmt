#!/bin/bash
# Env for the models run on VLMEvalKit's vLLM backend (config.yaml `env: vllm`): vLLM 0.10.2,
# which is built for torch 2.8 (the same torch as .venv-open), and transformers 4.x.
set -euo pipefail
cd ~/a3 && export PATH="$HOME/.local/bin:$PATH"
rm -rf .venv-vllm
uv venv -q --python 3.11 .venv-vllm
source .venv-vllm/bin/activate
uv pip install -q vllm==0.10.2 torch==2.8.0 torchvision==0.23.0
uv pip install -q -e third_party/VLMEvalKit -r experiment/requirements-extra.txt "vllm==0.10.2" "torch==2.8.0" "torchvision==0.23.0" "transformers>=4.55.2,<5" "datasets>=3" qwen-vl-utils
python -m playwright install chromium
python -c "import torch, transformers, vllm, vlmeval; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), 'transformers', transformers.__version__, 'vllm', vllm.__version__)"
echo VLLM_ENV_DONE
