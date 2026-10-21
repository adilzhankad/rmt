#!/bin/bash
# Env for models whose code needs an older transformers: Ovis2-8B (4.46.2 per VLMEvalKit; newer
# versions register 'aimv2'), deepseek_vl2_small (the deepseek_vl2 package imports
# LlamaFlashAttention2, removed in 4.48) and the piloted Phi-3.5-Vision (DynamicCache.seen_tokens)
set -euo pipefail
cd ~/a3 && export PATH="$HOME/.local/bin:$PATH"
rm -rf .venv-legacy
uv venv -q --python 3.11 .venv-legacy
source .venv-legacy/bin/activate
uv pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q -e third_party/VLMEvalKit -r experiment/requirements-extra.txt "torch==2.8.0" "torchvision==0.23.0" "transformers==4.46.2" "datasets>=3" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
# deepseek_vl2: pinned commit, no deps (it pins torch 2.0.1); xformers for its vision tower,
# attrdict3 because attrdict does not import on Python >= 3.10
uv pip install -q "xformers==0.0.32.post2" --index-url https://download.pytorch.org/whl/cu128
uv pip install -q timm attrdict3 einops sentencepiece
uv pip install -q --no-deps "git+https://github.com/deepseek-ai/DeepSeek-VL2.git@ef9f91e2b6426536b83294c11742c27be66361b1"
python -c "import torch, transformers, flash_attn, xformers, deepseek_vl2.models, vlmeval; from vlmeval.config import supported_VLM; print('torch', torch.__version__, 'transformers', transformers.__version__, 'models ok', all(k in supported_VLM for k in ('Ovis2-8B', 'deepseek_vl2_small')))"
echo LEGACY_ENV_DONE
