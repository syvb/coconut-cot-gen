#!/bin/bash
# Run on a worker: install deps, clone repo, download model.
# Idempotent — safe to re-run.
set -e

cd $HOME

if ! python -c "import torch_xla" 2>/dev/null; then
    echo "[setup] installing torch + torch_xla"
    pip install --quiet 'torch~=2.5.0' 'torch_xla[tpu]~=2.5.0' \
        -f https://storage.googleapis.com/libtpu-releases/index.html >/dev/null 2>&1
fi

if ! python -c "import transformers,datasets" 2>/dev/null; then
    echo "[setup] installing transformers, datasets, etc."
    pip install --quiet transformers datasets huggingface_hub accelerate sentencepiece >/dev/null 2>&1
fi

if [ ! -d "coconut-cot-gen/.git" ]; then
    echo "[setup] cloning repo"
    git clone --quiet https://github.com/syvb/coconut-cot-gen.git
else
    echo "[setup] updating repo"
    cd coconut-cot-gen && git pull --quiet && cd ..
fi

cd coconut-cot-gen
mkdir -p models data outputs

# Download model if needed
if [ ! -f "models/codi_llama1b/pytorch_model.bin" ]; then
    echo "[setup] downloading CODI model"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('bcywinski/codi_llama1b-answer_only', local_dir='models/codi_llama1b')
" >/dev/null 2>&1
fi

# Download config file (used by codi_loader)
if [ ! -f "models/codi_llama1b/config.json" ]; then
    curl -L -sSf -o models/codi_llama1b/config.json https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/config.json
fi

echo "[setup] OK: $(hostname)"
