#!/bin/bash
# Install prebuilt cluster wheels only. No model imports or source compilation.
set -euo pipefail
# Resolve the full lightweight dependency closure. --no-deps previously hid a
# missing requests package until GPU startup. These packages do not require torch.
venv/bin/python -m pip install --no-index --only-binary=:all: --upgrade \
  --target c2c_runtime transformers==4.52.4+computecanada \
  tokenizers==0.21.1+computecanada huggingface-hub==0.36.0+computecanada
PYTHONPATH="$PWD/c2c_runtime" TOKENIZERS_PARALLELISM=false \
  venv/bin/python c2c_tokenizer_preflight_v1.py
