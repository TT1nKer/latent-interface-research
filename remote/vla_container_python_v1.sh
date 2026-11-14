#!/bin/bash
# Isolated inference-only environment; never execute models outside Slurm.
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
cd "$HOME/latent_interfaces_v1"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PIP_CONFIG_FILE=/dev/null
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_OMP_NUM_THREADS=2
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_TF_NUM_INTRAOP_THREADS=2
export APPTAINERENV_TF_NUM_INTEROP_THREADS=1
export APPTAINERENV_HF_HOME="$PWD/vla_assets_v1/hf_cache"
export APPTAINERENV_HF_HUB_DISABLE_XET=1
export APPTAINERENV_HF_HUB_DOWNLOAD_TIMEOUT=120
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export APPTAINERENV_TRANSFORMERS_OFFLINE="${HF_HUB_OFFLINE:-0}"
export APPTAINERENV_WANDB_MODE=disabled
export APPTAINERENV_PYTHONPATH="${VLA_SOURCE_ROOT:-$PWD/vendor/fastvla_31644f9}:$PWD"
gpu_args=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then gpu_args=(--nv); fi
exec apptainer exec --cleanenv "${gpu_args[@]}" --bind "$PWD:$PWD" --pwd "$PWD" \
  "$PWD/containers/python310_3a93854a.sif" "$PWD/vla_env_v1/bin/python" "$@"
