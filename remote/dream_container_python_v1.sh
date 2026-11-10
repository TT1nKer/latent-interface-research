#!/bin/bash
# Same isolated interpreter for CPU import validation and Slurm GPU runs.
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
cd "$HOME/latent_interfaces_v1"
image="$PWD/containers/python311_06bb5357.sif"
test -f "$image"
test -f "$PWD/dream_container_env/pyvenv.cfg"
export APPTAINERENV_PIP_CONFIG_FILE=/dev/null
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export APPTAINERENV_OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_HF_HUB_DISABLE_XET=1
export APPTAINERENV_HF_HOME="$PWD/hf_cache"
export APPTAINERENV_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export APPTAINERENV_PYTHONPATH="$PWD/vendor/hidden_messages_f85b6d8/code/src:$PWD/vendor/hidden_messages_f85b6d8/code/scripts:$PWD/vendor/dllm_ca17675"
gpu_args=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  gpu_args=(--nv)
fi
exec apptainer exec --cleanenv "${gpu_args[@]}" --bind "$PWD:$PWD" \
  --pwd "$PWD" "$image" "$PWD/dream_container_env/bin/python" "$@"
