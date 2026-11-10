#!/bin/bash
# Preserve exact author pins without overriding the cluster's ABI policy.
# The old module-based environment and failed-job logs remain intact.
#SBATCH --job-name=dream-container
# Set your account using sbatch --account=YOUR_ACCOUNT
#SBATCH --partition=cpubase_6hrs
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dream-container-%j.out
#SBATCH --error=logs/dream-container-%j.err
set -euo pipefail
module load python/3.11 apptainer/1.4.5
cd "$HOME/latent_interfaces_v1"
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
mkdir -p containers results
test "$(git -C vendor/hidden_messages_f85b6d8 rev-parse HEAD)" = f85b6d8563a2236117407e0df9c6cd60ea406dac
test "$(git -C vendor/dllm_ca17675 rev-parse HEAD)" = ca176752fbceec49c6b4777a2c18ae88e4eb10ed
test "$(git -C vendor/lm_eval_dca091d rev-parse HEAD)" = dca091dbee3d6ee1dab268e9f649aa64e8db97d8
venv/bin/python -u dream_acquire_v1.py
image="$PWD/containers/python311_06bb5357.sif"
source_image='docker://python@sha256:06bb5357fbe87fed6fd7793c8c525ba09417bb23361b161511bb2993e8f0a7e3'
if [ ! -f "$image" ]; then
  apptainer pull "$image" "$source_image"
fi
printf '%s\n' "$source_image" > "results/dream_container_source_${SLURM_JOB_ID}.txt"
sha256sum "$image" > "results/dream_container_sha256_${SLURM_JOB_ID}.txt"
apptainer exec --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" "$image" \
  sh -c 'git --version && python --version'
if [ ! -d dream_container_env ]; then
  apptainer exec --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" "$image" \
    python -m venv "$PWD/dream_container_env"
fi
# Keep the actual package state even when an editable build/import fails.
trap 'bash dream_container_python_v1.sh -m pip freeze > "results/dream_environment_${SLURM_JOB_ID}.txt" || true' EXIT
# --no-build-isolation makes us responsible for Hatchling's dynamically
# declared editable-build dependency as well as its static build backend.
bash dream_container_python_v1.sh -m pip install --index-url https://pypi.org/simple \
  'setuptools>=61,<82' wheel hatchling==1.27.0 editables==0.5
bash dream_container_python_v1.sh -m pip install --index-url https://pypi.org/simple \
  --prefer-binary --no-build-isolation -r dream_release_requirements.txt \
  -e vendor/lm_eval_dca091d --report "results/dream_install_${SLURM_JOB_ID}.json"
bash dream_container_python_v1.sh -m pip install --no-index --no-build-isolation --no-deps \
  -e vendor/hidden_messages_f85b6d8/code -e vendor/dllm_ca17675
bash dream_container_python_v1.sh -u dream_import_preflight_v1.py
