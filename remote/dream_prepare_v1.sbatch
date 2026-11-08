#!/bin/bash
#SBATCH --job-name=dream-prepare
# Set your CPU account using sbatch --account=YOUR_ACCOUNT
#SBATCH --partition=cpubase_6hrs
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dream-prepare-%j.out
#SBATCH --error=logs/dream-prepare-%j.err
set -euo pipefail
module load python/3.11
cd "$HOME/latent_interfaces_v1"
export OMP_NUM_THREADS=2
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""
mkdir -p vendor results
pin_checkout() {
  local url="$1" destination="$2" revision="$3"
  if [ ! -d "$destination/.git" ]; then
    git clone --no-checkout "$url" "$destination"
    git -C "$destination" checkout --detach "$revision"
  fi
  test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
  test -z "$(git -C "$destination" status --porcelain)"
}
pin_checkout https://github.com/alanrbtx/hidden-messages-diffusion-agents.git vendor/hidden_messages_f85b6d8 f85b6d8563a2236117407e0df9c6cd60ea406dac
pin_checkout https://github.com/ZHZisZZ/dllm.git vendor/dllm_ca17675 ca176752fbceec49c6b4777a2c18ae88e4eb10ed
pin_checkout https://github.com/ZHZisZZ/lm-evaluation-harness.git vendor/lm_eval_dca091d dca091dbee3d6ee1dab268e9f649aa64e8db97d8
venv/bin/python -u dream_acquire_v1.py
if [ ! -d dream_env ]; then
  python -m venv dream_env
fi
dream_env/bin/python -m pip install --no-index --only-binary=:all: -r dream_release_requirements.txt
dream_env/bin/python -m pip install --no-index --no-build-isolation --no-deps \
  -e vendor/lm_eval_dca091d -e vendor/hidden_messages_f85b6d8/code -e vendor/dllm_ca17675
export PYTHONPATH="$PWD/vendor/hidden_messages_f85b6d8/code/src:$PWD/vendor/hidden_messages_f85b6d8/code/scripts:$PWD/vendor/dllm_ca17675"
dream_env/bin/python -u dream_import_preflight_v1.py
