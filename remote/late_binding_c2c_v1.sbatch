#!/bin/bash
#SBATCH --job-name=c2c-late-dev
# Set your GPU account using sbatch --account=YOUR_ACCOUNT
#SBATCH --partition=gpu-L4
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/c2c-late-dev-%j.out
#SBATCH --error=logs/c2c-late-dev-%j.err
set -euo pipefail
module load python/3.11
cd "$HOME/latent_interfaces_v1"
export HF_HOME="$HOME/latent_interfaces_v1/hf_cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/c2c_runtime:$PWD/vendor/c2c_official"
srun --ntasks=1 venv/bin/python -u late_binding_c2c_v1.py
