#!/bin/bash
#SBATCH --job-name=c2c-acquire
# Set your CPU account using sbatch --account=YOUR_ACCOUNT
#SBATCH --partition=cpubase_6hrs
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=logs/c2c-acquire-%j.out
#SBATCH --error=logs/c2c-acquire-%j.err
set -euo pipefail
module load python/3.11
cd "$HOME/latent_interfaces_v1"
export HF_HOME="$HOME/latent_interfaces_v1/hf_cache"
export HF_HUB_DISABLE_XET=1
srun --ntasks=1 venv/bin/python c2c_acquire_v1.py
# Isolated compatibility packages; existing model environments are not modified.
bash c2c_runtime_install_v1.sh
