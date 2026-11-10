#!/bin/bash
#SBATCH --job-name=dream-trace1000
# Set your account using sbatch --account=YOUR_ACCOUNT
#SBATCH --partition=gpu-L4
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dream-trace1000-%j.out
#SBATCH --error=logs/dream-trace1000-%j.err
set -euo pipefail
module load apptainer/1.4.5
cd "$HOME/latent_interfaces_v1"
test -n "$DREAM_VALIDITY_RECEIPT"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/vendor/hidden_messages_f85b6d8/code/src:$PWD/vendor/hidden_messages_f85b6d8/code/scripts:$PWD/vendor/dllm_ca17675"
srun --ntasks=1 bash dream_container_python_v1.sh -u dream_recurrent_trace_v1.py \
  --stage full --out "results/dream_trace_full_v1_$SLURM_JOB_ID" \
  --validity-receipt "$DREAM_VALIDITY_RECEIPT"
