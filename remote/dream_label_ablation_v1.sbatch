#!/bin/bash
#SBATCH --job-name=dream-label-ablation-v1
# Set your allocation with sbatch --account=YOUR_ACCOUNT when submitting.
#SBATCH --partition=gpu-L4
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dream-label-ablation-v1-%j.out
#SBATCH --error=logs/dream-label-ablation-v1-%j.err
set -euo pipefail
module load apptainer/1.4.5
cd "$HOME/latent_interfaces_v1"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
srun --ntasks=1 bash dream_container_python_v1.sh -u dream_label_ablation_v1.py --out "results/dream_label_ablation_v1_$SLURM_JOB_ID"
