#!/bin/bash
#SBATCH --job-name=vla-latency-v1
# Supply your authorized Slurm account at submission with --account.
#SBATCH --partition=gpu-L4
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vla-latency-v1-%j.out
#SBATCH --error=logs/vla-latency-v1-%j.err
set -euo pipefail
module load apptainer/1.4.5
cd "$HOME/latent_interfaces_v1"
test -n "${VLA_PREPARE_RECEIPT:-}"
export HF_HUB_OFFLINE=1
out="results/vla_latency_probe_v1_${SLURM_JOB_ID}"
mkdir "$out"
sha256sum vla_latency_worker_v1.py vla_latency_assets_v1.json vla_latency_requirements_v1.txt \
  vla_container_python_v1.sh vla_latency_probe_v1.sbatch vla_latency_protocol_v1.json > "$out/code_sha256.txt"
nvidia-smi > "$out/nvidia_smi.txt"
export VLA_SOURCE_ROOT="$PWD/vendor/openvla_oft_e4287e9"
bash vla_container_python_v1.sh -u vla_latency_worker_v1.py --source native --prepare "$VLA_PREPARE_RECEIPT" --out "$out"
export VLA_SOURCE_ROOT="$PWD/vendor/fastvla_31644f9"
bash vla_container_python_v1.sh -u vla_latency_worker_v1.py --source fastvla --prepare "$VLA_PREPARE_RECEIPT" --out "$out"
