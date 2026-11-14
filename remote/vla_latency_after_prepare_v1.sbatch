#!/bin/bash
# One finite successor job; no timer, watcher, retry loop, or chat callback.
# Submit with --dependency=afterok:<prepare_job> --kill-on-invalid-dep=yes.
#SBATCH --job-name=vla-latency-chain-v1
# Supply your authorized Slurm account at submission with --account.
#SBATCH --partition=gpu-L4
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vla-latency-chain-v1-%j.out
#SBATCH --error=logs/vla-latency-chain-v1-%j.err
set -euo pipefail
module load apptainer/1.4.5
cd "$HOME/latent_interfaces_v1"
test -n "${VLA_PREPARE_RECEIPT:-}"
test -n "${VLA_PREPARE_JOB:-}"
receipt_dir="results/vla_latency_chain_v1_${SLURM_JOB_ID}"
mkdir "$receipt_dir"
stage=prepare_gate
trap 'rc=$?; printf "{\"exit_code\":%d,\"stage\":\"%s\"}\n" "$rc" "$stage" > "$receipt_dir/exit.json"' EXIT
bash vla_container_python_v1.sh vla_latency_prepare_gate_v1.py \
  --prepare "$VLA_PREPARE_RECEIPT" --prepare-job "$VLA_PREPARE_JOB" \
  --bundle vla_latency_chain_bundle_v1.json --out "$receipt_dir/gate.json"
stage=gpu_probe
bash vla_latency_probe_v1.sbatch
stage=fixed_analysis
bash vla_container_python_v1.sh analyze_vla_latency_v1.py \
  "results/vla_latency_probe_v1_${SLURM_JOB_ID}" \
  --out "results/vla_latency_probe_v1_${SLURM_JOB_ID}/analysis.json"
stage=complete
