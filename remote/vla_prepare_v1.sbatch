#!/bin/bash
# Preparation only; no model loading, no local execution, no automatic retries.
#SBATCH --job-name=vla-prepare-v1
# Supply your authorized Slurm account at submission with --account.
#SBATCH --partition=cpubase_6hrs
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/vla-prepare-v1-%j.out
#SBATCH --error=logs/vla-prepare-v1-%j.err
set -euo pipefail
module load apptainer/1.4.5
cd "$HOME/latent_interfaces_v1"
export CUDA_VISIBLE_DEVICES=""
mkdir -p containers vla_assets_v1 results vendor
out="results/vla_prepare_v1_${SLURM_JOB_ID}"
mkdir "$out"
image="$PWD/containers/python310_3a93854a.sif"
source_image='docker://python@sha256:3a93854ab059302d9d88aef36cf11a8d8e81de310695698adadc18137d222216'
printf '%s\n' "$source_image" > "$out/container_source.txt"
if [ ! -f "$image" ]; then apptainer pull "$image" "$source_image"; fi
sha256sum "$image" > "$out/container_sha256.txt"
if [ ! -d vla_env_v1 ]; then
  apptainer exec --cleanenv --bind "$PWD:$PWD" --pwd "$PWD" "$image" python -m venv "$PWD/vla_env_v1"
fi
trap 'bash vla_container_python_v1.sh -m pip freeze > "$out/freeze.txt" || true' EXIT
bash vla_container_python_v1.sh -m pip install 'pip==24.3.1' 'setuptools==75.3.0' wheel
bash vla_container_python_v1.sh -m pip install --prefer-binary \
  -r vla_latency_requirements_v1.txt --report "$out/install_report.json"
bash vla_container_python_v1.sh -u vla_acquire_v1.py --out "$out"
