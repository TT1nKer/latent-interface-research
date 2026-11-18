#!/bin/bash
# CPU-only regression for the observed ProcessorMixin dispatch failure.
#SBATCH --job-name=vla-processor-check-v1
# Set your CPU account using sbatch --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/vla-processor-check-v1-%j.out
#SBATCH --error=logs/vla-processor-check-v1-%j.err
set -euo pipefail
module load apptainer/1.4.5
cd "$HOME/latent_interfaces_v1"
test -n "${VLA_PREPARE_RECEIPT:-}"
export HF_HUB_OFFLINE=1
out="results/vla_processor_check_v1_${SLURM_JOB_ID}"
mkdir "$out"
export VLA_SOURCE_ROOT="$PWD/vendor/openvla_oft_e4287e9"
bash vla_container_python_v1.sh vla_latency_worker_v1.py --processor-only \
  --source native --prepare "$VLA_PREPARE_RECEIPT" --out "$out"
export VLA_SOURCE_ROOT="$PWD/vendor/fastvla_31644f9"
bash vla_container_python_v1.sh vla_latency_worker_v1.py --processor-only \
  --source fastvla --prepare "$VLA_PREPARE_RECEIPT" --out "$out"
bash vla_container_python_v1.sh - "$out" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1])
checks={k:json.loads((p/(k+'_processor_check.json')).read_text()) for k in ['native','fastvla']}
assert all(c['passed'] is True and c['model_loaded'] is False for c in checks.values())
for key in ['worker_sha256','prepare_sha256','processor_source_sha256','config_sha256','checks']:
    assert checks['native'][key]==checks['fastvla'][key], key
(p/'summary.json').write_text(json.dumps({'passed':True,'model_loaded':False,'checks':checks},indent=2)+'\n')
PY
