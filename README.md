# Reusable communication between frozen language models

Independent, noncommercial student research by Zhenghao Yang, Queen's University.
This private development snapshot contains the current experimental code, checkpoint
revisions, controls, and analysis. Evaluation code, methodology, and positive or
negative findings will be released publicly when the pilot is completed, subject to
upstream model, code, and data licences.

## Scientific status, 2026-11-07

**No validated paper-level contribution or scaling result is claimed.**

The initial C2C experiment asks whether an earlier answer encoding continues to
influence a frozen receiver after its instruction changes. It compares the native
receiver, fusion before the new instruction, and fusion including the new instruction.
The 32-question development run completed on Frontenac. Across 96 correlated
nonidentity answer rotations, native accuracy was 28/96, fusion before the update
31/96, and fusion including the update 30/96. The native competence gate of 60%
failed, so this configuration is not being expanded. A scoring-format issue remains
a possible explanation; the result does not establish an interface-specific failure.
See `results/late_binding_c2c_dev32_v1.decision.json` and the raw predictions.

A separate Dream/QASC branch studies repeated latent-message exchange. The
implementation records cross/self trajectories at depths 0 through 6 and compares
equal incoming messages under different private contexts. This branch has no new
neural results in this snapshot. The protocol and analysis were fixed before those
results. Initial 16-question implementation/resource validation must precede the
1,000-question development experiment. This work uses a full L4 on Frontenac;
the 8 GB MyResearchCloud vGPU is intended for bounded small-model profiling and
controls. Successful migration or a Dream-7B fit on 8 GB is not assumed.

## Contents and provenance

- `remote/`: C2C acquisition/evaluation, pinned model manifests, Dream preparation,
  trace driver, frozen protocol, and Slurm templates.
- `analyze_*.py` and `test_*.py`: standard-library analysis and measurement tests.
- `results/`: observed C2C predictions, resource measurements, analysis, and decision.
- `vendor/c2c_official/PROVENANCE.json`: upstream code URL and exact revision;
  upstream implementation and model weights are not bundled.
- `SNAPSHOT_MANIFEST.json`: local source commit and SHA-256 of source/exported files.
  Cluster home paths in metadata and personal Slurm accounts were generalized;
  experimental Python code and raw predictions are unchanged.

This is a scoped snapshot, not a copy of private conversations, correspondence,
credentials, or the complete historical research archive. Old result file hashes
refer to the original run; the snapshot manifest identifies any metadata edits.

## Recheck existing analysis without a GPU

Run from this repository root with Python 3.11 or later:

```bash
python3 -m unittest test_late_binding_analysis_v1 test_analyze_dream_recurrent_v1
python3 analyze_late_binding_c2c_v1.py results/late_binding_c2c_dev32_v1.jsonl \
  --metadata results/late_binding_c2c_dev32_v1.json --out /tmp/c2c-reanalysis.json
```

The output path must not already exist. These checks do not load models or constitute
a new neural experiment. The four answer rotations of one question are correlated;
analysis resamples questions, not individual rotations.

## Neural execution and dependencies

Scripts under `remote/` are staged flat into a cluster project work directory with
the root analysis scripts. They retain Slurm/CUDA guards to prevent accidental local
model execution. Set a valid Slurm account and adjust module/wheelhouse settings for
your site; do not run the supplied jobs blindly on a workstation or login node.
The C2C runtime uses a site-specific Transformers 4.52.4 override. Its actual
software versions and measured allocation peak are recorded in the run metadata.
`remote/dream_release_requirements.txt` and `remote/dream_prepare_v1.sbatch`
record the Dream package and upstream Git pins; that environment is not yet validated.

Exact C2C model revisions and file hashes are in
`remote/c2c_acquire_v1_manifest.json`; the upstream code is
https://github.com/thu-nics/C2C at `5eac72b90faf7b3dc2638e874fe7cba1e74c8bcf`.
Dream reproduces the author release at
https://github.com/alanrbtx/hidden-messages-diffusion-agents/tree/f85b6d8563a2236117407e0df9c6cd60ea406dac.
The upstream repository provides the exact QASC development data and reference
predictions used by the driver; author-reported reproduction is separate from our
independent validation.

## Rights and acknowledgement

Upstream code, model weights, and benchmark text remain subject to their respective
licences. ARC question text is included in the small development data/predictions
with its source URL and dataset revision recorded in
`remote/late_binding_arc_dev32_v1.json`. This snapshot does not relicense third-party
materials. A licence and full third-party notices will accompany the public release.

MyResearchCloud / Computing for Humanity approved an allocation on 2026-11-07;
provisioning is pending. No existing experimental result used that allocation.
Future outputs using it will acknowledge MyResearchCloud / Computing for Humanity.
