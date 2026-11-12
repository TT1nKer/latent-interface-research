# Reusable communication between frozen language models

Independent, noncommercial student research by Zhenghao Yang, Queen's University.
This public development snapshot contains the current experimental code, checkpoint
revisions, controls, and analysis. It is ongoing research, not a finished paper or
a validated new method. Further evaluation code, methodology, and positive or
negative findings will be released as the pilot progresses, subject to upstream
model, code, and data licences.

## Scientific status, 2026-11-10

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
equal incoming messages under different private contexts. The first 16-question
L4 inference check has now completed all four requested conditions:

| Condition | Correct / 16 | Labels equal to the author reference / 16 |
| --- | ---: | ---: |
| One cross exchange (`final_only`) | 15 | 16 |
| Four cross exchanges (`matched`) | 15 | 16 |
| Four self exchanges (`self`) | 15 | 15 |
| No incoming message (`no_message`) | 13 | 15 |

**Exact release validation failed on two labels.** Both self and no-message differ
on one example, in opposite correctness directions. Full-model/decoder parity
was exact in the author's implementation check. The run used PyTorch 2.12.0+cu130
on an L4, took 62.26 seconds including model loading and recording, and peaked at
14.70 GiB of PyTorch-allocated GPU memory (not total device usage). See
`results/dream_validity16_v1.summary.json` for provenance, mismatches and limits.
These selected 16 questions do not provide a scientific accuracy estimate.

A targeted diagnostic has also completed. Repeating the same 16 questions
preserved all self/no-message labels and all four self-message hashes. Restoring
the original complete 64-question blocks around the two discrepancies (128
questions total), including model and channel minibatch composition, preserved
both discrepant predictions. Self labels match 127/128 references and no-message
labels match 126/128. This does not support changed batch membership as the cause
of the original two discrepancies; hardware, numerical and other reference
runtime differences remain unresolved. This post-failure subset is diagnostic,
not an unbiased accuracy estimate. See `results/dream_batch_diagnostic_v1.summary.json`.

The unchanged 0–6-round trace validity stage has now completed on the fixed
16 questions. All six independent author-function prediction checks passed,
as did all four applicable message-hash checks in the same L4 environment.
The trace took 93.56 seconds including setup and those independent calls, with
14.70 GiB peak allocated memory. The same two public-reference discrepancies
remain, so the job correctly exited with a failed release-validation gate.
See `results/dream_trace_validity_v1.summary.json` for checks and provenance.

The L4 self discrepancy is not a final near-tie (B exceeds reference D by
3.890625 logits); the no-message discrepancy is close (H exceeds E by 0.078125).
Neither observation establishes the underlying cause. Further reference-side
runtime/output evidence is needed; unchanged jobs will not be resubmitted.
The 1,000-question mechanism experiment remains unstarted. The protocol and
primary analysis were fixed before the neural results; no gate was relaxed.

This work uses a full L4 on Frontenac;
the 8 GB MyResearchCloud vGPU is intended for bounded small-model profiling and
controls. Successful migration or a Dream-7B fit on 8 GB is not assumed.

## Explicit protocol amendment, 2026-11-10

After the documented small-cohort discrepancies, the researcher authorized a
separate within-environment mechanism experiment. The original v1 files and
failed reference-validation outcomes are preserved. This is an amendment after
observing validation data, recorded before any full1,000-question trajectory.
It does not retroactively make the release reproduction successful.

`remote/dream_trace_amendment_v2.json` specifies the change: exact reference-label
agreement is reported separately, while execution requires matching runtime and
source fingerprints, successful independent author-function comparisons, and
measured memory/time limits. V2 repeats all six implementation comparisons on
the full1,000 questions. The original primary statistic, all depth/shape checks,
common-input controls and inclusion of every question are unchanged.

A fixed192-question numerical probe uses the first64-question block plus the two
previously discrepant blocks, preserving original minibatches. It disables the
BF16 reduced-precision-reduction option described in the
[PyTorch2.12 numerical-accuracy documentation](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html).
This is one setting sensitivity check, not a full-FP32 or cross-hardware test;
identical outputs may mean the selected kernels did not use that option.
All-question analysis remains primary; mismatch-exclusion and worst-case-score
analyses are secondary diagnostics. `analyze_dream_recurrent_v2.py` imports the
unchanged v1 measurements and separately labels the amended interpretation.
The v2 validity job completed successfully: all six prediction checks and four
applicable message-hash checks passed. Its per-question results and token rows
are byte-identical to the earlier v1 trace. The two public-reference discrepancies
remain. The measured projection is 6,061.74 seconds (101.03 minutes, including the
registered margin), with 15.73 GiB peak reserved GPU memory; both resource gates
pass. See `results/dream_trace_validity_v2.summary.json`.

The full 1,000-question run saved every trajectory, then failed its implementation
gate after 67 minutes 51 seconds. All four applicable message-hash comparisons
passed, but five of six independent prediction comparisons failed (only self at
round 4 passed). The dependent numerical probe was cancelled without running.
The primary scientific analysis remains blocked. The small validation cohort did
not establish full-cohort implementation equivalence. Failure evidence and the
bounded diagnostic plan are in `results/dream_trace_full_v2.failure.json`.

Source inspection identifies a concrete difference to test: the trace takes the
designated agent's answer from a joint two-agent forward, while the author uses a
separate receiver-only terminal forward. Native output also has a full-model
versus decoder path difference; final-only projects receiver messages separately.
The diagnostic completed in 13 minutes 49 seconds. Joint replay exactly repeats
the saved hidden states, labels and logits; receiver-only matched1 readout matches
a fresh author evaluation on all 1,000 questions. Switching from joint to
receiver-only readout changes 2 matched1, 4 matched4, 1 self1, 0 self4, 2 final-only
and 12 native labels. Native receiver-only full-model and decoder logits are
identical, as are the two tested final-only projection paths. This supports a
terminal readout correction; it does not establish the cause of the H100 reference
differences. See `results/dream_readout_diagnostic_v1.summary.json`.

The v3 correction is registered before corrected scientific analysis in
`remote/dream_readout_amendment_v3.json`. It reuses every saved recurrence message,
separately measures both agents at every depth using the author terminal readout,
and repeats the four common-input receiver measurements. The original primary
weights, shape criterion, six depths, all-question inclusion and all six
independent author checks remain fixed. Old failed receipts remain failed.
The measured conservative full-run projection is 6,720 seconds within the
single-L4 two-hour cap; actual corrected full-cohort validity remains untested.

Only after full validity passes will `analyze_dream_recurrent_v3.py` apply the
original primary test. A negative result stops the parity branch. A positive
result remains provisional pending the fixed 192-question numerical probe.
That probe recomputes recurrence under the changed flag and uses the corrected
readouts; it cannot reuse baseline messages as the intervention. No corrected
mechanism result or numerical robustness result is claimed yet.

Each job uses one L4 and32GiB host RAM. The new validity and sensitivity jobs
are capped at20minutes each; the full experiment is capped at2hours and must
pass a conservative projection from the new validity timing before execution.

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
python3 -m unittest test_late_binding_analysis_v1 test_analyze_dream_recurrent_v1 test_dream_trace_v2 test_analyze_dream_recurrent_v2
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
`remote/dream_release_requirements.txt` records the exact Dream package pins.
Use `remote/dream_prepare_container_v1.sbatch` and
`remote/dream_container_python_v1.sh` for the pinned Python 3.11.14 Apptainer
runtime. The module-based `dream_prepare_v1.sbatch` is retained as historical
setup code: its wheel policy failed on this cluster. The container environment
passed CPU imports and ran the 16-question GPU check; the two prediction
mismatches above still block complete release validation. The editable-build
helper `editables==0.5` is installed explicitly because build isolation is disabled.
This is a staged cluster workflow: the pinned upstream checkouts, acquisition
environment and cached model/data must already be present at the paths used by
the preparation script. It is not a self-contained one-command installer.
The optional lexical-context analysis also expects the exact upstream development
data at `external/hidden_messages/data/development.jsonl`; that data is not bundled.

Exact C2C model revisions and file hashes are in
`remote/c2c_acquire_v1_manifest.json`; the upstream code is
https://github.com/thu-nics/C2C at `5eac72b90faf7b3dc2638e874fe7cba1e74c8bcf`.
The Dream implementation targets reproduction of the author release at
https://github.com/alanrbtx/hidden-messages-diffusion-agents/tree/f85b6d8563a2236117407e0df9c6cd60ea406dac.
The upstream repository provides the exact QASC development data and reference
predictions used by the driver; author-reported reproduction is separate from our
independent validation.

## Rights and acknowledgement

Original project code and documentation are available under the [MIT licence](LICENSE).
The ARC-derived development dataset and prediction records containing question text
are provided under **CC BY-SA 4.0**, as specified in their adjacent `.license` files.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution, modifications,
source revisions, and the distinction between bundled data and external dependencies.
The project licence does not relicense upstream code, model weights, or benchmark text.

MyResearchCloud / Computing for Humanity approved an allocation on 2026-11-07;
provisioning is pending. No existing experimental result used that allocation.
Future outputs using it will acknowledge MyResearchCloud / Computing for Humanity.
