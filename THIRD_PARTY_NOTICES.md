# Third-party materials and licence scope

The root MIT licence covers original project code and documentation. It does not
replace the licences of third-party data, code, models, or other assets. Merely
listing an upstream project or file in a provenance manifest does not mean its
implementation or weights are included in this repository.

## ARC question data included in this snapshot

Source: **ARC / AI2 Reasoning Challenge**, Allen Institute for Artificial Intelligence
(Ai2), described by Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish
Sabharwal, Carissa Schoenick, and Oyvind Tafjord, *Think you have Solved Question
Answering? Try ARC, the AI2 Reasoning Challenge* (2018),
https://arxiv.org/abs/1803.05457.

- Dataset: https://huggingface.co/datasets/allenai/ai2_arc
- Exact source revision: `210d026faf9955653af8916fad021475a3f00453`
- Licence declaration verified in the [pinned dataset card](https://huggingface.co/datasets/allenai/ai2_arc/blob/210d026faf9955653af8916fad021475a3f00453/README.md): **CC BY-SA 4.0**.
- Licence: https://creativecommons.org/licenses/by-sa/4.0/
- Legal terms: https://creativecommons.org/licenses/by-sa/4.0/legalcode

The following files contain ARC-derived material and are provided under CC BY-SA
4.0, including this project's contributions to those adapted data files to the
extent applicable:

- `remote/late_binding_arc_dev32_v1.json`
- `results/late_binding_c2c_dev32_v1.jsonl`

Changes: 32 eligible four-option questions were selected from the first 100
ARC-Challenge validation rows with seed 20261106. The selected rows were converted
to the project's schema. The prediction file repeats these questions in constructed
prompts, introduces four cyclic answer-label mappings, and adds model predictions,
probabilities, timing-related token counts, and checksums. Those prompts and
predictions are our experiment, not an official ARC release or benchmark result.
Question IDs and source information are retained. See the files' `.license`
sidecars. No endorsement by Ai2 or the original question authors is implied.

The aggregate numerical summaries and original analysis code are separate from
these adapted data files. The root MIT licence must not be interpreted as removing
the attribution or share-alike requirements from the included ARC material.

## External code and models, not redistributed here

**C2C.** The experiment imports the `rosetta` implementation from
https://github.com/thu-nics/C2C at
`5eac72b90faf7b3dc2638e874fe7cba1e74c8bcf`, whose root `LICENSE` is Apache-2.0.
Only our provenance metadata is included under `vendor/c2c_official/`.
The upstream file list in that metadata describes a separate local checkout,
not files included in this public repository. Preserve upstream notices when
obtaining or redistributing that implementation.

**Hidden Messages / Dream.** The trace and validity drivers call the external
author helpers from https://github.com/alanrbtx/hidden-messages-diffusion-agents
at `f85b6d8563a2236117407e0df9c6cd60ea406dac`. Author implementation, released
channel weights, reference predictions, and QASC question text are not bundled.
The root MIT licence applies to our wrappers, not that upstream release. A root
`LICENSE` was not found at the pinned upstream revision during this review;
no redistribution permission for its contents is inferred from its public visibility.
Review the upstream terms before redistributing any of those materials.

**Other runtime dependencies.** The preparation script fetches
https://github.com/ZHZisZZ/dllm at
`ca176752fbceec49c6b4777a2c18ae88e4eb10ed` and
https://github.com/ZHZisZZ/lm-evaluation-harness at
`dca091dbee3d6ee1dab268e9f649aa64e8db97d8`. Their source is not included;
use them under their respective upstream licences. Package versions recorded in
requirements files do not grant rights to redistribute those packages.

**Model weights.** No model weights are included. C2C model/checkpoint IDs,
revisions, and hashes are recorded in `remote/c2c_acquire_v1_manifest.json`;
Dream's model revision and weight-manifest checksum are recorded in
`remote/dream_acquire_v1.py`. Obtain weights from their original publishers under
their applicable terms. Download references and checksums are not relicensing.

Licence-source review date: 2026-11-07. Public release does not imply independent
validation of an upstream method or completion of this project's pending experiments.
