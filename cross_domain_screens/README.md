# Two tests of when switching between experts has value

## Bidirectional image and text retrieval

A specialist can rank texts for an image and another can rank images for a text. We tested whether mixing them corrects complete groups of four decisions on BiVLC. We grouped 2,933 items by their 1,161 source photographs and held photographs intact across five folds. Each training fold selected the same TROHN-Img checkpoint in both directions. The held-out rule and this single model solved exactly the same 2,031 groups (69.25%). Across the 12 ordered pairs of distinct checkpoints we found 60 incompatible directional decision patterns among 35,196 item-pair combinations. For any fixed pair, even a perfect fallback restricted to these patterns could rescue at most 10 of 2,933 items.

With access to the answers, choosing a whole model per item would solve 2,303 groups. Allowing the two directions to choose different models would add 46 further groups (1.57 percentage points). These counts separate the opportunity for choosing an expert from the additional opportunity for combining their directions. They measure four related CLIP checkpoints. The published tables contain correctness bits rather than score margins, and an unsuccessful strict comparison can include a tie. We closed the cycle-triggered mixing rule on this panel. [Design](bivlc_cycle_protocol_20261120.json) · [Results](bivlc_results.json) · [Author predictions](https://github.com/IMirandaM/BiVLC/tree/84f0d1751e7212d97cdeaeb01353862c09908bcb/results)

## Solving equivalent linear programs

We analyzed 840 released solver trials for 30 linear programs, each written in four equivalent permutations and run with seven solver configurations. For each problem we compared two retrospective choices: its best single configuration across all four presentations and the best configuration separately for each presentation. Both choices achieved 100% verified solves. Mean charged CPU time fell from 0.180599 to 0.180339 seconds. For 29 of the 30 problems, at least one fastest configuration was shared across all four presentations. These are averages over a quick, single-machine panel with about 15.6 ms timing granularity in individual observations. On this evidence we closed the quick-panel selector. Hard mixed-integer or SAT instances could behave differently. [Counts and estimands](lp_results.json) · [Reanalysis](analyze_lp_permutations.py) · [Source measurements](https://github.com/h-a-forster/lpbench/blob/cbd8a53ec250602eff4b53f4e583a4ad24856252/docs/results/quick/units.csv)

## Recompute from author data

The BiVLC fetcher needs Python with requests and PyArrow. Analysis also uses NumPy. The fetcher reads only the metadata portions of the two image-data shards and the authors' four prediction tables. The LP check uses the Python standard library and verifies the pinned table's SHA-256 before scoring.

```sh
python fetch_bivlc_direction_inputs.py /tmp/bivlc-scores
python analyze_bivlc_direction_cycles.py /tmp/bivlc-scores --out /tmp/bivlc-result.json
curl -fL https://raw.githubusercontent.com/h-a-forster/lpbench/cbd8a53ec250602eff4b53f4e583a4ad24856252/docs/results/quick/units.csv -o /tmp/lpbench-units.csv
python analyze_lp_permutations.py /tmp/lpbench-units.csv
```
