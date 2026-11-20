# Transfer and expert selection across three domains

## Which use of the same action corresponds across people?

Two people assembled the same toy. Both picked up a screw several times. The screw pickup during the annotated step “attach body” was the **seventh of eight** pickups for one person and the **fifth of eleven** for the other. This is the first eligible pair and role in the sorted analysis, selected independently of each method's result. The fine action name alone does not distinguish these occurrences. [Source dataset](https://openaccess.thecvf.com/content/CVPR2022/html/Sener_Assembly101_A_Large-Scale_Multi-View_Video_Dataset_for_Understanding_Procedural_Activities_CVPR_2022_paper.html)

We grouped camera copies into physical demonstrations and matched the public fine and coarse Assembly101 annotations for 60 assembly recordings. Among 1,521 within-recording fine-action types that repeated inside coarse operations, 691 occurred under multiple coarse operation names. We then required a same-toy pair with different participants, two distinct roles for the fine action in each recording, and a specific role appearing precisely once for that action in both recordings. This yielded 212 directional tests across 34 pairs and 12 toys. These tests reuse recordings and are clustered by toy. [Source coverage](assembly101_coverage.json) · [Eligibility rules](assembly101_occurrence_screen_20261120.json)

For each test, a method receives the source marked occurrence and the two complete fine-action streams. Every target occurrence of the matching fine action can be selected, including events outside coarse action boundaries. The role labels are reserved for constructing and scoring cases. Uniform selection scored 13.32%, relative occurrence order 15.80%, and an edit-distance baseline matching the full sequences through each candidate 33.22%. Shared best scores receive fractional credit. Giving the 34 recording pairs equal weight yielded 12.75%, 14.17% and 33.04%. The comparison uses annotated action names, full future context and no pixels. It establishes a measurable task and a strong baseline. A claim about visual recognition or generalization to new toys requires a separate test. [Fixed comparator](assembly101_forced_alignment_protocol_20261120.json) · [Results](assembly101_alignment_results.json) · [Official annotations](https://github.com/assembly-101/assembly101-annotations)

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
