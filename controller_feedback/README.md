# Does the next action depend on what came back?

In the first experiment, a probe returns one of five values. The learner sees that value and an ordered string of operations, then chooses which leaf to call. Four single operations and twelve ordered pairs supply the training examples. Each of five independently generated worlds has four distinct noncommuting operations. The tests exhaust four held-out pairs and 36 three-operation strings per world, including every possible probe value.

| Method | Held-out pairs | New triples |
|---|---:|---:|
| Best policy blind to the returned value | 20% | 20% |
| Positional multilayer perceptron | 25% | 21.67% |
| Single shared GRU | 62% | 68% |
| Interpret the given string with tables fitted from single-step examples | 100% | 100% |

The table interpreter establishes a shortcut to perfect accuracy. The examples supply the order of execution and identify each primitive separately. This benchmark therefore cannot establish that a learner discovered how to organize its own computation. The neural controls differ in parameter count and exposure of positional inputs. The five-world results are descriptive. [Protocol](controller_feedback_protocol_20261120.json) · [Aggregate results and splits](supplied_schedule_results.json)

## Demonstrations that follow different paths

The Karel experiments hide program text at prediction time. Each problem supplies several observed demonstrations of the same program in different grid worlds. Withhold one world, predict from the remaining demonstrations, and score the final grid, including location, orientation and marker counts. This provides richer supervision than the classic input/output-only synthesis task. The selected benchmark worlds were designed to exercise program branches. [Dataset](https://msr-redmond.github.io/karel-dataset/) · [Bunel et al.](https://arxiv.org/abs/1805.04276v2)

Three prespecified comparisons use the same inputs and outcome definition.

- Replay complete demonstrated action sequences, compare exact paths with successful endpoints, and compute how often any supported sequence could have worked. [Fixed replay protocol and source checksums](karel_trace_control_20261120.json)
- Try complete demonstrated sequences in a known simulator before acting. Compare legality, local predicate agreement and a ranker fitted on disjoint program folds. [Plan selection](karel_plan_selection_protocol_20261120.json)
- Predict one action at a time from the other demonstrations using a flat memory of observed actions and local sensor readings. Compare fresh readings after each step with readings fixed at their initial values and with readings masked throughout. [Online control](karel_online_control_20261120.json)

A separate paired experiment holds the five demonstrations, world and expert action prefix fixed. Just after the first successful move, set the marker count on the arrived-at cell to zero or one. Ask the expert program and each predictor for the next primitive in both worlds. Score both answers together, separating cases where the correct action changes from those where it stays the same. A predictor with identical inputs in both arms must miss at least one answer whenever the expert switches. Because the destination was previously unvisited, local perception alone cannot tell when its marker was introduced. [Paired protocol](karel_return_pairs_20261120.json)

Run the Python entry points in a CPU Slurm allocation. The supplied-string experiment requires PyTorch, the plan selector requires NumPy, and replay and online control use the standard library. Obtain the four upstream inputs and expected hashes from the replay protocol and `karel_trace_control.py`. Run that script first. The two subsequent scripts require its `summary.json` as a cohort and simulator receipt. Each entry point lists its path arguments with `--help`.

## Development results

We evaluated 2,500 records covering 2,045 distinct program token sequences and 15,000 worlds. All 1,000 program strings in the independent author reference were found in the mirror, in a different order. The interpreter reproduced every recorded primitive and endpoint on the original worlds. The figures below describe this validation bank. [Machine-readable results](karel_results.json)

| How a held-out world is solved | Exact final grid |
|---|---:|
| Copy a uniformly chosen demonstration | 67.42% |
| Choose the most common of five demonstrated plans | 70.34% |
| Simulate the plans and choose by legality, sensor agreement and frequency | 82.15% |
| Fit a cross-program plan ranker using those three features | 78.38% |
| Select a demonstrated plan with hindsight | 85.97% |
| Predict each next action using live local readings | 35.60% |
| Same predictor with the initial readings held fixed | 52.59% |
| Same predictor with readings masked | 49.51% |

Simulation-based selectors can inspect the whole initial world and try multiple plans before acting. The live local policy works under a different information and computation budget. It succeeded on 429 of the 2,105 worlds where executing any one complete demonstration would fail. On 7,984 other worlds it failed despite a successful demonstrated plan being available. The learned plan ranker missed its prespecified retention criterion. Of the worlds where a complete demonstration exactly matched all sensor readings through the terminal decision, all 3,035 matched its path and endpoint.

| First-arrival marker intervention | Pairs | Live predictor changed its answer | Live predictor correct on both | Initial-only correct on both |
|---|---:|---:|---:|---:|
| Expert next command changes | 1,435 | 80.07% | 42.86% | 0% |
| Expert next command stays the same | 13,560 | 63.20% | 33.12% | 95.52% |

Five further pairs had no next command within the interpreter limit in the one-marker arm. The primary analysis scores the command requested by the expert, including a request that would be illegal. An additional check retained only pairs for which both requested commands could legally execute or stop. Among 10,994 such stay pairs, the live predictor still changed its answer in 6,598 (60.01%). All 615 joint successes on switch pairs were in the both-legal group. This test supplies the correct past actions up to the first move. It measures the choice at that boundary. The complete-grid success rates above come from separate rollouts on unmodified worlds.

### Closest comparisons

Sun et al. already evaluated a sequential predictor that takes demonstrations and scene predicates and outputs the next Karel action. Our paired measurement asks whether a changed marker causes an appropriate change in that decision, including cases where the expert should continue unchanged. [Sun et al., ICML 2018](https://proceedings.mlr.press/v80/sun18a.html) · [Time-step prediction baseline, supplement §C](https://proceedings.mlr.press/v80/sun18a/sun18a-supp.pdf)

Compositional generalization and learned state-dependent selection of operations also have direct precedents. A conclusion about reusable hierarchy would require performance on new callers or tasks against a flat controller with the same observations and comparable supervision. [Chang et al.](https://arxiv.org/abs/1807.04640) · [Kong et al.](https://proceedings.mlr.press/v306/kong26c.html)
