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

Run the Python entry points in a CPU Slurm allocation. The supplied-string experiment requires PyTorch, the plan selector requires NumPy, and replay and online control use the standard library. Obtain the four upstream inputs and expected hashes from the replay protocol and `karel_trace_control.py`. Run that script first. The two subsequent scripts require its `summary.json` as a cohort and simulator receipt. Each entry point lists its path arguments with `--help`.
