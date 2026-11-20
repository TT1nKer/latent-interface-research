# Which part of a demonstrated action transfers?

A policy can choose the correct action even when several occurrences of it have different meanings. We tested three ways those differences might matter. The first changes a frozen policy's explanation while it drives. The second gives its choice of occurrence a new observable consequence. The third changes which demonstration the policy gets.

## Moving weight between indistinguishable actions

The frozen retriever votes for a motor primitive by summing its weights across all support positions showing that primitive. Its recurrent state receives local observations and the action it executed. The identity of the source position is never passed forward. Holding each action's total fixed therefore preserves both the real-arithmetic action distribution and the subsequent state trajectory. This argument applies to this architecture. A decoder that reads the selected source features again can behave differently.

We redistributed the available mass uniformly over all real support positions with the same observed primitive, including STOP. This transformation used no target action, program statement or correspondence labels. We reran the single frozen seed on 764 held-out programs (4,584 worlds). Native float32 reaggregation altered 17,261 out of 22,345 score vectors, with maximum absolute change 0.00000132. **Zero action choices changed.** All archived stops, crashes, action counts and successes matched. Both policies solved the same 3,816 worlds.

On reference histories, the original position weights located the correct statement 97.70% of the time in 524 programs with ambiguous same-action positions. Uniform redistribution reduced this to 45.61%, counting exact ties fractionally and giving each program equal weight. This is a post-hoc construct-validity check with one checkpoint. The large change establishes how little this particular whole-task measurement can tell us about correspondence. The learned rankings themselves retain substantial information about location. Related work already formalizes summation over repeated copy tokens and non-identifiability of attention. [Intervention](karel_equal_action_mass_20261120.json) · [Measurements](position_mass_results.json) · [See et al.](https://aclanthology.org/P17-1099/) · [Brunner et al.](https://openreview.net/forum?id=BJg1f6EFDB)

## Transfer an extra event to a new execution

Hold the motor controller fixed. Two statements in a program emit the same primitive, for example two different calls to move. Mark the first statement with a pulse in all five demonstrations. In the paired mission mark the second instead. Only pulse bits and ordinary action and sensor traces reach the policy. Neither the source statement names nor the target time to pulse are provided. At test time every readout receives the common controller's chosen primitive, including its mistakes. A world succeeds if both missions stop at its exact intended grid and emit the correct timestamp lists. This synthetic output isolates a possible downstream use of a previously learned ranking. Its pulses cannot change the physical grid. [Design and thresholds](karel_event_transfer_20261120.json)

Prior work already transfers temporal metadata through video alignment, uses selective reads of repeated copy tokens, and learns to apply a demonstrated edit elsewhere. A credible new result needs to establish when these distinctions improve measured execution against simple sequence alignment, preferably in a setting with naturally occurring event semantics. [Temporal Cycle-Consistency](https://debidatta.github.io/tcc_interactive_paper/index.html) · [CopyNet](https://aclanthology.org/P16-1154/) · [Learning to Represent Edits](https://openreview.net/forum?id=BJl6AjC5F7)

## Does the same fifth demonstration help both policies?

We verified all ten native executions for 768 preselected programs and removed two with repeated code. On each of the remaining 766 programs, four demonstrations formed the initial memory. Two others served as candidate additions, two supplied calibration endpoints and the final two supplied unseen test worlds. Roles were assigned by a hash before scoring. The neural retriever and the support-weighted edit aligner were frozen. Success required independent autonomous execution, explicit STOP and the exact final grid. [Design](karel_teaching_transfer_20261120.json) · [Full estimates](teaching_results.json)

| Intervention | Mean change in success | Paired 95% interval |
|---|---:|---:|
| Give the retriever a random fifth demonstration | +2.12 pp | +1.11 to +3.13 |
| Give the edit aligner a random fifth demonstration | +2.84 pp | +1.99 to +3.72 |
| Calibrate the fifth demonstration for the receiving policy versus calibrating it for the other policy | +0.41 pp | +0.20 to +0.64 |
| Use the acquired trajectory to pick a policy after adding it to memory | +0.39 pp | +0.07 to +0.75 |

The third row measures specialization. It requires observing both candidates and two validation targets. Only two programs yielded strictly opposite rankings of the candidates across the two receivers on the two test worlds. The fourth row instead uses a single acquired demonstration. It chooses the neural policy only if that policy alone correctly predicted the acquired endpoint from the original four examples. All other cases retain the edit policy. Both experiments fell below the prespecified two-point thresholds. This closes these two specific teaching and routing rules. Two candidates and two test worlds give coarse measurements of each program's preferences. [Factorial design](karel_demo_dual_role_20261120.json) · [Factorial results](demo_roles_results.json) · [Prior Karel active querying](https://arxiv.org/abs/2205.07857)

Intervals resample whole programs. Results are conditional on the fitted models and these task distributions. The teaching cohort is disjoint in exact program code from the earlier 764-program cohort and development data. Both cohorts share a Karel generator family.
