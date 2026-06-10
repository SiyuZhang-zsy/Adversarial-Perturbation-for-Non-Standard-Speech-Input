# Direct Whisper Large-v3-turbo Recovery on All Confirmed 10-Word Failures

The 10-word held-out test set contains 136 utterances. Faster-Whisper produced 39 mapped failures; 33 remained failures when re-decoded by the differentiable Whisper implementation used for optimization, while six were already correct and therefore did not trigger recovery. We evaluate all 33 confirmed failures, without success-based selection.

The fixed policy was epsilon=0.0002 and K=3, matching the first configuration in the previously frozen Wav2Vec2 policy order. No Whisper-specific parameter search was used.

- Mapped repairs: 18/33 (54.5%; 95% Wilson CI 38.0-70.2%).
- Exact target transcriptions: 15/33.
- Matched random controls: 3/165 trials, affecting 2/33 utterances (targeted vs. any-random paired p=0.000145).
- Target rank improved for 20/33 items; median rank changed from 3.0 to 1.0 (one-sided Wilcoxon p=4.13e-05).
- Mean target-token loss changed from 4.848 to 3.941 (one-sided Wilcoxon p=1.16e-10).
- Median signal-to-perturbation ratio: 56.2 dB.
- Across five paired random seeds, the largest exact paired p-value was 7.63e-05.
- Overall mapped accuracy under the differentiable implementation: 75.7% before recovery and 89.0% after recovery (103/136 to 121/136).

| Speaker | Targeted repairs | Targeted rate | Random repairs / trials |
|---|---:|---:|---:|
| F03 | 3/5 | 60.0% | 2/25 |
| M01 | 3/8 | 37.5% | 1/40 |
| M02 | 6/6 | 100.0% | 0/30 |
| M03 | 2/2 | 100.0% | 0/10 |
| M04 | 2/5 | 40.0% | 0/25 |
| M05 | 2/7 | 28.6% | 0/35 |

The large between-speaker range should be reported rather than collapsed into the aggregate alone. These results establish direct recoverability for a stronger recognizer; they do not imply cross-model transfer.