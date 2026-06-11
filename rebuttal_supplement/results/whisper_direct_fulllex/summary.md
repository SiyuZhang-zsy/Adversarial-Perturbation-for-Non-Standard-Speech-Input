# Direct Whisper Large-v3-turbo Recovery on All Confirmed Full-Lexicon Failures

The full-lexicon held-out test set contains 1,844 utterances. Faster-Whisper produced 1,035 mapped failures; 950 remained failures when re-decoded by the differentiable Whisper implementation used for optimization, while 85 were already correct and therefore did not trigger recovery. We evaluate all 950 confirmed failures, without success-based selection.

The fixed policy was epsilon=0.0002 and K=3, matching the first configuration in the previously frozen Wav2Vec2 policy order. No Whisper-specific parameter search was used.

- Mapped repairs: 354/950 (37.3%; 95% Wilson CI 34.2-40.4%).
- Exact target transcriptions: 306/950.
- Matched random controls: 28/4750 trials, affecting 16/950 utterances (targeted vs. any-random paired p=3.79e-97).
- Target rank improved for 545/950 items; median rank changed from 36.5 to 4.0 (one-sided Wilcoxon p=8.37e-47).
- Mean target-token loss changed from 5.653 to 4.390 (one-sided Wilcoxon p=3.89e-157).
- Median signal-to-perturbation ratio: 52.7 dB.
- Across five paired random seeds, the largest exact paired p-value was 2.69e-101.
- Overall mapped accuracy under the differentiable implementation: 48.5% before recovery and 67.7% after recovery (894/1844 to 1248/1844).

| Speaker | Targeted repairs | Targeted rate | Random repairs / trials |
|---|---:|---:|---:|
| F03 | 80/176 | 45.5% | 12/880 |
| M01 | 64/200 | 32.0% | 2/1000 |
| M02 | 87/197 | 44.2% | 9/985 |
| M03 | 15/33 | 45.5% | 0/165 |
| M04 | 31/154 | 20.1% | 0/770 |
| M05 | 77/190 | 40.5% | 5/950 |

The large between-speaker range should be reported rather than collapsed into the aggregate alone. These results establish direct recoverability for a stronger recognizer; they do not imply cross-model transfer.