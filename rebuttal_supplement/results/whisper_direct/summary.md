# Direct Whisper Large-v3-turbo Targeted-Recovery Experiment

The evaluation used 60 confirmed full-lexicon failures, balanced at 10 utterances for each of six held-out speakers. For each speaker, we took the first 10 items in dataset order that remained failures when re-decoded by the differentiable implementation. The fixed policy was epsilon=0.0002 and K=3, matching the first configuration in the previously frozen Wav2Vec2 policy order. No Whisper-specific parameter search was used.

- Mapped repairs: 27/60 (45.0%; 95% Wilson CI 33.1-57.5%).
- Exact target transcriptions: 26/60.
- Matched random controls: 0/300.
- Target rank improved for 31/60 items; median rank changed from 26.0 to 4.0 (one-sided Wilcoxon p=1.18e-06).
- Mean target-token loss changed from 5.493 to 4.375 (one-sided Wilcoxon p=8.14e-12).
- Median signal-to-perturbation ratio: 54.5 dB.
- Across five paired random seeds, the largest exact paired p-value was 1.49e-08.

| Speaker | Targeted repairs | Targeted rate | Random repairs / trials |
|---|---:|---:|---:|
| F03 | 9/10 | 90.0% | 0/50 |
| M01 | 1/10 | 10.0% | 0/50 |
| M02 | 7/10 | 70.0% | 0/50 |
| M03 | 6/10 | 60.0% | 0/50 |
| M04 | 2/10 | 20.0% | 0/50 |
| M05 | 2/10 | 20.0% | 0/50 |

The large between-speaker range should be reported rather than collapsed into the aggregate alone. These results establish direct recoverability for a stronger recognizer; they do not imply cross-model transfer.