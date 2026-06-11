# Unified Hugging Face Whisper Large-v3-turbo Recovery

The same Hugging Face differentiable Whisper large-v3-turbo runtime is used for baseline decoding, gradient optimization, and assisted decoding on all 1,844 held-out full-lexicon utterances. The unified baseline contains 1,005 mapped failures. We evaluate all 1,005 failures with one fixed policy, without success-based selection.

The fixed policy was epsilon=0.0002 and K=3, matching the first configuration in the previously frozen Wav2Vec2 policy order. No Whisper-specific parameter search was used.

- Mapped repairs: 396/1005 (39.4%; 95% Wilson CI 36.4-42.5%).
- Exact target transcriptions: 347/1005.
- Matched random controls: 47/5025 trials, affecting 23/1005 utterances (targeted vs. any-random paired p=6.88e-105).
- Target rank improved for 588/1005 items; median rank changed from 36.0 to 4.0 (one-sided Wilcoxon p=7.19e-51).
- Mean target-token loss changed from 5.600 to 4.356 (one-sided Wilcoxon p=7e-166).
- Median signal-to-perturbation ratio: 52.6 dB.
- Across five paired random seeds, the largest exact paired p-value was 1.94e-109.
- Overall mapped accuracy under the differentiable implementation: 45.5% before recovery and 67.0% after recovery (839/1844 to 1235/1844).

| Speaker | Targeted repairs | Targeted rate | Random repairs / trials |
|---|---:|---:|---:|
| F03 | 95/193 | 49.2% | 21/965 |
| M01 | 65/204 | 31.9% | 5/1020 |
| M02 | 95/207 | 45.9% | 12/1035 |
| M03 | 17/36 | 47.2% | 0/180 |
| M04 | 35/160 | 21.9% | 0/800 |
| M05 | 89/205 | 43.4% | 9/1025 |

The large between-speaker range should be reported rather than collapsed into the aggregate alone. These results establish direct recoverability for a stronger recognizer; they do not imply cross-model transfer.