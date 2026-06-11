# Whisper Success-Conditioned Mechanism Analysis

This analysis includes all 396 mapped recoveries produced by the complete fixed-policy Whisper large-v3-turbo experiment. It characterizes successful transformations; efficacy and generic-noise specificity remain established on all 1,005 confirmed failures.

- Exact target transcription: 347/396.
- Mean target loss among successes: 5.250 to 3.838 (one-sided Wilcoxon p=6.19e-67).
- Median signal-to-perturbation ratio: 50.6 dB.
- Mean 4-8 kHz energy share: 42.4%.
- However, 4-8 kHz energy share does not differ between successful and unsuccessful perturbations (42.4% vs. 41.9%; Holm p=0.349), so high-frequency energy alone is not sufficient for recovery.
- Among these successful items, matched random controls repair 36/1980 trials and at least one of five random controls repairs 18/396 items. This contrast is success-conditioned and is descriptive; the primary random-control test uses all 1,005 failures.

## Frequency Ablation on Successful Recoveries

- Retaining only 4-8 kHz preserves 151/396 recoveries (38.1%).
- Removing 4-8 kHz preserves 271/396 recoveries (68.4%).
- Retaining only 2-4 kHz preserves 160/396 (40.4%); removing it preserves 266/396 (67.2%).
- There is no paired evidence that 4-8 kHz is more important than 2-4 kHz (two-sided retain p=0.368; remove p=0.603).

| Condition | Recoveries preserved | Preservation rate | Mean target loss |
|---|---:|---:|---:|
| retain_0-0.5k | 17/396 | 4.3% | 5.142 |
| remove_0-0.5k | 377/396 | 95.2% | 3.862 |
| retain_0.5-2k | 118/396 | 29.8% | 4.622 |
| remove_0.5-2k | 286/396 | 72.2% | 3.999 |
| retain_2-4k | 160/396 | 40.4% | 4.448 |
| remove_2-4k | 266/396 | 67.2% | 4.040 |
| retain_4-8k | 151/396 | 38.1% | 4.488 |
| remove_4-8k | 271/396 | 68.4% | 4.078 |

The perturbation is additive and length-preserving. Frequency conditions are projected to the original L-infinity bound without amplitude amplification. The ablation supports a functional role for distributed contributions across approximately 0.5-8 kHz. No single band is necessary or sufficient, and high-frequency energy share alone does not distinguish successful from unsuccessful attacks.