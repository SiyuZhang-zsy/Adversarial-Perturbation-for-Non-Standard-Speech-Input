# Whisper Success-Conditioned Mechanism Analysis

This analysis includes all 354 mapped recoveries produced by the complete fixed-policy Whisper large-v3-turbo experiment. It characterizes successful transformations; efficacy and generic-noise specificity remain established on all 950 confirmed failures.

- Exact target transcription: 306/354.
- Mean target loss among successes: 5.332 to 3.869 (one-sided Wilcoxon p=4.54e-60).
- Median signal-to-perturbation ratio: 50.8 dB.
- Mean 4-8 kHz energy share: 42.3%.
- However, 4-8 kHz energy share does not differ between successful and unsuccessful perturbations (42.3% vs. 41.9%; Holm p=0.497), so high-frequency energy alone is not sufficient for recovery.
- Among these successful items, matched random controls repair 23/1770 trials and at least one of five random controls repairs 13/354 items. This contrast is success-conditioned and is descriptive; the primary random-control test uses all 950 failures.

## Frequency Ablation on Successful Recoveries

- Retaining only 4-8 kHz preserves 115/354 recoveries (32.5%).
- Removing 4-8 kHz preserves 232/354 recoveries (65.5%).
- Retaining only 2-4 kHz preserves 127/354 (35.9%); removing it preserves 229/354 (64.7%).
- There is no paired evidence that 4-8 kHz is more important than 2-4 kHz (two-sided retain p=0.195; remove p=0.788).

| Condition | Recoveries preserved | Preservation rate | Mean target loss |
|---|---:|---:|---:|
| retain_0-0.5k | 9/354 | 2.5% | 5.222 |
| remove_0-0.5k | 336/354 | 94.9% | 3.894 |
| retain_0.5-2k | 90/354 | 25.4% | 4.689 |
| remove_0.5-2k | 247/354 | 69.8% | 4.037 |
| retain_2-4k | 127/354 | 35.9% | 4.510 |
| remove_2-4k | 229/354 | 64.7% | 4.084 |
| retain_4-8k | 115/354 | 32.5% | 4.551 |
| remove_4-8k | 232/354 | 65.5% | 4.123 |

The perturbation is additive and length-preserving. Frequency conditions are projected to the original L-infinity bound without amplitude amplification. The ablation supports a functional role for distributed contributions across approximately 0.5-8 kHz. No single band is necessary or sufficient, and high-frequency energy share alone does not distinguish successful from unsuccessful attacks.