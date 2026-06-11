# Complete Fixed-Policy Mechanism Analysis

The historical held-out failure set contains 1,212 utterances. The prior batch-decoding confirmation identifies 1,116 failures. Length-aware decoding in this analysis reclassifies 13 padding-sensitive items as already correct, leaving 1,103 confirmed failures for all recovery comparisons. In total, 109 historical failures do not trigger recovery in this runtime.

Every evaluated utterance uses the first dev-frozen policy configuration (`epsilon=0.0002`, `K=3`). There is no per-item configuration selection and no success-based sampling.

- Targeted repairs: 64/1103 (5.8%; 95% Wilson CI 4.6-7.3%); 18 raw outputs exactly equal the target.
- Five L-infinity/RMS-matched random controls: 64/5515 trials; at least one random repair for 24/1103 utterances.
- Targeted versus any-random paired comparison: targeted-only=60, random-only=20, p=8.58e-06.
- Across the five separately paired random seeds, the largest exact paired p-value is 1.55e-09.
- Mean target CTC loss on 1,093 finite paired items: original 121.971, targeted 112.375, matched random 121.945.
- 10 short utterances have undefined/infinite CTC loss because the target token sequence exceeds the available output frames; they remain included in recognition and acoustic analyses but are excluded from loss summaries.
- Targeted loss < original: one-sided Wilcoxon p=2.48e-112; random versus original: two-sided p=0.26.
- Post-hoc edit-distance target rank does not improve overall: 394/1103 improve, while the median changes from 37.0 to 50.0 (p=1). This rank is not the PGD objective and shows that lower CTC loss need not produce monotonic rank movement under a three-step fixed policy.
- Median signal-to-perturbation ratio: 55.3 dB.

## Perturbation Energy

- 0-0.5k: 2.7%.
- 0.5-2k: 15.5%.
- 2-4k: 22.6%.
- 4-8k: 58.8%.

## Frequency Ablation

| Condition | Repairs | Full-targeted repairs retained | Mean target loss | Paired success p (Holm) |
|---|---:|---:|---:|---:|
| retain_0-0.5k | 1/1103 | 0/64 | 121.698 | 2.86e-17 |
| retain_0.5-2k | 13/1103 | 5/64 | 118.718 | 7.12e-10 |
| retain_2-4k | 32/1103 | 12/64 | 117.467 | 0.00125 |
| retain_4-8k | 47/1103 | 28/64 | 114.299 | 0.12 |
| remove_0-0.5k | 60/1103 | 60/64 | 112.459 | 0.375 |
| remove_0.5-2k | 64/1103 | 52/64 | 112.951 | 1 |
| remove_2-4k | 60/1103 | 45/64 | 113.125 | 1 |
| remove_4-8k | 37/1103 | 18/64 | 115.477 | 0.00545 |

Band-retained conditions keep only the named portion of the original targeted perturbation. Band-removed conditions subtract that portion. Each resulting perturbation is projected back to the same L-infinity bound without amplitude amplification. Thus the ablation tests the contribution of each band under the original budget; it does not claim that spectral energy alone explains recovery.