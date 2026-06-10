# Model-Space Mechanism and Random-Noise Control

Random perturbations are matched to each adversarial perturbation's measured L-infinity and RMS norms.

| Condition | N | Mean target CTC loss | Median target CTC loss | Mapped accuracy | Median target rank |
|---|---:|---:|---:|---:|---:|
| adversarial | 50 | 120.949 | 112.657 | 82.0% | 1.0 |
| original | 50 | 127.207 | 120.005 | 0.0% | 134.5 |
| random | 250 | 127.225 | 122.737 | 0.0% | 128.0 |

Adversarial perturbation reduced target loss relative to the original for 70.0% of paired utterances.

Norm-matched random noise reduced target loss relative to the original for 44.0% of paired utterances.

Paired Wilcoxon test, adversarial target loss < original: W=298.0, p=0.000392.

Paired Wilcoxon test, mean random-noise target loss vs. original: W=600.0, p=0.723.

Saved-waveform recovery reproduced for 41/50 historically selected examples; at least one of five random controls succeeded for 0/50 examples.

Exact paired success comparison: adversarial-only=41, random-only=0, p=9.09e-13.