# Whisper Cross-Model Transfer Probe

The perturbations were optimized for Wav2Vec2, not Whisper. The 50 pairs were historically selected for strong Wav2Vec2 recovery and therefore do not constitute an unbiased Whisper test set.

| Subset | N | Original Whisper accuracy | Assisted Whisper accuracy | Original failures | Repaired | Broken | Rank improved / same / worsened |
|---|---:|---:|---:|---:|---:|---:|---:|
| All selected pairs | 50 | 24.0% | 26.0% | 38 | 2 | 1 | 4 / 40 / 6 |
| Source-reproduced pairs | 41 | 26.8% | 26.8% | 30 | 1 | 1 | 3 / 32 / 6 |

A positive transfer result requires more wrong-to-correct than correct-to-wrong transitions and a systematic improvement in target rank. A null result indicates recognizer-specific perturbations rather than a failure of the target-conditioned translator for its source recognizer.