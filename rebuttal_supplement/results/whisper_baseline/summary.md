# Legacy Faster-Whisper Contextualization

These CTranslate2/Faster-Whisper results are retained for transparency and
bounded-vocabulary sensitivity analyses. They are not used as the baseline for
the unified full-lexicon recovery claim. That claim uses the Hugging Face
differentiable runtime throughout and reports 45.5% baseline accuracy; see
[`../whisper_hf_baseline/`](../whisper_hf_baseline/).

| Vocabulary | Split | N | Raw WER | Mapped top-1 | Failed inputs | Failed target recall@5 | Failed target recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| 10-word | dev | 24 | 45.8% | 87.5% | 3 | 66.7% | 100.0% |
| 10-word | test | 136 | 66.9% | 71.3% | 39 | 66.7% | 100.0% |
| 100-word | dev | 144 | 65.3% | 48.6% | 74 | 32.4% | 43.2% |
| 100-word | test | 880 | 101.7% | 48.9% | 450 | 23.8% | 41.3% |
| 30-word | dev | 65 | 63.1% | 61.5% | 25 | 44.0% | 64.0% |
| 30-word | test | 415 | 76.4% | 53.0% | 195 | 39.0% | 62.1% |
| 50-word | dev | 90 | 60.0% | 55.6% | 40 | 30.0% | 45.0% |
| 50-word | test | 570 | 92.5% | 51.9% | 274 | 30.7% | 47.1% |
| Full lexicon | dev | 270 | 54.1% | 55.9% | 119 | 19.3% | 29.4% |
| Full lexicon | test | 1844 | 106.9% | 43.9% | 1035 | 18.6% | 26.4% |
