# Unified Hugging Face Whisper Baseline

The same Hugging Face differentiable Whisper large-v3-turbo FP16 runtime is used for baseline decoding, gradient optimization, and assisted decoding.

- Held-out full-lexicon utterances: 1844.
- Baseline mapped correct: 839/1844 (45.5%).
- Baseline mapped failures: 1005.

For implementation transparency, Faster-Whisper/CTranslate2 and the Hugging Face runtime agree on 754 correct and 950 failed inputs. Faster-Whisper alone is correct on 55; Hugging Face alone is correct on 85. The unified recovery analysis uses only the Hugging Face runtime and does not mix these outcomes.
