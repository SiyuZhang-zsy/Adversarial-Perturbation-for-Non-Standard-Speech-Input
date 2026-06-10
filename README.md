# Adversarial Perturbation for Non-Standard Speech Input

This repository contains code, selected results, and paired examples for
experiments on recovering failed non-standard speech inputs with bounded
adversarial perturbations.

## Rebuttal Supplement

New analyses conducted in response to reviewer questions are available in
[`rebuttal_supplement/`](rebuttal_supplement/README.md). They cover:

- a development-learned, held-out-test-frozen parameter policy;
- acoustic and model-space mechanism analyses;
- a Whisper large-v3-turbo baseline;
- direct fixed-hyperparameter recovery against Whisper;
- matched random-noise controls; and
- a null cross-model transfer probe.

## Repository Structure

- `scripts/`: original experiment scripts.
- `results/raw/`: original experiment outputs.
- `audio_examples/`: 50 selected pairs of TORGO-derived original and assisted
  clips.
- `rebuttal_supplement/`: new protocols, aggregate results, figures, and
  analysis scripts.
- `requirements.txt`: original experiment dependencies.

## Data

The experiments use dysarthric speech prepared from
[TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html).
The complete dataset is not redistributed. The repository contains only the
selected paired clips already used as examples; reproduction of the full
experiments requires obtaining TORGO separately and preparing a local manifest.

## Running the Original Experiments

1. Install the packages in `requirements.txt`.
2. Prepare a local manifest with speaker ID, utterance ID, target word, split,
   and waveform path.
3. Configure the paths required by the selected script.
4. Run a script from `scripts/` and inspect its output under `results/`.
