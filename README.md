This repository contains code and result files for experiments on recovering failed non-standard speech inputs with bounded adversarial perturbation.

## Structure

- `scripts/` — main scripts
- `results/summaries/` — summary files
- `results/raw/` — raw outputs
- `results/examples/` — selected example files
- `audio_examples/` — paired original and assisted audio examples
- `requirements.txt` — dependencies

## Requirements

Install packages from:

- `requirements.txt`

## Expected data format

The scripts expect a local CSV manifest containing at least:

- speaker ID
- utterance ID
- target word
- split
- waveform path

The original dataset is not redistributed here.

## Running

1. Prepare the local manifest file
2. Update paths in the script config if needed
3. Run the desired script from `scripts/`
4. Check outputs in `results/`
