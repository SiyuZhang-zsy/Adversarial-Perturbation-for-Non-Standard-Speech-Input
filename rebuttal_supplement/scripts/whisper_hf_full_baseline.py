from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_direct_attack_pilot import (
    load_audio,
    normalize_text,
    rank_prediction,
    transcribe,
    verify_frontend,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="openai/whisper-large-v3-turbo"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("datasets/torgo_single_word_headmic_split.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "analysis/results/whisper_hf_full_baseline/"
            "closed_vocab_predictions.csv"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(args.dataset_csv)
    dataset = dataset[dataset["split"] == "test"].copy()
    dataset["target_word"] = dataset["target_word"].map(normalize_text)
    lexicon = sorted(dataset["target_word"].unique().tolist())

    device = torch.device(args.device)
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(
        args.model,
        language="en",
        task="transcribe",
        local_files_only=args.local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    first_audio = load_audio(dataset.iloc[0]["wav_head_path"])
    difference = verify_frontend(
        first_audio, processor.feature_extractor, device
    )
    print(f"Frontend max absolute difference: {difference:.3g}")

    rows: list[dict[str, object]] = []
    completed: set[str] = set()
    if args.resume and args.output_csv.exists():
        existing = pd.read_csv(args.output_csv)
        rows = existing.to_dict("records")
        completed = set(existing["wav_head_path"].astype(str))
        print(f"Resuming with {len(completed)} completed utterances")

    for index, row in enumerate(dataset.itertuples(index=False), start=1):
        if str(row.wav_head_path) in completed:
            continue
        waveform = load_audio(row.wav_head_path)
        prediction = transcribe(
            model, processor, waveform, device, model_dtype
        )
        mapped, rank, distance = rank_prediction(
            prediction, row.target_word, lexicon
        )
        rows.append(
            {
                "experiment": "Full lexicon",
                "speaker": row.speaker,
                "session": row.session,
                "utt_id": row.utt_id,
                "target_word": row.target_word,
                "split": row.split,
                "wav_head_path": row.wav_head_path,
                "raw_prediction": prediction,
                "mapped_prediction": mapped,
                "mapped_correct": int(mapped == row.target_word),
                "target_rank": rank,
                "target_distance": distance,
            }
        )
        if index % 25 == 0 or index == len(dataset):
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
            print(f"Completed {index}/{len(dataset)}")

    predictions = pd.DataFrame(rows)
    predictions.to_csv(args.output_csv, index=False)
    correct = int(predictions["mapped_correct"].sum())
    total = len(predictions)
    print(
        f"Hugging Face differentiable Whisper mapped accuracy: "
        f"{correct}/{total} ({100 * correct / total:.1f}%)"
    )
    print(f"Mapped failures: {total - correct}")


if __name__ == "__main__":
    main()
