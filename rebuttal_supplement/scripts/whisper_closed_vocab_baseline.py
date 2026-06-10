from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel


VOCABULARIES = [
    ("10-word", Path("datasets/torgo_10word_split_clean.csv")),
    ("30-word", Path("datasets/torgo_30word_split_clean.csv")),
    ("50-word", Path("datasets/torgo_50word_split_clean.csv")),
    ("100-word", Path("datasets/torgo_100word_split_clean.csv")),
    ("Full lexicon", Path("datasets/torgo_single_word_headmic_split.csv")),
]


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(left: str, right: str) -> int:
    left = normalize_text(left)
    right = normalize_text(right)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def sequence_edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def rank_prediction(
    prediction: str, target: str, lexicon: list[str]
) -> tuple[str, int, int]:
    ranked = sorted(
        ((word, levenshtein(prediction, word)) for word in lexicon),
        key=lambda item: (item[1], item[0]),
    )
    words = [word for word, _ in ranked]
    target_rank = words.index(target) + 1
    return words[0], target_rank, ranked[target_rank - 1][1]


def transcribe_dataset(
    dataset: pd.DataFrame,
    model: WhisperModel,
    output_path: Path,
    language: str,
    resume: bool,
) -> pd.DataFrame:
    rows = []
    completed_paths: set[str] = set()
    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        rows = existing.to_dict("records")
        completed_paths = set(existing["wav_head_path"].astype(str))
        print(f"Resuming with {len(completed_paths)} completed utterances")

    for _, row in dataset.iterrows():
        if str(row["wav_head_path"]) in completed_paths:
            continue
        segments, _ = model.transcribe(
            row["wav_head_path"],
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )
        prediction = normalize_text(" ".join(segment.text for segment in segments))
        rows.append(
            {
                "speaker": row["speaker"],
                "session": row["session"],
                "utt_id": row["utt_id"],
                "target_word": normalize_text(row["target_word"]),
                "split": row["split"],
                "wav_head_path": row["wav_head_path"],
                "raw_prediction": prediction,
                "raw_exact_correct": int(
                    prediction == normalize_text(row["target_word"])
                ),
                "word_errors": sequence_edit_distance(
                    normalize_text(row["target_word"]).split(),
                    prediction.split(),
                ),
            }
        )
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            print(f"[{len(rows)}/{len(dataset)}] checkpoint saved")
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_path, index=False)
    return predictions


def map_vocabulary(
    name: str,
    vocabulary_csv: Path,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    population = pd.read_csv(vocabulary_csv)
    population["target_word"] = population["target_word"].map(normalize_text)
    lexicon = sorted(population["target_word"].unique().tolist())
    prediction_lookup = predictions.set_index("wav_head_path")["raw_prediction"]
    error_lookup = predictions.set_index("wav_head_path")["word_errors"]
    rows = []
    for _, row in population.iterrows():
        prediction = prediction_lookup.loc[row["wav_head_path"]]
        mapped, target_rank, target_distance = rank_prediction(
            prediction, row["target_word"], lexicon
        )
        rows.append(
            {
                "experiment": name,
                "speaker": row["speaker"],
                "split": row["split"],
                "utt_id": row["utt_id"],
                "target_word": row["target_word"],
                "wav_head_path": row["wav_head_path"],
                "raw_prediction": prediction,
                "raw_exact_correct": int(prediction == row["target_word"]),
                "word_errors": int(error_lookup.loc[row["wav_head_path"]]),
                "mapped_prediction": mapped,
                "mapped_correct": int(mapped == row["target_word"]),
                "target_rank": target_rank,
                "target_distance": target_distance,
                "top3_hit": int(target_rank <= 3),
                "top5_hit": int(target_rank <= 5),
                "top10_hit": int(target_rank <= 10),
            }
        )
    return pd.DataFrame(rows)


def summarize(mapped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, split), group in mapped.groupby(["experiment", "split"]):
        failed = group[group["mapped_correct"] == 0]
        rows.append(
            {
                "experiment": experiment,
                "split": split,
                "n": len(group),
                "raw_exact_accuracy": group["raw_exact_correct"].mean(),
                "raw_wer": group["word_errors"].mean(),
                "mapped_top1_accuracy": group["mapped_correct"].mean(),
                "failed_inputs": len(failed),
                "failed_target_recall_at_3": failed["top3_hit"].mean(),
                "failed_target_recall_at_5": failed["top5_hit"].mean(),
                "failed_target_recall_at_10": failed["top10_hit"].mean(),
                "mean_target_rank": group["target_rank"].mean(),
                "median_target_rank": group["target_rank"].median(),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_speaker(mapped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, split, speaker), group in mapped.groupby(
        ["experiment", "split", "speaker"]
    ):
        failed = group[group["mapped_correct"] == 0]
        rows.append(
            {
                "experiment": experiment,
                "split": split,
                "speaker": speaker,
                "n": len(group),
                "raw_exact_accuracy": group["raw_exact_correct"].mean(),
                "raw_wer": group["word_errors"].mean(),
                "mapped_top1_accuracy": group["mapped_correct"].mean(),
                "failed_inputs": len(failed),
                "failed_target_recall_at_5": failed["top5_hit"].mean(),
                "failed_target_recall_at_10": failed["top10_hit"].mean(),
            }
        )
    return pd.DataFrame(rows)


def write_markdown_summary(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Whisper Large-v3-turbo Baseline",
        "",
        "| Vocabulary | Split | N | Raw WER | Mapped top-1 | Failed inputs | "
        "Failed target recall@5 | Failed target recall@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.experiment} | {row.split} | {row.n} | "
            f"{100 * row.raw_wer:.1f}% | {100 * row.mapped_top1_accuracy:.1f}% | "
            f"{row.failed_inputs} | {100 * row.failed_target_recall_at_5:.1f}% | "
            f"{100 * row.failed_target_recall_at_10:.1f}% |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results/whisper_baseline"),
    )
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Skip inference and reuse raw_predictions.csv.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "raw_predictions.csv"

    full_dataset = pd.read_csv(VOCABULARIES[-1][1])
    if args.max_items:
        full_dataset = full_dataset.head(args.max_items)

    if args.reuse_predictions:
        predictions = pd.read_csv(prediction_path)
    else:
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
        predictions = transcribe_dataset(
            full_dataset, model, prediction_path, args.language, args.resume
        )

    mapped_frames = []
    for name, vocabulary_csv in VOCABULARIES:
        population = pd.read_csv(vocabulary_csv)
        available = set(predictions["wav_head_path"])
        population = population[population["wav_head_path"].isin(available)]
        temporary_path = args.output_dir / f"_{name}_population.csv"
        population.to_csv(temporary_path, index=False)
        mapped_frames.append(map_vocabulary(name, temporary_path, predictions))
        temporary_path.unlink()

    mapped = pd.concat(mapped_frames, ignore_index=True)
    summary = summarize(mapped)
    speaker_summary = summarize_by_speaker(mapped)
    mapped.to_csv(args.output_dir / "closed_vocab_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    speaker_summary.to_csv(args.output_dir / "speaker_summary.csv", index=False)
    write_markdown_summary(summary, args.output_dir / "summary.md")
    print(summary.to_string(index=False))
    print(f"Saved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
