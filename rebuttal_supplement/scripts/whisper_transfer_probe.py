from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from faster_whisper import WhisperModel
from scipy.stats import binomtest, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def map_prediction(
    prediction: str, target: str, lexicon: list[str]
) -> tuple[str, int, int]:
    ranked = sorted(
        ((word, levenshtein(prediction, word)) for word in lexicon),
        key=lambda item: (item[1], item[0]),
    )
    words = [word for word, _ in ranked]
    target_rank = words.index(target) + 1
    return words[0], target_rank, ranked[target_rank - 1][1]


def transcribe(model: WhisperModel, audio_path: str, language: str) -> str:
    segments, _ = model.transcribe(
        audio_path,
        language=language,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=False,
    )
    return normalize_text(" ".join(segment.text for segment in segments))


def paired_exact_pvalue(
    original_correct: pd.Series, assisted_correct: pd.Series
) -> tuple[int, int, float]:
    repaired = int(((original_correct == 0) & (assisted_correct == 1)).sum())
    broken = int(((original_correct == 1) & (assisted_correct == 0)).sum())
    discordant = repaired + broken
    p_value = (
        binomtest(repaired, discordant, p=0.5).pvalue if discordant else 1.0
    )
    return repaired, broken, p_value


def summarize_subset(name: str, pairs: pd.DataFrame) -> dict[str, float | int | str]:
    repaired, broken, top1_p = paired_exact_pvalue(
        pairs["original_mapped_correct"], pairs["assisted_mapped_correct"]
    )
    original_wrong = pairs[pairs["original_mapped_correct"] == 0]
    rank_delta = pairs["assisted_target_rank"] - pairs["original_target_rank"]
    distance_delta = (
        pairs["assisted_target_distance"] - pairs["original_target_distance"]
    )

    nonzero_rank = rank_delta[rank_delta != 0]
    if len(nonzero_rank):
        rank_test = wilcoxon(
            pairs["assisted_target_rank"],
            pairs["original_target_rank"],
            alternative="less",
        )
        rank_p = float(rank_test.pvalue)
    else:
        rank_p = 1.0

    return {
        "subset": name,
        "n": len(pairs),
        "original_mapped_accuracy": float(
            pairs["original_mapped_correct"].mean()
        ),
        "assisted_mapped_accuracy": float(
            pairs["assisted_mapped_correct"].mean()
        ),
        "original_failures": len(original_wrong),
        "repaired": repaired,
        "broken": broken,
        "repair_rate_on_original_failures": (
            repaired / len(original_wrong) if len(original_wrong) else float("nan")
        ),
        "top1_exact_p": top1_p,
        "rank_improved": int((rank_delta < 0).sum()),
        "rank_same": int((rank_delta == 0).sum()),
        "rank_worsened": int((rank_delta > 0).sum()),
        "mean_rank_change": float(rank_delta.mean()),
        "median_rank_change": float(rank_delta.median()),
        "rank_improvement_p": rank_p,
        "distance_improved": int((distance_delta < 0).sum()),
        "distance_same": int((distance_delta == 0).sum()),
        "distance_worsened": int((distance_delta > 0).sum()),
        "mean_distance_change": float(distance_delta.mean()),
    }


def plot_transitions(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    x = np.arange(len(summary))
    width = 0.34
    axes[0].bar(
        x - width / 2,
        100 * summary["original_mapped_accuracy"],
        width,
        label="Original",
    )
    axes[0].bar(
        x + width / 2,
        100 * summary["assisted_mapped_accuracy"],
        width,
        label="Assisted",
    )
    axes[0].set_xticks(x, summary["subset"], rotation=12)
    axes[0].set_ylabel("Whisper mapped top-1 accuracy (%)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, summary["repaired"], width, label="Wrong to correct")
    axes[1].bar(x + width / 2, summary["broken"], width, label="Correct to wrong")
    axes[1].set_xticks(x, summary["subset"], rotation=12)
    axes[1].set_ylabel("Paired utterances")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_summary(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Whisper Cross-Model Transfer Probe",
        "",
        "The perturbations were optimized for Wav2Vec2, not Whisper. The 50 pairs "
        "were historically selected for strong Wav2Vec2 recovery and therefore do "
        "not constitute an unbiased Whisper test set.",
        "",
        "| Subset | N | Original Whisper accuracy | Assisted Whisper accuracy | "
        "Original failures | Repaired | Broken | Rank improved / same / worsened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.subset} | {row.n} | "
            f"{100 * row.original_mapped_accuracy:.1f}% | "
            f"{100 * row.assisted_mapped_accuracy:.1f}% | "
            f"{row.original_failures} | {row.repaired} | {row.broken} | "
            f"{row.rank_improved} / {row.rank_same} / {row.rank_worsened} |"
        )
    lines.extend(
        [
            "",
            "A positive transfer result requires more wrong-to-correct than "
            "correct-to-wrong transitions and a systematic improvement in target rank. "
            "A null result indicates recognizer-specific perturbations rather than a "
            "failure of the target-conditioned translator for its source recognizer.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/audio_examples/audio_example_manifest.csv"),
    )
    parser.add_argument(
        "--lexicon-csv",
        type=Path,
        default=Path("datasets/torgo_single_word_headmic_split.csv"),
    )
    parser.add_argument(
        "--source-reproduction-csv",
        type=Path,
        default=Path(
            "analysis/results/model_mechanism_random_control/"
            "model_mechanism_results.csv"
        ),
    )
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results/whisper_transfer_probe"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    lexicon = sorted(
        pd.read_csv(args.lexicon_csv)["target_word"]
        .astype(str)
        .map(normalize_text)
        .unique()
        .tolist()
    )

    source_results = pd.read_csv(args.source_reproduction_csv)
    source_reproduced = (
        source_results[source_results["condition"] == "adversarial"]
        .set_index("index")["mapped_correct"]
        .astype(int)
    )

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    rows = []
    for item_number, row in manifest.iterrows():
        target = normalize_text(row["target_word"])
        original_prediction = transcribe(
            model, row["original_audio"], args.language
        )
        assisted_prediction = transcribe(
            model, row["assisted_audio"], args.language
        )
        original_mapped, original_rank, original_distance = map_prediction(
            original_prediction, target, lexicon
        )
        assisted_mapped, assisted_rank, assisted_distance = map_prediction(
            assisted_prediction, target, lexicon
        )
        rows.append(
            {
                "index": int(row["index"]),
                "speaker": row["speaker"],
                "split": row["split"],
                "utt_id": row["utt_id"],
                "target_word": target,
                "source_wav2vec2_reproduced": int(
                    source_reproduced.get(int(row["index"]), 0)
                ),
                "original_raw_prediction": original_prediction,
                "assisted_raw_prediction": assisted_prediction,
                "original_mapped_prediction": original_mapped,
                "assisted_mapped_prediction": assisted_mapped,
                "original_mapped_correct": int(original_mapped == target),
                "assisted_mapped_correct": int(assisted_mapped == target),
                "original_target_rank": original_rank,
                "assisted_target_rank": assisted_rank,
                "target_rank_change": assisted_rank - original_rank,
                "original_target_distance": original_distance,
                "assisted_target_distance": assisted_distance,
                "target_distance_change": assisted_distance - original_distance,
            }
        )
        print(
            f"[{item_number + 1}/{len(manifest)}] {row['speaker']} {target}: "
            f"{original_mapped} -> {assisted_mapped}"
        )

    pairs = pd.DataFrame(rows)
    pairs.to_csv(args.output_dir / "paired_predictions.csv", index=False)

    summary_rows = [summarize_subset("All selected pairs", pairs)]
    reproduced = pairs[pairs["source_wav2vec2_reproduced"] == 1]
    summary_rows.append(
        summarize_subset("Source-reproduced pairs", reproduced)
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_transitions(summary, args.output_dir / "transfer_transitions.png")
    write_summary(summary, args.output_dir / "summary.md")

    print(summary.to_string(index=False))
    print(f"Saved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
