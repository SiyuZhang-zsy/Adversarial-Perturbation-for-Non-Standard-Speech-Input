from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.stats import binomtest, wilcoxon
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

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


def rank_in_lexicon(prediction: str, target: str, lexicon: list[str]) -> int:
    ranked = sorted(lexicon, key=lambda word: (levenshtein(prediction, word), word))
    return ranked.index(target) + 1


def mapped_word(prediction: str, lexicon: list[str]) -> str:
    return min(lexicon, key=lambda word: (levenshtein(prediction, word), word))


def read_wav(path: Path, target_sample_rate: int = 16000) -> np.ndarray:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim == 2:
        audio = audio.astype(np.float64).mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    if sample_rate != target_sample_rate:
        audio = resample_poly(audio, target_sample_rate, sample_rate)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def norm_matched_random_noise(
    length: int,
    target_linf: float,
    target_rms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if target_linf <= 0 or target_rms <= 0:
        return np.zeros(length, dtype=np.float32)
    target_ratio = min(target_rms / target_linf, 1.0)
    base = rng.random(length)
    signs = rng.choice(np.array([-1.0, 1.0]), size=length)
    base[np.argmax(base)] = 1.0

    low, high = 0.0, 32.0
    for _ in range(50):
        exponent = (low + high) / 2
        ratio = rms(np.power(base, exponent))
        if ratio > target_ratio:
            low = exponent
        else:
            high = exponent
    noise = signs * np.power(base, (low + high) / 2) * target_linf
    return noise.astype(np.float32)


@torch.no_grad()
def score_audio(
    audio: np.ndarray,
    target: str,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
) -> tuple[float, str]:
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
    )
    labels = processor.tokenizer(
        target,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    outputs = model(
        input_values=inputs.input_values.to(device),
        attention_mask=(
            inputs.attention_mask.to(device)
            if "attention_mask" in inputs
            else None
        ),
        labels=labels.to(device),
    )
    prediction_ids = outputs.logits.argmax(dim=-1)
    prediction = normalize_text(processor.batch_decode(prediction_ids)[0])
    return float(outputs.loss.item()), prediction


def plot_loss_changes(results: pd.DataFrame, output_path: Path) -> None:
    plot_df = results[
        results["condition"].isin(["original", "adversarial", "random"])
    ]
    order = ["original", "random", "adversarial"]
    values = [
        plot_df.loc[plot_df["condition"] == condition, "target_ctc_loss"]
        for condition in order
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.boxplot(values, tick_labels=order, showfliers=False)
    ax.set_ylabel("Target CTC loss (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_summary(results: pd.DataFrame, output_path: Path) -> None:
    aggregate = (
        results.groupby("condition")
        .agg(
            n=("target_ctc_loss", "size"),
            mean_target_loss=("target_ctc_loss", "mean"),
            median_target_loss=("target_ctc_loss", "median"),
            mapped_accuracy=("mapped_correct", "mean"),
            median_target_rank=("target_rank", "median"),
        )
        .reset_index()
    )
    paired = results.pivot_table(
        index="index", columns="condition", values="target_ctc_loss", aggfunc="mean"
    )
    success_by_sample = results.pivot_table(
        index="index", columns="condition", values="mapped_correct", aggfunc="max"
    )
    adversarial_better = float(
        (paired["adversarial"] < paired["original"]).mean()
    )
    random_better = float((paired["random"] < paired["original"]).mean())
    adversarial_wilcoxon = wilcoxon(
        paired["adversarial"], paired["original"], alternative="less"
    )
    random_wilcoxon = wilcoxon(
        paired["random"], paired["original"], alternative="two-sided"
    )
    adversarial_only = int(
        (
            (success_by_sample["adversarial"] == 1)
            & (success_by_sample["random"] == 0)
        ).sum()
    )
    random_only = int(
        (
            (success_by_sample["adversarial"] == 0)
            & (success_by_sample["random"] == 1)
        ).sum()
    )
    discordant = adversarial_only + random_only
    mcnemar_p = (
        binomtest(adversarial_only, discordant, p=0.5).pvalue
        if discordant
        else 1.0
    )

    lines = [
        "# Model-Space Mechanism and Random-Noise Control",
        "",
        "Random perturbations are matched to each adversarial perturbation's "
        "measured L-infinity and RMS norms.",
        "",
        "| Condition | N | Mean target CTC loss | Median target CTC loss | "
        "Mapped accuracy | Median target rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.n} | {row.mean_target_loss:.3f} | "
            f"{row.median_target_loss:.3f} | {100 * row.mapped_accuracy:.1f}% | "
            f"{row.median_target_rank:.1f} |"
        )
    lines.extend(
        [
            "",
            f"Adversarial perturbation reduced target loss relative to the original "
            f"for {100 * adversarial_better:.1f}% of paired utterances.",
            "",
            f"Norm-matched random noise reduced target loss relative to the original "
            f"for {100 * random_better:.1f}% of paired utterances.",
            "",
            f"Paired Wilcoxon test, adversarial target loss < original: "
            f"W={adversarial_wilcoxon.statistic:.1f}, p={adversarial_wilcoxon.pvalue:.3g}.",
            "",
            f"Paired Wilcoxon test, mean random-noise target loss vs. original: "
            f"W={random_wilcoxon.statistic:.1f}, p={random_wilcoxon.pvalue:.3g}.",
            "",
            f"Saved-waveform recovery reproduced for "
            f"{int(success_by_sample['adversarial'].sum())}/{len(success_by_sample)} "
            f"historically selected examples; at least one of five random controls "
            f"succeeded for {int(success_by_sample['random'].sum())}/"
            f"{len(success_by_sample)} examples.",
            "",
            f"Exact paired success comparison: adversarial-only={adversarial_only}, "
            f"random-only={random_only}, p={mcnemar_p:.3g}.",
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
        "--model",
        default="facebook/wav2vec2-base-960h",
    )
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results/model_mechanism_random_control"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if args.max_items:
        manifest = manifest.head(args.max_items)
    lexicon = sorted(
        pd.read_csv(args.lexicon_csv)["target_word"]
        .astype(str)
        .map(normalize_text)
        .unique()
        .tolist()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device).eval()
    rng = np.random.default_rng(args.seed)
    rows = []

    for item_number, row in manifest.iterrows():
        original = read_wav(Path(row["original_audio"]))
        adversarial = read_wav(Path(row["assisted_audio"]))
        length = min(len(original), len(adversarial))
        original = original[:length]
        adversarial = adversarial[:length]
        target = normalize_text(row["target_word"])
        perturbation = adversarial - original
        perturbation_linf = float(np.max(np.abs(perturbation)))
        perturbation_rms = rms(perturbation)

        for condition, audio, random_seed in [
            ("original", original, -1),
            ("adversarial", adversarial, -1),
        ]:
            loss, prediction = score_audio(
                audio, target, processor, model, device
            )
            mapped = mapped_word(prediction, lexicon)
            rows.append(
                {
                    "index": int(row["index"]),
                    "speaker": row["speaker"],
                    "target_word": target,
                    "condition": condition,
                    "random_seed": random_seed,
                    "target_ctc_loss": loss,
                    "raw_prediction": prediction,
                    "mapped_prediction": mapped,
                    "mapped_correct": int(mapped == target),
                    "target_rank": rank_in_lexicon(prediction, target, lexicon),
                    "perturbation_linf": perturbation_linf,
                    "perturbation_rms": perturbation_rms,
                }
            )

        for random_seed in range(args.random_seeds):
            noise = norm_matched_random_noise(
                length, perturbation_linf, perturbation_rms, rng
            )
            random_audio = np.clip(original + noise, -1.0, 1.0)
            loss, prediction = score_audio(
                random_audio, target, processor, model, device
            )
            mapped = mapped_word(prediction, lexicon)
            rows.append(
                {
                    "index": int(row["index"]),
                    "speaker": row["speaker"],
                    "target_word": target,
                    "condition": "random",
                    "random_seed": random_seed,
                    "target_ctc_loss": loss,
                    "raw_prediction": prediction,
                    "mapped_prediction": mapped,
                    "mapped_correct": int(mapped == target),
                    "target_rank": rank_in_lexicon(prediction, target, lexicon),
                    "perturbation_linf": float(np.max(np.abs(noise))),
                    "perturbation_rms": rms(noise),
                }
            )
        print(f"[{item_number + 1}/{len(manifest)}] {row['speaker']} {target}")

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "model_mechanism_results.csv", index=False)
    plot_loss_changes(results, args.output_dir / "target_ctc_loss.png")
    write_summary(results, args.output_dir / "summary.md")
    print(f"Saved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
