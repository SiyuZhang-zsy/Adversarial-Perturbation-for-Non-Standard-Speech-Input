from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, mannwhitneyu, wilcoxon
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from full_mechanism_fixed_policy import (
    BANDS,
    band_component,
    holm_adjust,
    waveform_metrics,
    wilson_interval,
)
try:
    from whisper_direct_attack_pilot import (
        load_audio,
        normalize_text,
        rank_prediction,
        target_loss,
        transcribe,
    )
except ImportError:
    from whisper_direct_attack import (
        load_audio,
        normalize_text,
        rank_prediction,
        target_loss,
        transcribe,
    )

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEYS = ["speaker", "utt_id", "target_word"]


def sample_id(row: pd.Series) -> str:
    return "||".join(str(row[key]) for key in KEYS)


def exact_paired_directional(
    primary: pd.Series,
    comparator: pd.Series,
    alternative: str,
) -> tuple[int, int, float]:
    primary_only = int(((primary == 1) & (comparator == 0)).sum())
    comparator_only = int(((primary == 0) & (comparator == 1)).sum())
    discordant = primary_only + comparator_only
    if not discordant:
        return primary_only, comparator_only, 1.0
    if alternative == "greater":
        p_value = binomtest(
            primary_only, discordant, p=0.5, alternative="greater"
        ).pvalue
    elif alternative == "less":
        p_value = binomtest(
            primary_only, discordant, p=0.5, alternative="less"
        ).pvalue
    else:
        p_value = binomtest(primary_only, discordant, p=0.5).pvalue
    return primary_only, comparator_only, float(p_value)


def build_acoustic_table(results: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    completed: set[str] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        rows = existing.to_dict("records")
        if "sample_id" in existing.columns:
            completed = set(existing["sample_id"].astype(str))
    for index, row in results.iterrows():
        current_id = sample_id(row)
        if current_id in completed:
            continue
        original = load_audio(row["wav_head_path"])
        assisted = load_audio(row["assisted_audio"])
        rows.append(
            {
                "sample_id": current_id,
                "speaker": row["speaker"],
                "utt_id": row["utt_id"],
                "target_word": row["target_word"],
                "mapped_correct": int(row["mapped_correct"]),
                **waveform_metrics(original, assisted),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"Acoustic features: {index + 1}/{len(results)}")
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(output_path, index=False)
    acoustic = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last")
    acoustic.to_csv(output_path, index=False)
    return acoustic


def load_model(
    model_name: str,
    device: torch.device,
    local_files_only: bool,
) -> tuple[WhisperProcessor, WhisperForConditionalGeneration, torch.dtype]:
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language="en",
        task="transcribe",
        local_files_only=local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model, model_dtype


def run_ablation(
    successes: pd.DataFrame,
    lexicon: list[str],
    output_path: Path,
    model_name: str,
    device: torch.device,
    local_files_only: bool,
) -> pd.DataFrame:
    if output_path.exists():
        existing = pd.read_csv(output_path)
    else:
        existing = pd.DataFrame()
    completed = (
        set(zip(existing["sample_id"], existing["condition"]))
        if len(existing)
        else set()
    )
    processor, model, model_dtype = load_model(
        model_name, device, local_files_only
    )
    rows = existing.to_dict("records")
    conditions = []
    for label, low, high in BANDS:
        conditions.append((f"retain_{label}", label, low, high, "retain"))
        conditions.append((f"remove_{label}", label, low, high, "remove"))

    for item_index, row in successes.reset_index(drop=True).iterrows():
        current_id = sample_id(row)
        missing = [
            condition
            for condition, _, _, _, _ in conditions
            if (current_id, condition) not in completed
        ]
        if not missing:
            continue
        original = load_audio(row["wav_head_path"])
        assisted = load_audio(row["assisted_audio"])
        length = min(len(original), len(assisted))
        original = original[:length]
        assisted = assisted[:length]
        delta = assisted - original
        epsilon = float(row["epsilon"])

        for condition, label, low, high, operation in conditions:
            if condition not in missing:
                continue
            component = band_component(delta, 16000, low, high)
            if operation == "retain":
                ablated_delta = component
            else:
                ablated_delta = delta - component
            ablated_delta = np.clip(ablated_delta, -epsilon, epsilon)
            waveform = np.clip(original + ablated_delta, -1.0, 1.0).astype(
                np.float32
            )
            prediction = transcribe(
                model, processor, waveform, device, model_dtype
            )
            mapped, rank, distance = rank_prediction(
                prediction, row["target_word"], lexicon
            )
            loss = target_loss(
                model,
                processor,
                waveform,
                row["target_word"],
                device,
                model_dtype,
            )
            rows.append(
                {
                    "sample_id": current_id,
                    "speaker": row["speaker"],
                    "utt_id": row["utt_id"],
                    "target_word": row["target_word"],
                    "condition": condition,
                    "band": label,
                    "operation": operation,
                    "target_loss": loss,
                    "raw_prediction": prediction,
                    "mapped_prediction": mapped,
                    "mapped_correct": int(mapped == row["target_word"]),
                    "raw_exact": int(prediction == row["target_word"]),
                    "target_rank": rank,
                    "target_distance": distance,
                    "linf": float(np.max(np.abs(waveform - original))),
                }
            )
            completed.add((current_id, condition))

        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Ablation: {item_index + 1}/{len(successes)}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize(
    all_results: pd.DataFrame,
    acoustic: pd.DataFrame,
    ablation: pd.DataFrame,
    random_controls: pd.DataFrame,
    output_dir: Path,
) -> None:
    successes = all_results[all_results["mapped_correct"] == 1].copy()
    successes["sample_id"] = successes.apply(sample_id, axis=1)
    evaluated_ids = set(ablation["sample_id"])
    successes = successes[successes["sample_id"].isin(evaluated_ids)].copy()
    success_ids = set(successes["sample_id"])
    success_acoustic = acoustic[acoustic["sample_id"].isin(success_ids)].copy()
    failed_acoustic = acoustic[~acoustic["sample_id"].isin(success_ids)].copy()

    random_controls = random_controls.copy()
    random_controls["sample_id"] = random_controls.apply(sample_id, axis=1)
    success_random = random_controls[
        random_controls["sample_id"].isin(success_ids)
    ]
    random_any = success_random.groupby("sample_id")["mapped_correct"].max()

    acoustic_metrics = [
        "snr_db",
        "mean_absolute_log_spectral_change_db",
        "energy_fraction_4-8k",
    ]
    group_rows = []
    group_p_values = []
    for metric in acoustic_metrics:
        success_values = success_acoustic[metric].dropna()
        failure_values = failed_acoustic[metric].dropna()
        test = mannwhitneyu(
            success_values, failure_values, alternative="two-sided"
        )
        group_rows.append(
            {
                "metric": metric,
                "success_n": len(success_values),
                "success_mean": float(success_values.mean()),
                "success_median": float(success_values.median()),
                "failure_n": len(failure_values),
                "failure_mean": float(failure_values.mean()),
                "failure_median": float(failure_values.median()),
                "mann_whitney_p": float(test.pvalue),
            }
        )
        group_p_values.append(float(test.pvalue))
    for row, adjusted in zip(group_rows, holm_adjust(group_p_values)):
        row["holm_p"] = adjusted
    group_comparison = pd.DataFrame(group_rows)
    group_comparison.to_csv(
        output_dir / "success_vs_failure_acoustic.csv", index=False
    )

    ablation_summary_rows = []
    for condition, group in ablation.groupby("condition", sort=False):
        repairs = int(group["mapped_correct"].sum())
        ci_low, ci_high = wilson_interval(repairs, len(group))
        ablation_summary_rows.append(
            {
                "condition": condition,
                "operation": group["operation"].iloc[0],
                "band": group["band"].iloc[0],
                "n": len(group),
                "repairs_preserved": repairs,
                "preservation_rate": repairs / len(group),
                "preservation_ci_low": ci_low,
                "preservation_ci_high": ci_high,
                "raw_exact": int(group["raw_exact"].sum()),
                "mean_target_loss": float(group["target_loss"].mean()),
                "median_target_rank": float(group["target_rank"].median()),
                "max_linf": float(group["linf"].max()),
            }
        )
    ablation_summary = pd.DataFrame(ablation_summary_rows)
    ablation_summary.to_csv(
        output_dir / "frequency_ablation_summary.csv", index=False
    )

    paired_rows = []
    paired_p_values = []
    for operation, alternative in [("retain", "greater"), ("remove", "less")]:
        primary_condition = f"{operation}_4-8k"
        primary = (
            ablation[ablation["condition"] == primary_condition]
            .set_index("sample_id")["mapped_correct"]
            .sort_index()
        )
        for label, _, _ in BANDS[:-1]:
            comparator_condition = f"{operation}_{label}"
            comparator = (
                ablation[ablation["condition"] == comparator_condition]
                .set_index("sample_id")["mapped_correct"]
                .reindex(primary.index)
            )
            primary_only, comparator_only, p_value = exact_paired_directional(
                primary, comparator, alternative
            )
            paired_rows.append(
                {
                    "operation": operation,
                    "primary": primary_condition,
                    "comparator": comparator_condition,
                    "primary_only": primary_only,
                    "comparator_only": comparator_only,
                    "alternative": alternative,
                    "paired_p": p_value,
                }
            )
            paired_p_values.append(p_value)
    for row, adjusted in zip(paired_rows, holm_adjust(paired_p_values)):
        row["holm_p"] = adjusted
    paired_tests = pd.DataFrame(paired_rows)
    paired_tests.to_csv(
        output_dir / "frequency_paired_tests.csv", index=False
    )

    original_loss = successes["original_target_loss"]
    assisted_loss = successes["assisted_target_loss"]
    loss_test = wilcoxon(
        assisted_loss, original_loss, alternative="less"
    )
    band_columns = [f"energy_fraction_{label}" for label, _, _ in BANDS]
    band_means = success_acoustic[band_columns].mean()
    summary = pd.DataFrame(
        [
            {
                "all_confirmed_failures": len(all_results),
                "successful_recoveries": len(successes),
                "exact_target_transcriptions": int(
                    (
                        successes["assisted_raw_prediction"].map(normalize_text)
                        == successes["target_word"].map(normalize_text)
                    ).sum()
                ),
                "success_random_trials": len(success_random),
                "success_random_trial_repairs": int(
                    success_random["mapped_correct"].sum()
                ),
                "success_items_any_random_repair": int(random_any.sum()),
                "mean_original_loss_successes": float(original_loss.mean()),
                "mean_assisted_loss_successes": float(assisted_loss.mean()),
                "success_loss_wilcoxon_p": float(loss_test.pvalue),
                "median_success_snr_db": float(
                    success_acoustic["snr_db"].median()
                ),
                "mean_success_4_8khz_energy": float(
                    success_acoustic["energy_fraction_4-8k"].mean()
                ),
            }
        ]
    )
    summary.to_csv(output_dir / "statistical_summary.csv", index=False)

    retain_4_8 = ablation_summary.set_index("condition").loc["retain_4-8k"]
    remove_4_8 = ablation_summary.set_index("condition").loc["remove_4-8k"]
    success_vs_failure_4_8 = group_comparison.set_index("metric").loc[
        "energy_fraction_4-8k"
    ]
    retain_2_4 = ablation_summary.set_index("condition").loc["retain_2-4k"]
    remove_2_4 = ablation_summary.set_index("condition").loc["remove_2-4k"]
    retain_4_8_values = (
        ablation[ablation["condition"] == "retain_4-8k"]
        .set_index("sample_id")["mapped_correct"]
        .sort_index()
    )
    retain_2_4_values = (
        ablation[ablation["condition"] == "retain_2-4k"]
        .set_index("sample_id")["mapped_correct"]
        .reindex(retain_4_8_values.index)
    )
    _, _, retain_4_8_vs_2_4 = exact_paired_directional(
        retain_4_8_values, retain_2_4_values, "two-sided"
    )
    remove_4_8_values = (
        ablation[ablation["condition"] == "remove_4-8k"]
        .set_index("sample_id")["mapped_correct"]
        .sort_index()
    )
    remove_2_4_values = (
        ablation[ablation["condition"] == "remove_2-4k"]
        .set_index("sample_id")["mapped_correct"]
        .reindex(remove_4_8_values.index)
    )
    _, _, remove_4_8_vs_2_4 = exact_paired_directional(
        remove_4_8_values, remove_2_4_values, "two-sided"
    )

    lines = [
        "# Whisper Success-Conditioned Mechanism Analysis",
        "",
        f"This analysis includes all {len(successes)} mapped recoveries produced "
        "by the complete fixed-policy Whisper large-v3-turbo experiment. It "
        "characterizes successful transformations; efficacy and generic-noise "
        f"specificity remain established on all {len(all_results):,} confirmed "
        "failures.",
        "",
        f"- Exact target transcription: "
        f"{int((successes['assisted_raw_prediction'].map(normalize_text) == successes['target_word'].map(normalize_text)).sum())}/{len(successes)}.",
        f"- Mean target loss among successes: {original_loss.mean():.3f} to "
        f"{assisted_loss.mean():.3f} (one-sided Wilcoxon p={loss_test.pvalue:.3g}).",
        f"- Median signal-to-perturbation ratio: "
        f"{success_acoustic['snr_db'].median():.1f} dB.",
        f"- Mean 4-8 kHz energy share: "
        f"{100 * success_acoustic['energy_fraction_4-8k'].mean():.1f}%.",
        f"- However, 4-8 kHz energy share does not differ between successful and "
        f"unsuccessful perturbations ({100 * success_vs_failure_4_8.success_mean:.1f}% "
        f"vs. {100 * success_vs_failure_4_8.failure_mean:.1f}%; "
        f"Holm p={success_vs_failure_4_8.holm_p:.3g}), so high-frequency energy "
        "alone is not sufficient for recovery.",
        f"- Among these successful items, matched random controls repair "
        f"{int(success_random['mapped_correct'].sum())}/{len(success_random)} trials "
        f"and at least one of five random controls repairs "
        f"{int(random_any.sum())}/{len(successes)} items. This contrast is "
        "success-conditioned and is descriptive; the primary random-control test "
        f"uses all {len(all_results):,} failures.",
        "",
        "## Frequency Ablation on Successful Recoveries",
        "",
        f"- Retaining only 4-8 kHz preserves "
        f"{int(retain_4_8.repairs_preserved)}/{len(successes)} recoveries "
        f"({100 * retain_4_8.preservation_rate:.1f}%).",
        f"- Removing 4-8 kHz preserves "
        f"{int(remove_4_8.repairs_preserved)}/{len(successes)} recoveries "
        f"({100 * remove_4_8.preservation_rate:.1f}%).",
        f"- Retaining only 2-4 kHz preserves "
        f"{int(retain_2_4.repairs_preserved)}/{len(successes)} "
        f"({100 * retain_2_4.preservation_rate:.1f}%); removing it preserves "
        f"{int(remove_2_4.repairs_preserved)}/{len(successes)} "
        f"({100 * remove_2_4.preservation_rate:.1f}%).",
        f"- There is no paired evidence that 4-8 kHz is more important than "
        f"2-4 kHz (two-sided retain p={retain_4_8_vs_2_4:.3g}; "
        f"remove p={remove_4_8_vs_2_4:.3g}).",
        "",
        "| Condition | Recoveries preserved | Preservation rate | "
        "Mean target loss |",
        "|---|---:|---:|---:|",
    ]
    for row in ablation_summary.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.repairs_preserved}/{row.n} | "
            f"{100 * row.preservation_rate:.1f}% | {row.mean_target_loss:.3f} |"
        )
    lines.extend(
        [
            "",
            "The perturbation is additive and length-preserving. Frequency "
            "conditions are projected to the original L-infinity bound without "
            "amplitude amplification. The ablation supports a functional role for "
            "distributed contributions across approximately 0.5-8 kHz. No single "
            "band is necessary or sufficient, and high-frequency energy share "
            "alone does not distinguish successful from unsuccessful attacks.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plot_summary(
        success_acoustic,
        failed_acoustic,
        ablation_summary,
        output_dir / "whisper_success_mechanism.png",
    )


def plot_summary(
    success_acoustic: pd.DataFrame,
    failed_acoustic: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    band_labels = [label for label, _, _ in BANDS]
    band_columns = [f"energy_fraction_{label}" for label in band_labels]
    band_values = success_acoustic[band_columns] * 100
    axes[0, 0].bar(
        band_labels,
        band_values.mean(),
        yerr=band_values.std(ddof=1) / np.sqrt(len(band_values)),
        color="#4e79a7",
        capsize=3,
    )
    axes[0, 0].set_ylabel("Energy in successful perturbations (%)")
    axes[0, 0].set_title("A  Successful perturbation spectrum", loc="left", fontweight="bold")

    axes[0, 1].boxplot(
        [
            success_acoustic["snr_db"],
            failed_acoustic["snr_db"],
        ],
        tick_labels=["Successful", "Unsuccessful"],
        showfliers=False,
    )
    axes[0, 1].set_ylabel("Signal-to-perturbation ratio (dB)")
    axes[0, 1].set_title("B  Relative perturbation magnitude", loc="left", fontweight="bold")

    retain = (
        ablation_summary[ablation_summary["operation"] == "retain"]
        .set_index("band")
        .loc[band_labels]
    )
    remove = (
        ablation_summary[ablation_summary["operation"] == "remove"]
        .set_index("band")
        .loc[band_labels]
    )
    axes[1, 0].bar(
        band_labels,
        100 * retain["preservation_rate"],
        color="#59a14f",
    )
    axes[1, 0].set_ylabel("Successful recoveries preserved (%)")
    axes[1, 0].set_title("C  Retain only one band", loc="left", fontweight="bold")

    axes[1, 1].bar(
        band_labels,
        100 * remove["preservation_rate"],
        color="#f28e2b",
    )
    axes[1, 1].set_ylabel("Successful recoveries preserved (%)")
    axes[1, 1].set_title("D  Remove one band", loc="left", fontweight="bold")

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(
        f"Whisper mechanism analysis on all {len(success_acoustic)} successful recoveries",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(
            "analysis/results/whisper_direct_attack_fulllex_all_failures"
        ),
    )
    parser.add_argument(
        "--lexicon-csv",
        type=Path,
        default=Path("datasets/torgo_single_word_headmic_split.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results/whisper_success_mechanism"),
    )
    parser.add_argument(
        "--model", default="openai/whisper-large-v3-turbo"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = pd.read_csv(args.result_dir / "results.csv")
    all_results["target_word"] = all_results["target_word"].map(normalize_text)
    all_results["sample_id"] = all_results.apply(sample_id, axis=1)
    successes = all_results[all_results["mapped_correct"] == 1].copy()
    if args.max_items is not None:
        successes = successes.head(args.max_items).copy()
    print(f"Successful Whisper recoveries: {len(successes)}")

    acoustic = build_acoustic_table(
        all_results,
        args.output_dir / "acoustic_all.csv",
    )
    if "sample_id" not in acoustic.columns:
        if len(acoustic) != len(all_results):
            raise ValueError(
                "Existing acoustic table has no sample_id and does not align "
                "with the complete Whisper result table."
            )
        acoustic.insert(
            0,
            "sample_id",
            all_results["sample_id"].to_numpy(),
        )
        acoustic.to_csv(
            args.output_dir / "acoustic_all.csv", index=False
        )
    if args.summarize_existing:
        ablation = pd.read_csv(args.output_dir / "ablation_results.csv")
    else:
        lexicon = sorted(
            pd.read_csv(args.lexicon_csv)["target_word"]
            .astype(str)
            .map(normalize_text)
            .unique()
            .tolist()
        )
        ablation = run_ablation(
            successes,
            lexicon,
            args.output_dir / "ablation_results.csv",
            args.model,
            torch.device(args.device),
            args.local_files_only,
        )

    random_controls = pd.read_csv(args.result_dir / "random_controls.csv")
    summarize(
        all_results,
        acoustic,
        ablation,
        random_controls,
        args.output_dir,
    )
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
