from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KEYS = ["speaker", "utt_id", "target_word"]


def paired_exact(
    targeted: pd.DataFrame, random_control: pd.DataFrame
) -> tuple[int, int, float]:
    paired = targeted[KEYS + ["mapped_correct"]].merge(
        random_control[KEYS + ["mapped_correct"]],
        on=KEYS,
        suffixes=("_targeted", "_random"),
    )
    targeted_only = int(
        (
            (paired["mapped_correct_targeted"] == 1)
            & (paired["mapped_correct_random"] == 0)
        ).sum()
    )
    random_only = int(
        (
            (paired["mapped_correct_targeted"] == 0)
            & (paired["mapped_correct_random"] == 1)
        ).sum()
    )
    discordant = targeted_only + random_only
    p_value = (
        float(binomtest(targeted_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return targeted_only, random_only, p_value


def plot_results(
    speaker_summary: pd.DataFrame,
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    x = np.arange(len(speaker_summary))
    axes[0].bar(
        x,
        100 * speaker_summary["targeted_repair_rate"],
        color="#3567A8",
        label="Targeted PGD",
    )
    axes[0].scatter(
        x,
        100 * speaker_summary["random_repair_rate"],
        color="#C44E52",
        marker="x",
        s=45,
        label="Matched random",
        zorder=3,
    )
    axes[0].set_xticks(x, speaker_summary["speaker"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Failed-input repair rate (%)")
    axes[0].set_xlabel("Held-out speaker")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    for _, row in results.iterrows():
        axes[1].plot(
            [0, 1],
            [row["original_target_loss"], row["assisted_target_loss"]],
            color="#777777",
            alpha=0.25,
            linewidth=0.8,
        )
    axes[1].scatter(
        np.zeros(len(results)),
        results["original_target_loss"],
        color="#777777",
        s=14,
        alpha=0.65,
    )
    axes[1].scatter(
        np.ones(len(results)),
        results["assisted_target_loss"],
        color="#3567A8",
        s=14,
        alpha=0.65,
    )
    axes[1].set_xticks([0, 1], ["Original", "Targeted"])
    axes[1].set_ylabel("Whisper target-token loss")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(
            "analysis/results/whisper_direct_attack_large_balanced60_masked"
        ),
    )
    args = parser.parse_args()

    results = pd.read_csv(args.result_dir / "results.csv")
    random_controls = pd.read_csv(args.result_dir / "random_controls.csv")

    exact_rows = []
    for random_index, group in random_controls.groupby("random_index"):
        targeted_only, random_only, p_value = paired_exact(results, group)
        exact_rows.append(
            {
                "random_index": random_index,
                "targeted_only": targeted_only,
                "random_only": random_only,
                "paired_exact_p": p_value,
            }
        )
    exact = pd.DataFrame(exact_rows)
    exact.to_csv(args.result_dir / "paired_random_tests.csv", index=False)

    speaker_targeted = (
        results.groupby("speaker", as_index=False)
        .agg(
            n=("mapped_correct", "size"),
            targeted_repairs=("mapped_correct", "sum"),
            targeted_repair_rate=("mapped_correct", "mean"),
            rank_improved=("rank_improved", "sum"),
            median_snr_db=("snr_db", "median"),
        )
    )
    speaker_random = (
        random_controls.groupby("speaker", as_index=False)
        .agg(
            random_trials=("mapped_correct", "size"),
            random_repairs=("mapped_correct", "sum"),
            random_repair_rate=("mapped_correct", "mean"),
        )
    )
    speaker_summary = speaker_targeted.merge(speaker_random, on="speaker")
    speaker_summary.to_csv(
        args.result_dir / "speaker_summary.csv", index=False
    )

    loss_test = wilcoxon(
        results["assisted_target_loss"],
        results["original_target_loss"],
        alternative="less",
    )
    rank_test = wilcoxon(
        results["assisted_target_rank"],
        results["original_target_rank"],
        alternative="less",
    )
    targeted_successes = int(results["mapped_correct"].sum())
    targeted_ci = binomtest(
        targeted_successes, len(results)
    ).proportion_ci(method="wilson")
    raw_exact = int(
        (
            results["assisted_raw_prediction"].str.lower()
            == results["target_word"].str.lower()
        ).sum()
    )
    statistical_summary = pd.DataFrame(
        [
            {
                "n": len(results),
                "speakers": results["speaker"].nunique(),
                "epsilon": results["epsilon"].iloc[0],
                "steps": results["steps"].iloc[0],
                "mapped_repairs": targeted_successes,
                "mapped_repair_rate": results["mapped_correct"].mean(),
                "mapped_repair_ci_low": targeted_ci.low,
                "mapped_repair_ci_high": targeted_ci.high,
                "raw_exact_targets": raw_exact,
                "rank_improved": int(results["rank_improved"].sum()),
                "median_original_rank": results[
                    "original_target_rank"
                ].median(),
                "median_assisted_rank": results[
                    "assisted_target_rank"
                ].median(),
                "mean_original_loss": results[
                    "original_target_loss"
                ].mean(),
                "mean_assisted_loss": results[
                    "assisted_target_loss"
                ].mean(),
                "loss_wilcoxon_p": float(loss_test.pvalue),
                "rank_wilcoxon_p": float(rank_test.pvalue),
                "median_snr_db": results["snr_db"].median(),
                "random_trials": len(random_controls),
                "random_repairs": int(
                    random_controls["mapped_correct"].sum()
                ),
                "max_paired_random_p": exact["paired_exact_p"].max(),
            }
        ]
    )
    statistical_summary.to_csv(
        args.result_dir / "statistical_summary.csv", index=False
    )

    summary_row = statistical_summary.iloc[0]
    lines = [
        "# Direct Whisper Large-v3-turbo Targeted-Recovery Experiment",
        "",
        "The evaluation used 60 confirmed full-lexicon failures, balanced at "
        "10 utterances for each of six held-out speakers. For each speaker, "
        "we took the first 10 items in dataset order that remained failures "
        "when re-decoded by the differentiable implementation. The fixed "
        "policy was epsilon=0.0002 and K=3, matching the first configuration "
        "in the previously frozen Wav2Vec2 policy order. No Whisper-specific "
        "parameter search was used.",
        "",
        f"- Mapped repairs: {targeted_successes}/{len(results)} "
        f"({100 * summary_row.mapped_repair_rate:.1f}%; "
        f"95% Wilson CI {100 * summary_row.mapped_repair_ci_low:.1f}-"
        f"{100 * summary_row.mapped_repair_ci_high:.1f}%).",
        f"- Exact target transcriptions: {raw_exact}/{len(results)}.",
        f"- Matched random controls: "
        f"{int(summary_row.random_repairs)}/{int(summary_row.random_trials)}.",
        f"- Target rank improved for {int(summary_row.rank_improved)}/"
        f"{len(results)} items; median rank changed from "
        f"{summary_row.median_original_rank:.1f} to "
        f"{summary_row.median_assisted_rank:.1f} "
        f"(one-sided Wilcoxon p={summary_row.rank_wilcoxon_p:.3g}).",
        f"- Mean target-token loss changed from "
        f"{summary_row.mean_original_loss:.3f} to "
        f"{summary_row.mean_assisted_loss:.3f} "
        f"(one-sided Wilcoxon p={summary_row.loss_wilcoxon_p:.3g}).",
        f"- Median signal-to-perturbation ratio: "
        f"{summary_row.median_snr_db:.1f} dB.",
        f"- Across five paired random seeds, the largest exact paired p-value "
        f"was {summary_row.max_paired_random_p:.3g}.",
        "",
        "| Speaker | Targeted repairs | Targeted rate | Random repairs / trials |",
        "|---|---:|---:|---:|",
    ]
    for row in speaker_summary.itertuples(index=False):
        lines.append(
            f"| {row.speaker} | {row.targeted_repairs}/{row.n} | "
            f"{100 * row.targeted_repair_rate:.1f}% | "
            f"{row.random_repairs}/{row.random_trials} |"
        )
    lines.extend(
        [
            "",
            "The large between-speaker range should be reported rather than "
            "collapsed into the aggregate alone. These results establish "
            "direct recoverability for a stronger recognizer; they do not "
            "imply cross-model transfer.",
        ]
    )
    (args.result_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    plot_results(
        speaker_summary,
        results,
        args.result_dir / "direct_whisper_results.png",
    )
    print(statistical_summary.to_string(index=False))
    print(speaker_summary.to_string(index=False))


if __name__ == "__main__":
    main()
