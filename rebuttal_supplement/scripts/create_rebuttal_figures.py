from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "figures"

BLUE = "#3567A8"
LIGHT_BLUE = "#DCE8F5"
ORANGE = "#D97706"
LIGHT_ORANGE = "#FCE7C7"
GREEN = "#2E7D5B"
LIGHT_GREEN = "#DCEFE7"
RED = "#B94A48"
GRAY = "#5F6368"
LIGHT_GRAY = "#F1F3F4"


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 10.5,
    linewidth: float = 1.6,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#202124",
        linespacing=1.25,
    )
    return box


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = GRAY,
    connectionstyle: str = "arc3",
    linestyle: str = "-",
    linewidth: float = 1.7,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def draw_workflow(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "AdaptaVoice: intent-provided, per-utterance test-time recovery",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    add_box(
        ax,
        (0.025, 0.57),
        0.15,
        0.22,
        "User utterance\n$x$",
        LIGHT_BLUE,
        BLUE,
        fontsize=12,
    )
    add_box(
        ax,
        (0.025, 0.18),
        0.15,
        0.22,
        "Supplied intent\n$y$\n(user/context)",
        LIGHT_GREEN,
        GREEN,
        fontsize=11,
    )
    add_box(
        ax,
        (0.245, 0.39),
        0.23,
        0.34,
        "AdaptaVoice\n\nPGD computes $\\delta_i$\nfor this $(x,y)$ pair\nat test time",
        LIGHT_ORANGE,
        ORANGE,
        fontsize=11.5,
        linewidth=2,
    )
    add_box(
        ax,
        (0.245, 0.08),
        0.23,
        0.18,
        "Frozen configuration order\n$(\\epsilon_1,K_1),\\ldots,(\\epsilon_{14},K_{14})$",
        LIGHT_GRAY,
        GRAY,
        fontsize=10.5,
    )
    add_box(
        ax,
        (0.55, 0.46),
        0.17,
        0.20,
        "Transformed input\n$x+\\delta_i$",
        LIGHT_BLUE,
        BLUE,
        fontsize=12,
    )
    add_box(
        ax,
        (0.75, 0.46),
        0.18,
        0.20,
        "Fixed downstream\nrecognizer $f$\n(no weight updates)",
        LIGHT_GRAY,
        GRAY,
        fontsize=11,
    )
    add_box(
        ax,
        (0.75, 0.12),
        0.18,
        0.19,
        "Does $f(x+\\delta_i)=y$?",
        LIGHT_GREEN,
        GREEN,
        fontsize=11.5,
    )

    add_arrow(ax, (0.175, 0.68), (0.245, 0.60), BLUE)
    add_arrow(ax, (0.175, 0.29), (0.245, 0.47), GREEN)
    add_arrow(ax, (0.36, 0.26), (0.36, 0.39), GRAY)
    add_arrow(ax, (0.475, 0.56), (0.55, 0.56), ORANGE)
    add_box(
        ax,
        (0.945, 0.155),
        0.048,
        0.12,
        "Yes:\nstop",
        LIGHT_GREEN,
        GREEN,
        fontsize=8.8,
        linewidth=1.3,
    )
    add_arrow(ax, (0.72, 0.56), (0.75, 0.56), BLUE)
    add_arrow(ax, (0.84, 0.46), (0.84, 0.31), GRAY)
    add_arrow(ax, (0.93, 0.215), (0.945, 0.215), GREEN)

    ax.text(0.67, 0.17, "No: try next predefined configuration", ha="center", fontsize=10.5, color=RED)
    add_arrow(
        ax,
        (0.75, 0.17),
        (0.475, 0.17),
        RED,
        connectionstyle="arc3,rad=-0.2",
    )

    ax.text(
        0.025,
        0.015,
        "Offline evaluation: the dataset label substitutes for user-confirmed intent. "
        "No reusable conversion model, recognizer weights, or user representation is trained.",
        ha="left",
        va="bottom",
        fontsize=9.6,
        color=GRAY,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_mechanism_summary(output_path: Path) -> None:
    acoustic = pd.read_csv(
        RESULTS / "mechanism" / "paired_acoustic_features.csv"
    )
    model = pd.read_csv(
        RESULTS
        / "mechanism"
        / "model_mechanism_results.csv"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))

    band_labels = ["0-0.5", "0.5-2", "2-4", "4-8"]
    band_columns = [
        "perturbation_energy_fraction_0-0.5 kHz",
        "perturbation_energy_fraction_0.5-2 kHz",
        "perturbation_energy_fraction_2-4 kHz",
        "perturbation_energy_fraction_4-8 kHz",
    ]
    band_values = acoustic[band_columns] * 100
    band_means = band_values.mean()
    band_se = band_values.std(ddof=1) / np.sqrt(len(band_values))
    axes[0, 0].bar(
        band_labels,
        band_means,
        yerr=band_se,
        color=["#4169A1", "#628DB8", "#86AEC9", "#ACCBDD"],
        capsize=3,
    )
    axes[0, 0].set_ylabel("Perturbation energy (%)")
    axes[0, 0].set_xlabel("Frequency band (kHz)")
    axes[0, 0].set_title("A  Spectral distribution", loc="left", fontweight="bold")
    axes[0, 0].grid(axis="y", alpha=0.22)

    median_snr = acoustic["snr_db"].median()
    axes[0, 1].hist(
        acoustic["snr_db"],
        bins=11,
        color=BLUE,
        edgecolor="white",
        alpha=0.9,
    )
    axes[0, 1].axvline(
        median_snr,
        color=ORANGE,
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_snr:.1f} dB",
    )
    axes[0, 1].set_xlabel("Signal-to-perturbation ratio (dB)")
    axes[0, 1].set_ylabel("Utterances")
    axes[0, 1].set_title("B  Perturbation magnitude", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[0, 1].grid(axis="y", alpha=0.22)

    original = (
        model[model["condition"] == "original"]
        .set_index("index")["target_ctc_loss"]
        .sort_index()
    )
    adversarial = (
        model[model["condition"] == "adversarial"]
        .set_index("index")["target_ctc_loss"]
        .sort_index()
    )
    random_mean = (
        model[model["condition"] == "random"]
        .groupby("index")["target_ctc_loss"]
        .mean()
        .sort_index()
    )
    common = original.index.intersection(adversarial.index).intersection(random_mean.index)
    delta_random = random_mean.loc[common] - original.loc[common]
    delta_targeted = adversarial.loc[common] - original.loc[common]
    box = axes[1, 0].boxplot(
        [delta_random, delta_targeted],
        tick_labels=["Matched random", "Targeted PGD"],
        showfliers=True,
        patch_artist=True,
        widths=0.55,
    )
    box["boxes"][0].set_facecolor(LIGHT_GRAY)
    box["boxes"][1].set_facecolor(LIGHT_BLUE)
    for median in box["medians"]:
        median.set_color(ORANGE)
        median.set_linewidth(2)
    axes[1, 0].axhline(0, color=GRAY, linestyle="--", linewidth=1.2)
    axes[1, 0].set_ylabel("$\\Delta$ target CTC loss vs. original")
    axes[1, 0].set_title("C  Model-directed loss change", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.22)

    targeted_success = int(
        model[model["condition"] == "adversarial"]
        .groupby("index")["mapped_correct"]
        .max()
        .sum()
    )
    targeted_n = model["index"].nunique()
    random_rows = model[model["condition"] == "random"]
    random_trials = len(random_rows)
    random_trial_success = int(random_rows["mapped_correct"].sum())
    random_any_success = int(
        random_rows.groupby("index")["mapped_correct"].max().sum()
    )
    rates = [
        100 * targeted_success / targeted_n,
        100 * random_any_success / targeted_n,
    ]
    bars = axes[1, 1].bar(
        ["Targeted PGD", "Matched random"],
        rates,
        color=[BLUE, "#B9BDC3"],
        width=0.58,
    )
    labels = [
        f"{targeted_success}/{targeted_n}",
        f"{random_any_success}/{targeted_n}",
    ]
    for bar, label in zip(bars, labels):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() + 3, 2),
            label,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_ylabel("Utterances recovered (%)")
    axes[1, 1].set_title("D  Saved-waveform recovery", loc="left", fontweight="bold")
    axes[1, 1].text(
        0.5,
        0.91,
        f"Exact paired $p=9.09\\times10^{{-13}}$; "
        f"{random_trial_success}/{random_trials} random trials",
        transform=axes[1, 1].transAxes,
        ha="center",
        fontsize=9.5,
        color=GRAY,
    )
    axes[1, 1].grid(axis="y", alpha=0.22)

    fig.suptitle(
        "Acoustic and model-space characterization (50 paired utterances)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    draw_workflow(OUTPUT / "adaptavoice_workflow.png")
    draw_mechanism_summary(OUTPUT / "mechanism_summary.png")
    print(f"Saved rebuttal figures to {OUTPUT}")


if __name__ == "__main__":
    main()
