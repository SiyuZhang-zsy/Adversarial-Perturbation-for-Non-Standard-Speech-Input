from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft, welch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BANDS = [
    ("0-0.5 kHz", 0, 500),
    ("0.5-2 kHz", 500, 2000),
    ("2-4 kHz", 2000, 4000),
    ("4-8 kHz", 4000, 8000),
]


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim == 2:
        audio = audio.astype(np.float64).mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    return int(sample_rate), audio


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def band_energy_fractions(
    frequencies: np.ndarray, psd: np.ndarray
) -> dict[str, float]:
    total = float(np.trapz(psd, frequencies))
    result = {}
    for label, low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        energy = float(np.trapz(psd[mask], frequencies[mask])) if mask.any() else 0.0
        result[label] = energy / total if total > 0 else float("nan")
    return result


def spectral_centroid(frequencies: np.ndarray, psd: np.ndarray) -> float:
    denominator = float(psd.sum())
    return float((frequencies * psd).sum() / denominator) if denominator > 0 else 0.0


def spectral_rolloff(
    frequencies: np.ndarray, psd: np.ndarray, fraction: float = 0.85
) -> float:
    cumulative = np.cumsum(psd)
    if cumulative.size == 0 or cumulative[-1] <= 0:
        return 0.0
    index = int(np.searchsorted(cumulative, fraction * cumulative[-1]))
    return float(frequencies[min(index, len(frequencies) - 1)])


def log_spectral_change(
    original: np.ndarray, assisted: np.ndarray, sample_rate: int
) -> tuple[float, float]:
    _, _, original_stft = stft(
        original, fs=sample_rate, nperseg=400, noverlap=240, boundary=None
    )
    _, _, assisted_stft = stft(
        assisted, fs=sample_rate, nperseg=400, noverlap=240, boundary=None
    )
    frames = min(original_stft.shape[1], assisted_stft.shape[1])
    original_log = 20 * np.log10(np.abs(original_stft[:, :frames]) + 1e-8)
    assisted_log = 20 * np.log10(np.abs(assisted_stft[:, :frames]) + 1e-8)
    difference = assisted_log - original_log
    return float(np.mean(np.abs(difference))), float(np.percentile(np.abs(difference), 95))


def analyze_pair(row: pd.Series) -> tuple[dict[str, float | str], pd.DataFrame]:
    original_path = Path(row["original_audio"])
    assisted_path = Path(row["assisted_audio"])
    sample_rate, original = read_wav(original_path)
    assisted_rate, assisted = read_wav(assisted_path)
    if assisted_rate != sample_rate:
        raise ValueError(f"Sample-rate mismatch: {original_path} and {assisted_path}")
    length = min(len(original), len(assisted))
    original = original[:length]
    assisted = assisted[:length]
    perturbation = assisted - original

    original_rms = rms(original)
    perturbation_rms = rms(perturbation)
    snr_db = (
        20 * np.log10(original_rms / perturbation_rms)
        if perturbation_rms > 0
        else float("inf")
    )
    frequencies, perturbation_psd = welch(
        perturbation,
        fs=sample_rate,
        nperseg=min(1024, length),
        scaling="density",
    )
    _, original_psd = welch(
        original,
        fs=sample_rate,
        nperseg=min(1024, length),
        scaling="density",
    )
    fractions = band_energy_fractions(frequencies, perturbation_psd)
    mean_log_change, p95_log_change = log_spectral_change(
        original, assisted, sample_rate
    )

    metrics: dict[str, float | str] = {
        "index": int(row["index"]),
        "speaker": row["speaker"],
        "split": row["split"],
        "utt_id": row["utt_id"],
        "target_word": row["target_word"],
        "epsilon": float(row["best_epsilon"]),
        "steps": int(row["best_steps"]),
        "duration_sec": length / sample_rate,
        "original_rms": original_rms,
        "perturbation_linf": float(np.max(np.abs(perturbation))),
        "perturbation_rms": perturbation_rms,
        "snr_db": float(snr_db),
        "perturbation_to_signal_rms_ratio": (
            perturbation_rms / original_rms if original_rms > 0 else float("nan")
        ),
        "perturbation_spectral_centroid_hz": spectral_centroid(
            frequencies, perturbation_psd
        ),
        "perturbation_spectral_rolloff85_hz": spectral_rolloff(
            frequencies, perturbation_psd
        ),
        "mean_absolute_log_spectral_change_db": mean_log_change,
        "p95_absolute_log_spectral_change_db": p95_log_change,
    }
    for label, fraction in fractions.items():
        metrics[f"perturbation_energy_fraction_{label}"] = fraction

    spectrum = pd.DataFrame(
        {
            "index": int(row["index"]),
            "frequency_hz": frequencies,
            "perturbation_psd": perturbation_psd,
            "original_psd": original_psd,
        }
    )
    return metrics, spectrum


def plot_psd(spectra: pd.DataFrame, output_path: Path) -> None:
    normalized = spectra.copy()
    normalized["normalized_psd"] = normalized.groupby("index")[
        "perturbation_psd"
    ].transform(lambda values: values / max(values.sum(), 1e-20))
    summary = (
        normalized.groupby("frequency_hz")["normalized_psd"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary = summary[summary["frequency_hz"] <= 8000]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    mean_db = 10 * np.log10(summary["mean"].to_numpy() + 1e-20)
    lower_db = 10 * np.log10(
        np.maximum(summary["mean"] - summary["std"], 1e-20)
    )
    upper_db = 10 * np.log10(summary["mean"] + summary["std"] + 1e-20)
    ax.plot(summary["frequency_hz"], mean_db, color="#2457a6", linewidth=2)
    ax.fill_between(
        summary["frequency_hz"],
        lower_db,
        upper_db,
        color="#2457a6",
        alpha=0.18,
        linewidth=0,
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized perturbation PSD (dB)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_band_energy(metrics: pd.DataFrame, output_path: Path) -> None:
    columns = [f"perturbation_energy_fraction_{label}" for label, _, _ in BANDS]
    values = metrics[columns] * 100
    means = values.mean()
    standard_errors = values.std(ddof=1) / np.sqrt(len(values))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        [label for label, _, _ in BANDS],
        means,
        yerr=standard_errors,
        color=["#355c9a", "#4e79a7", "#76a5c7", "#a6c8df"],
        capsize=4,
    )
    ax.set_ylabel("Perturbation energy (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_magnitude(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].hist(metrics["snr_db"], bins=12, color="#4e79a7", edgecolor="white")
    axes[0].set_xlabel("Signal-to-perturbation ratio (dB)")
    axes[0].set_ylabel("Utterances")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].scatter(
        metrics["epsilon"],
        metrics["perturbation_linf"],
        color="#d95f02",
        alpha=0.8,
    )
    limit = max(metrics["epsilon"].max(), metrics["perturbation_linf"].max())
    axes[1].plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("Configured epsilon")
    axes[1].set_ylabel("Measured perturbation L-infinity")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_summary(metrics: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Acoustic Mechanism Summary",
        "",
        f"Paired audio examples selected by the historical result exporter: {len(metrics)}.",
        "",
        "These measurements characterize what changes in the waveform. They do "
        "not by themselves establish that any individual acoustic feature causes recovery. "
        "Recovery must be re-verified from the saved waveform before these pairs are "
        "described as successful.",
        "",
        "| Measure | Median | IQR |",
        "|---|---:|---:|",
    ]
    summary_metrics = [
        ("Perturbation L-infinity", "perturbation_linf"),
        ("Perturbation RMS", "perturbation_rms"),
        ("Signal-to-perturbation ratio (dB)", "snr_db"),
        (
            "Perturbation spectral centroid (Hz)",
            "perturbation_spectral_centroid_hz",
        ),
        (
            "Mean absolute log-spectral change (dB)",
            "mean_absolute_log_spectral_change_db",
        ),
    ]
    for label, column in summary_metrics:
        values = metrics[column]
        lines.append(
            f"| {label} | {values.median():.4g} | "
            f"{values.quantile(0.25):.4g}-{values.quantile(0.75):.4g} |"
        )

    lines.extend(["", "## Perturbation energy by frequency band", ""])
    lines.append("| Band | Mean energy share |")
    lines.append("|---|---:|")
    for label, _, _ in BANDS:
        column = f"perturbation_energy_fraction_{label}"
        lines.append(f"| {label} | {100 * metrics[column].mean():.1f}% |")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/audio_examples/audio_example_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/results/acoustic_mechanism"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    metrics_rows = []
    spectra = []
    for _, row in manifest.iterrows():
        metrics, spectrum = analyze_pair(row)
        metrics_rows.append(metrics)
        spectra.append(spectrum)

    metrics_df = pd.DataFrame(metrics_rows)
    spectra_df = pd.concat(spectra, ignore_index=True)
    metrics_df.to_csv(args.output_dir / "paired_acoustic_features.csv", index=False)
    plot_psd(spectra_df, args.output_dir / "perturbation_psd.png")
    plot_band_energy(metrics_df, args.output_dir / "perturbation_band_energy.png")
    plot_magnitude(metrics_df, args.output_dir / "perturbation_magnitude.png")
    write_summary(metrics_df, args.output_dir / "summary.md")

    print(metrics_df.describe(include="all").to_string())
    print(f"\nSaved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
