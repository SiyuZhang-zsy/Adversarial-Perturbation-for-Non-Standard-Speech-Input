from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly, stft, welch
from scipy.stats import binomtest, wilcoxon
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAMPLE_KEYS = ["speaker", "split", "utt_id", "target_word"]
BANDS = [
    ("0-0.5k", 0.0, 500.0),
    ("0.5-2k", 500.0, 2000.0),
    ("2-4k", 2000.0, 4000.0),
    ("4-8k", 4000.0, 8000.1),
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


def map_prediction(
    prediction: str, target: str, lexicon: list[str]
) -> tuple[str, int]:
    ranked = sorted(
        lexicon, key=lambda word: (levenshtein(prediction, word), word)
    )
    return ranked[0], ranked.index(target) + 1


def read_wav(path: str | Path, target_sample_rate: int = 16000) -> np.ndarray:
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


def band_component(
    perturbation: np.ndarray, sample_rate: int, low: float, high: float
) -> np.ndarray:
    spectrum = np.fft.rfft(perturbation)
    frequencies = np.fft.rfftfreq(len(perturbation), d=1.0 / sample_rate)
    mask = (frequencies >= low) & (frequencies < high)
    filtered = np.fft.irfft(spectrum * mask, n=len(perturbation))
    return filtered.astype(np.float32)


def waveform_metrics(
    original: np.ndarray, assisted: np.ndarray, sample_rate: int = 16000
) -> dict[str, float]:
    length = min(len(original), len(assisted))
    original = original[:length].astype(np.float64)
    assisted = assisted[:length].astype(np.float64)
    perturbation = assisted - original
    original_rms = rms(original)
    perturbation_rms = rms(perturbation)
    frequencies, perturbation_psd = welch(
        perturbation,
        fs=sample_rate,
        nperseg=min(1024, length),
        scaling="density",
    )
    total_energy = float(np.trapz(perturbation_psd, frequencies))
    centroid_denominator = float(perturbation_psd.sum())
    centroid = (
        float((frequencies * perturbation_psd).sum() / centroid_denominator)
        if centroid_denominator > 0
        else 0.0
    )
    cumulative = np.cumsum(perturbation_psd)
    if cumulative.size and cumulative[-1] > 0:
        rolloff_index = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
        rolloff = float(frequencies[min(rolloff_index, len(frequencies) - 1)])
    else:
        rolloff = 0.0

    _, _, original_stft = stft(
        original, fs=sample_rate, nperseg=400, noverlap=240, boundary=None
    )
    _, _, assisted_stft = stft(
        assisted, fs=sample_rate, nperseg=400, noverlap=240, boundary=None
    )
    frames = min(original_stft.shape[1], assisted_stft.shape[1])
    log_difference = np.abs(
        20 * np.log10(np.abs(assisted_stft[:, :frames]) + 1e-8)
        - 20 * np.log10(np.abs(original_stft[:, :frames]) + 1e-8)
    )

    metrics = {
        "duration_sec": length / sample_rate,
        "original_rms": original_rms,
        "perturbation_linf": float(np.max(np.abs(perturbation))),
        "perturbation_rms": perturbation_rms,
        "snr_db": (
            float(20 * np.log10(original_rms / perturbation_rms))
            if perturbation_rms > 0
            else float("inf")
        ),
        "perturbation_spectral_centroid_hz": centroid,
        "perturbation_spectral_rolloff85_hz": rolloff,
        "mean_absolute_log_spectral_change_db": float(log_difference.mean()),
        "p95_absolute_log_spectral_change_db": float(
            np.percentile(log_difference, 95)
        ),
    }
    for label, low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        energy = (
            float(np.trapz(perturbation_psd[mask], frequencies[mask]))
            if mask.any()
            else 0.0
        )
        metrics[f"energy_fraction_{label}"] = (
            energy / total_energy if total_energy > 0 else float("nan")
        )
    return metrics


def normalize_waveform_for_wav2vec2(waveform: torch.Tensor) -> torch.Tensor:
    mean = waveform.mean()
    variance = waveform.var(unbiased=False)
    return (waveform - mean) / torch.sqrt(variance + 1e-7)


def target_labels(
    target: str,
    processor: Wav2Vec2Processor,
    device: torch.device,
) -> torch.Tensor:
    return processor.tokenizer(
        target,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)


def pgd_attack(
    original_audio: np.ndarray,
    target: str,
    epsilon: float,
    steps: int,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
) -> np.ndarray:
    original = torch.from_numpy(original_audio).to(device)
    adversarial = original.detach().clone()
    labels = target_labels(target, processor, device)
    alpha = epsilon / steps

    for _ in range(steps):
        adversarial.requires_grad_(True)
        normalized = normalize_waveform_for_wav2vec2(adversarial)
        loss = model(input_values=normalized.unsqueeze(0), labels=labels).loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            adversarial = adversarial - alpha * adversarial.grad.sign()
            delta = torch.clamp(adversarial - original, -epsilon, epsilon)
            adversarial = torch.clamp(original + delta, -1.0, 1.0).detach()
    return adversarial.cpu().numpy().astype(np.float32)


@torch.no_grad()
def score_batch(
    audio_batch: list[np.ndarray],
    targets: list[str],
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
) -> list[tuple[float, str]]:
    inputs = processor(
        audio_batch,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        input_lengths = attention_mask.sum(-1)
    else:
        input_lengths = torch.tensor(
            [len(audio) for audio in audio_batch], device=device
        )

    logits = model(
        input_values=input_values,
        attention_mask=attention_mask,
    ).logits
    output_lengths = model._get_feat_extract_output_lengths(input_lengths).long()
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

    label_rows = [
        processor.tokenizer(
            target,
            add_special_tokens=False,
        ).input_ids
        for target in targets
    ]
    target_lengths = torch.tensor(
        [len(labels) for labels in label_rows], device=device, dtype=torch.long
    )
    flat_targets = torch.tensor(
        [token for labels in label_rows for token in labels],
        device=device,
        dtype=torch.long,
    )
    losses = F.ctc_loss(
        log_probs,
        flat_targets,
        output_lengths,
        target_lengths,
        blank=model.config.pad_token_id,
        reduction="none",
        zero_infinity=model.config.ctc_zero_infinity,
    )

    prediction_ids = logits.argmax(dim=-1)
    predictions = []
    for index, output_length in enumerate(output_lengths.tolist()):
        decoded = processor.decode(prediction_ids[index, :output_length])
        predictions.append(normalize_text(decoded))
    return list(zip(losses.detach().cpu().tolist(), predictions))


def condition_record(
    sample: pd.Series,
    condition: str,
    random_seed: int | None,
    original: np.ndarray,
    audio: np.ndarray,
    score: tuple[float, str],
    lexicon: list[str],
) -> dict[str, object]:
    loss, prediction = score
    mapped, target_rank = map_prediction(
        prediction, sample["target_word"], lexicon
    )
    delta = audio - original
    return {
        "sample_id": sample["sample_id"],
        "speaker": sample["speaker"],
        "utt_id": sample["utt_id"],
        "target_word": sample["target_word"],
        "condition": condition,
        "random_seed": random_seed,
        "target_ctc_loss": loss,
        "raw_prediction": prediction,
        "mapped_prediction": mapped,
        "mapped_correct": int(mapped == sample["target_word"]),
        "target_rank": target_rank,
        "perturbation_linf": float(np.max(np.abs(delta))),
        "perturbation_rms": rms(delta),
        "snr_db": (
            float(20 * np.log10(rms(original) / rms(delta)))
            if rms(delta) > 0
            else float("inf")
        ),
    }


def exact_paired_pvalue(
    targeted: pd.Series, comparator: pd.Series
) -> tuple[int, int, float]:
    targeted_only = int(((targeted == 1) & (comparator == 0)).sum())
    comparator_only = int(((targeted == 0) & (comparator == 1)).sum())
    discordant = targeted_only + comparator_only
    p_value = (
        binomtest(targeted_only, discordant, p=0.5).pvalue
        if discordant
        else 1.0
    )
    return targeted_only, comparator_only, p_value


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min((count - rank) * p_values[index], 1.0)
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def safe_wilcoxon(
    left: pd.Series, right: pd.Series, alternative: str
) -> tuple[int, float, float]:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    left_values = left_values[finite]
    right_values = right_values[finite]
    difference = left_values - right_values
    if not len(difference):
        return 0, float("nan"), float("nan")
    if np.allclose(difference, 0):
        return len(difference), 0.0, 1.0
    result = wilcoxon(left_values, right_values, alternative=alternative)
    return len(difference), float(result.statistic), float(result.pvalue)


def summarize(
    results: pd.DataFrame,
    acoustic: pd.DataFrame,
    historical_failures: int,
    output_dir: Path,
) -> None:
    results = results.copy()
    results["raw_exact"] = (
        results["raw_prediction"].astype(str).map(normalize_text)
        == results["target_word"].astype(str).map(normalize_text)
    )
    original = (
        results[results["condition"] == "original"]
        .set_index("sample_id")
        .sort_index()
    )
    targeted = (
        results[results["condition"] == "targeted"]
        .set_index("sample_id")
        .sort_index()
    )
    common = original.index.intersection(targeted.index)
    original = original.loc[common]
    targeted = targeted.loc[common]
    baseline_confirmation_failures = len(common)
    common = original.index[original["mapped_correct"] == 0]
    original = original.loc[common]
    targeted = targeted.loc[common]
    analysis_results = results[results["sample_id"].isin(common)].copy()
    acoustic = acoustic[acoustic["sample_id"].isin(common)].copy()
    analysis_results.to_csv(
        output_dir / "analysis_condition_results.csv", index=False
    )
    acoustic.to_csv(
        output_dir / "analysis_acoustic_features.csv", index=False
    )

    random_rows = analysis_results[
        analysis_results["condition"] == "random"
    ].copy()
    random_any = (
        random_rows.groupby("sample_id")["mapped_correct"].max().reindex(common)
    )
    random_mean_loss = (
        random_rows.groupby("sample_id")["target_ctc_loss"].mean().reindex(common)
    )
    targeted_only, random_only, random_p = exact_paired_pvalue(
        targeted["mapped_correct"], random_any
    )
    target_loss_n, target_loss_w, target_loss_p = safe_wilcoxon(
        targeted["target_ctc_loss"],
        original["target_ctc_loss"],
        alternative="less",
    )
    random_loss_n, random_loss_w, random_loss_p = safe_wilcoxon(
        random_mean_loss,
        original["target_ctc_loss"],
        alternative="two-sided",
    )
    rank_n, rank_w, rank_p = safe_wilcoxon(
        targeted["target_rank"], original["target_rank"], alternative="less"
    )
    targeted_repairs = int(targeted["mapped_correct"].sum())
    targeted_ci_low, targeted_ci_high = wilson_interval(
        targeted_repairs, len(targeted)
    )
    finite_target_loss = np.isfinite(original["target_ctc_loss"]) & np.isfinite(
        targeted["target_ctc_loss"]
    )
    finite_random_loss = np.isfinite(original["target_ctc_loss"]) & np.isfinite(
        random_mean_loss
    )

    random_seed_rows = []
    for random_seed, group in random_rows.groupby("random_seed"):
        seed_success = (
            group.set_index("sample_id")["mapped_correct"].reindex(common)
        )
        seed_targeted_only, seed_random_only, seed_p = exact_paired_pvalue(
            targeted["mapped_correct"], seed_success
        )
        random_seed_rows.append(
            {
                "random_seed": int(random_seed),
                "n": len(seed_success),
                "random_repairs": int(seed_success.sum()),
                "targeted_repairs": int(targeted["mapped_correct"].sum()),
                "targeted_only": seed_targeted_only,
                "random_only": seed_random_only,
                "paired_p": seed_p,
            }
        )
    random_seed_tests = pd.DataFrame(random_seed_rows)
    random_seed_tests.to_csv(
        output_dir / "random_seed_paired_tests.csv", index=False
    )

    ablation_rows = []
    ablation_p_values = []
    for condition in [
        *(f"retain_{label}" for label, _, _ in BANDS),
        *(f"remove_{label}" for label, _, _ in BANDS),
    ]:
        comparator = (
            analysis_results[analysis_results["condition"] == condition]
            .set_index("sample_id")
            .reindex(common)
        )
        targeted_only_band, band_only, paired_p = exact_paired_pvalue(
            targeted["mapped_correct"], comparator["mapped_correct"]
        )
        loss_n, _, loss_p = safe_wilcoxon(
            comparator["target_ctc_loss"],
            original["target_ctc_loss"],
            alternative="less",
        )
        finite_comparator = np.isfinite(comparator["target_ctc_loss"])
        targeted_success_ids = targeted.index[targeted["mapped_correct"] == 1]
        retained_successes = int(
            comparator.loc[targeted_success_ids, "mapped_correct"].sum()
        )
        ablation_rows.append(
            {
                "condition": condition,
                "n": len(comparator),
                "repairs": int(comparator["mapped_correct"].sum()),
                "repair_rate": float(comparator["mapped_correct"].mean()),
                "repairs_retained_from_full_targeted": retained_successes,
                "full_targeted_repairs": int(targeted["mapped_correct"].sum()),
                "finite_loss_n": int(finite_comparator.sum()),
                "mean_target_loss": float(
                    comparator.loc[
                        finite_comparator, "target_ctc_loss"
                    ].mean()
                ),
                "median_target_rank": float(comparator["target_rank"].median()),
                "targeted_only": targeted_only_band,
                "ablation_only": band_only,
                "paired_success_p": paired_p,
                "loss_comparison_n": loss_n,
                "loss_vs_original_p": loss_p,
            }
        )
        ablation_p_values.append(paired_p)
    adjusted = holm_adjust(ablation_p_values)
    for row, adjusted_p in zip(ablation_rows, adjusted):
        row["paired_success_holm_p"] = adjusted_p
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(output_dir / "frequency_ablation_summary.csv", index=False)

    aggregate_input = analysis_results.copy()
    aggregate_input.loc[
        ~np.isfinite(aggregate_input["target_ctc_loss"]), "target_ctc_loss"
    ] = np.nan
    aggregate = (
        aggregate_input.groupby("condition")
        .agg(
            n=("target_ctc_loss", "size"),
            finite_loss_n=("target_ctc_loss", "count"),
            mean_target_loss=("target_ctc_loss", "mean"),
            median_target_loss=("target_ctc_loss", "median"),
            mapped_accuracy=("mapped_correct", "mean"),
            median_target_rank=("target_rank", "median"),
            median_snr_db=("snr_db", "median"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "condition_summary.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "historical_failed_inputs": historical_failures,
                "baseline_confirmation_failures": baseline_confirmation_failures,
                "current_stack_confirmed_failures": len(common),
                "length_aware_reclassified_correct": (
                    baseline_confirmation_failures - len(common)
                ),
                "historical_failures_now_correct": historical_failures
                - len(common),
                "epsilon": 0.0002,
                "steps": 3,
                "targeted_repairs": targeted_repairs,
                "targeted_repair_rate": float(targeted["mapped_correct"].mean()),
                "targeted_repair_ci_low": targeted_ci_low,
                "targeted_repair_ci_high": targeted_ci_high,
                "targeted_raw_exact": int(targeted["raw_exact"].sum()),
                "random_trials": len(random_rows),
                "random_trial_repairs": int(random_rows["mapped_correct"].sum()),
                "random_trial_raw_exact": int(random_rows["raw_exact"].sum()),
                "random_items_any_repair": int(random_any.sum()),
                "targeted_only": targeted_only,
                "random_only": random_only,
                "paired_success_p": random_p,
                "finite_target_loss_pairs": target_loss_n,
                "mean_original_loss": float(
                    original.loc[finite_target_loss, "target_ctc_loss"].mean()
                ),
                "mean_targeted_loss": float(
                    targeted.loc[finite_target_loss, "target_ctc_loss"].mean()
                ),
                "finite_random_loss_pairs": random_loss_n,
                "mean_random_loss": float(
                    random_mean_loss.loc[finite_random_loss].mean()
                ),
                "target_loss_wilcoxon_w": target_loss_w,
                "target_loss_wilcoxon_p": target_loss_p,
                "random_loss_wilcoxon_w": random_loss_w,
                "random_loss_wilcoxon_p": random_loss_p,
                "median_original_rank": float(original["target_rank"].median()),
                "median_targeted_rank": float(targeted["target_rank"].median()),
                "rank_comparison_n": rank_n,
                "rank_wilcoxon_w": rank_w,
                "rank_wilcoxon_p": rank_p,
                "rank_improved": int(
                    (targeted["target_rank"] < original["target_rank"]).sum()
                ),
                "median_snr_db": float(acoustic["snr_db"].median()),
            }
        ]
    )
    summary.to_csv(output_dir / "statistical_summary.csv", index=False)

    speaker = (
        targeted.reset_index()
        .groupby("speaker")
        .agg(
            n=("mapped_correct", "size"),
            targeted_repairs=("mapped_correct", "sum"),
            targeted_repair_rate=("mapped_correct", "mean"),
            median_target_rank=("target_rank", "median"),
        )
        .reset_index()
    )
    speaker.to_csv(output_dir / "speaker_summary.csv", index=False)

    band_columns = [f"energy_fraction_{label}" for label, _, _ in BANDS]
    band_means = acoustic[band_columns].mean()
    lines = [
        "# Complete Fixed-Policy Mechanism Analysis",
        "",
        f"The historical held-out failure set contains {historical_failures:,} "
        f"utterances. The prior batch-decoding confirmation identifies "
        f"{baseline_confirmation_failures:,} failures. Length-aware decoding in "
        f"this analysis reclassifies "
        f"{baseline_confirmation_failures - len(common):,} padding-sensitive "
        f"items as already correct, leaving {len(common):,} confirmed failures "
        f"for all recovery comparisons. In total, "
        f"{historical_failures - len(common):,} historical failures do not trigger "
        f"recovery in this runtime.",
        "",
        "Every evaluated utterance uses the first dev-frozen policy configuration "
        "(`epsilon=0.0002`, `K=3`). There is no per-item configuration selection "
        "and no success-based sampling.",
        "",
        f"- Targeted repairs: {targeted_repairs}/{len(common)} "
        f"({100 * targeted['mapped_correct'].mean():.1f}%; 95% Wilson CI "
        f"{100 * targeted_ci_low:.1f}-{100 * targeted_ci_high:.1f}%); "
        f"{int(targeted['raw_exact'].sum())} raw outputs exactly equal the target.",
        f"- Five L-infinity/RMS-matched random controls: "
        f"{int(random_rows['mapped_correct'].sum())}/{len(random_rows)} trials; "
        f"at least one random repair for {int(random_any.sum())}/{len(common)} "
        f"utterances.",
        f"- Targeted versus any-random paired comparison: targeted-only="
        f"{targeted_only}, random-only={random_only}, p={random_p:.3g}.",
        f"- Across the five separately paired random seeds, the largest exact "
        f"paired p-value is {random_seed_tests['paired_p'].max():.3g}.",
        f"- Mean target CTC loss on {target_loss_n:,} finite paired items: "
        f"original {original.loc[finite_target_loss, 'target_ctc_loss'].mean():.3f}, "
        f"targeted {targeted.loc[finite_target_loss, 'target_ctc_loss'].mean():.3f}, "
        f"matched random {random_mean_loss.loc[finite_random_loss].mean():.3f}.",
        f"- {len(common) - target_loss_n} short utterances have undefined/infinite "
        f"CTC loss because the target "
        f"token sequence exceeds the available output frames; they remain included "
        f"in recognition and acoustic analyses but are excluded from loss summaries.",
        f"- Targeted loss < original: one-sided Wilcoxon p="
        f"{target_loss_p:.3g}; random versus original: two-sided p="
        f"{random_loss_p:.3g}.",
        f"- Post-hoc edit-distance target rank does not improve overall: "
        f"{int((targeted['target_rank'] < original['target_rank']).sum())}/"
        f"{len(common)} improve, while the median changes from "
        f"{original['target_rank'].median():.1f} to "
        f"{targeted['target_rank'].median():.1f} (p={rank_p:.3g}). This rank is "
        f"not the PGD objective and shows that lower CTC loss need not produce "
        f"monotonic rank movement under a three-step fixed policy.",
        f"- Median signal-to-perturbation ratio: "
        f"{acoustic['snr_db'].median():.1f} dB.",
        "",
        "## Perturbation Energy",
        "",
    ]
    for label, _, _ in BANDS:
        lines.append(
            f"- {label}: {100 * band_means[f'energy_fraction_{label}']:.1f}%."
        )
    lines.extend(
        [
            "",
            "## Frequency Ablation",
            "",
            "| Condition | Repairs | Full-targeted repairs retained | "
            "Mean target loss | Paired success p (Holm) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in ablation.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.repairs}/{row.n} | "
            f"{row.repairs_retained_from_full_targeted}/"
            f"{row.full_targeted_repairs} | {row.mean_target_loss:.3f} | "
            f"{row.paired_success_holm_p:.3g} |"
        )
    lines.extend(
        [
            "",
            "Band-retained conditions keep only the named portion of the original "
            "targeted perturbation. Band-removed conditions subtract that portion. "
            "Each resulting perturbation is projected back to the same L-infinity "
            "bound without amplitude amplification. Thus the ablation tests the "
            "contribution of each band under the original budget; it does not claim "
            "that spectral energy alone explains recovery.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plot_summary(
        analysis_results,
        acoustic,
        ablation,
        output_dir / "mechanism_full.png",
    )


def plot_summary(
    results: pd.DataFrame,
    acoustic: pd.DataFrame,
    ablation: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    band_labels = [label for label, _, _ in BANDS]
    band_columns = [f"energy_fraction_{label}" for label in band_labels]
    band_values = acoustic[band_columns] * 100
    axes[0, 0].bar(
        band_labels,
        band_values.mean(),
        yerr=band_values.std(ddof=1) / np.sqrt(len(band_values)),
        color="#4e79a7",
        capsize=3,
    )
    axes[0, 0].set_ylabel("Perturbation energy (%)")
    axes[0, 0].set_title("A  Frequency distribution", loc="left", fontweight="bold")

    loss = results.pivot_table(
        index="sample_id",
        columns="condition",
        values="target_ctc_loss",
        aggfunc="mean",
    )
    random_delta = (loss["random"] - loss["original"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    targeted_delta = (loss["targeted"] - loss["original"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    axes[0, 1].boxplot(
        [random_delta, targeted_delta],
        tick_labels=["Matched random", "Targeted PGD"],
        showfliers=False,
    )
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_ylabel("Change in target CTC loss")
    axes[0, 1].set_title("B  Model-directed effect", loc="left", fontweight="bold")

    random_any = (
        results[results["condition"] == "random"]
        .groupby("sample_id")["mapped_correct"]
        .max()
    )
    targeted = (
        results[results["condition"] == "targeted"]
        .set_index("sample_id")["mapped_correct"]
    )
    axes[1, 0].bar(
        ["Targeted PGD", "Any of 5 random"],
        [100 * targeted.mean(), 100 * random_any.mean()],
        color=["#3b6fb6", "#c44e52"],
    )
    axes[1, 0].set_ylabel("Confirmed-failure repair rate (%)")
    axes[1, 0].set_title("C  Recovery control", loc="left", fontweight="bold")

    retain = ablation[ablation["condition"].str.startswith("retain_")].copy()
    remove = ablation[ablation["condition"].str.startswith("remove_")].copy()
    retain["band"] = retain["condition"].str.replace("retain_", "", regex=False)
    remove["band"] = remove["condition"].str.replace("remove_", "", regex=False)
    x = np.arange(len(band_labels))
    width = 0.36
    axes[1, 1].bar(
        x - width / 2,
        100 * retain.set_index("band").loc[band_labels, "repair_rate"],
        width,
        label="Retain only",
        color="#59a14f",
    )
    axes[1, 1].bar(
        x + width / 2,
        100 * remove.set_index("band").loc[band_labels, "repair_rate"],
        width,
        label="Remove band",
        color="#f28e2b",
    )
    axes[1, 1].set_xticks(x, band_labels)
    axes[1, 1].set_ylabel("Repair rate (%)")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].set_title("D  Frequency ablation", loc="left", fontweight="bold")

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle(
        f"Complete fixed-policy mechanism analysis ({len(acoustic):,} utterances)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-csv",
        type=Path,
        default=Path(
            "datasets/torgo_fulllex_wrong_subset_multi_None_results.csv"
        ),
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path(
            "analysis/results/full_test_random_control/"
            "current_baseline_predictions.csv"
        ),
    )
    parser.add_argument(
        "--lexicon-csv",
        type=Path,
        default=Path("datasets/torgo_single_word_headmic_split.csv"),
    )
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access instead of using the local cache.",
    )
    parser.add_argument("--epsilon", type=float, default=0.0002)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Regenerate summaries and figures from existing result CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/results/full_mechanism_fixed_policy"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.result_csv)
    raw = raw[raw["split"] == "test"].copy()
    raw["sample_id"] = raw[SAMPLE_KEYS].astype(str).agg("||".join, axis=1)
    samples = (
        raw.sort_values(["sample_id", "epsilon", "steps"])
        .groupby("sample_id", as_index=False)
        .first()
    )
    samples["target_word"] = samples["target_word"].map(normalize_text)
    historical_failures = len(samples)

    baseline = pd.read_csv(args.baseline_csv)
    confirmed_failure_ids = set(
        baseline.loc[baseline["mapped_correct"] == 0, "sample_id"]
    )
    samples = samples[samples["sample_id"].isin(confirmed_failure_ids)].copy()
    samples = samples.sort_values("sample_id").reset_index(drop=True)
    if args.max_items is not None:
        samples = samples.head(args.max_items).copy()
    print(
        f"Evaluating {len(samples)}/{historical_failures} current-stack "
        f"confirmed failures"
    )

    if args.summarize_existing:
        summarize(
            pd.read_csv(args.output_dir / "condition_results.csv"),
            pd.read_csv(args.output_dir / "acoustic_features.csv"),
            historical_failures=historical_failures,
            output_dir=args.output_dir,
        )
        print((args.output_dir / "summary.md").read_text(encoding="utf-8"))
        return

    lexicon = sorted(
        pd.read_csv(args.lexicon_csv)["target_word"]
        .astype(str)
        .map(normalize_text)
        .unique()
        .tolist()
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor = Wav2Vec2Processor.from_pretrained(
        args.model, local_files_only=not args.allow_model_download
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model, local_files_only=not args.allow_model_download
    ).to(device).eval()

    all_results: list[dict[str, object]] = []
    acoustic_rows: list[dict[str, object]] = []
    records = [row._asdict() for row in samples.itertuples(index=False)]

    for batch_start in range(0, len(records), args.batch_size):
        batch_records = records[batch_start : batch_start + args.batch_size]
        originals = [read_wav(row["wav_head_path"]) for row in batch_records]
        targets = [row["target_word"] for row in batch_records]
        targeted_audio = [
            pgd_attack(
                original,
                target,
                args.epsilon,
                args.steps,
                processor,
                model,
                device,
            )
            for original, target in zip(originals, targets)
        ]
        targeted_deltas = [
            targeted - original
            for original, targeted in zip(originals, targeted_audio)
        ]

        condition_audio: dict[str, list[np.ndarray]] = {
            "original": originals,
            "targeted": targeted_audio,
        }
        random_audio_by_seed: dict[int, list[np.ndarray]] = {}
        for random_seed in range(args.random_seeds):
            generated = []
            for sample_index, (original, delta) in enumerate(
                zip(originals, targeted_deltas)
            ):
                global_index = batch_start + sample_index
                rng = np.random.default_rng(
                    args.seed + global_index * 1000 + random_seed
                )
                noise = norm_matched_random_noise(
                    len(delta),
                    float(np.max(np.abs(delta))),
                    rms(delta),
                    rng,
                )
                generated.append(np.clip(original + noise, -1.0, 1.0))
            random_audio_by_seed[random_seed] = generated

        for label, low, high in BANDS:
            retained = [
                band_component(delta, 16000, low, high)
                for delta in targeted_deltas
            ]
            condition_audio[f"retain_{label}"] = [
                np.clip(
                    original
                    + np.clip(component, -args.epsilon, args.epsilon),
                    -1.0,
                    1.0,
                )
                for original, component in zip(originals, retained)
            ]
            condition_audio[f"remove_{label}"] = [
                np.clip(
                    original
                    + np.clip(
                        delta - component, -args.epsilon, args.epsilon
                    ),
                    -1.0,
                    1.0,
                )
                for original, delta, component in zip(
                    originals, targeted_deltas, retained
                )
            ]

        for condition, audio_batch in condition_audio.items():
            scores = score_batch(
                audio_batch, targets, processor, model, device
            )
            for row, original, audio, score in zip(
                batch_records, originals, audio_batch, scores
            ):
                all_results.append(
                    condition_record(
                        row,
                        condition,
                        None,
                        original,
                        audio,
                        score,
                        lexicon,
                    )
                )

        for random_seed, audio_batch in random_audio_by_seed.items():
            scores = score_batch(
                audio_batch, targets, processor, model, device
            )
            for row, original, audio, score in zip(
                batch_records, originals, audio_batch, scores
            ):
                all_results.append(
                    condition_record(
                        row,
                        "random",
                        random_seed,
                        original,
                        audio,
                        score,
                        lexicon,
                    )
                )

        for row, original, targeted in zip(
            batch_records, originals, targeted_audio
        ):
            metrics = waveform_metrics(original, targeted)
            acoustic_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "speaker": row["speaker"],
                    "utt_id": row["utt_id"],
                    "target_word": row["target_word"],
                    "epsilon": args.epsilon,
                    "steps": args.steps,
                    **metrics,
                }
            )

        completed = min(batch_start + len(batch_records), len(records))
        pd.DataFrame(all_results).to_csv(
            args.output_dir / "condition_results.csv", index=False
        )
        pd.DataFrame(acoustic_rows).to_csv(
            args.output_dir / "acoustic_features.csv", index=False
        )
        print(f"Completed {completed}/{len(records)}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    acoustic_df = pd.DataFrame(acoustic_rows)
    summarize(
        results_df,
        acoustic_df,
        historical_failures=historical_failures,
        output_dir=args.output_dir,
    )
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
