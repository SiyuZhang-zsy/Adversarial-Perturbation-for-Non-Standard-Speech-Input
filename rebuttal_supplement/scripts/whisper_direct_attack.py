from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


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


def load_audio(path: str, sampling_rate: int = 16000) -> np.ndarray:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if source_rate != sampling_rate:
        divisor = math.gcd(source_rate, sampling_rate)
        waveform = resample_poly(
            waveform, sampling_rate // divisor, source_rate // divisor
        ).astype(np.float32)
    return np.clip(waveform, -1.0, 1.0).astype(np.float32)


def differentiable_log_mel(
    waveform: torch.Tensor,
    feature_extractor: WhisperFeatureExtractor,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    max_samples = int(feature_extractor.n_samples)
    if waveform.shape[-1] < max_samples:
        waveform = F.pad(waveform, (0, max_samples - waveform.shape[-1]))
    else:
        waveform = waveform[..., :max_samples]

    window = torch.hann_window(
        feature_extractor.n_fft,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    stft = torch.stft(
        waveform,
        n_fft=feature_extractor.n_fft,
        hop_length=feature_extractor.hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs().square()
    mel_filters = torch.as_tensor(
        feature_extractor.mel_filters,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    mel_spec = mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    max_value = log_spec.amax(dim=(-2, -1), keepdim=True)
    log_spec = torch.maximum(log_spec, max_value - 8.0)
    return ((log_spec + 4.0) / 4.0).to(model_dtype)


def feature_attention_mask(
    waveform_length: int,
    feature_extractor: WhisperFeatureExtractor,
    device: torch.device,
) -> torch.Tensor:
    sample_mask = torch.zeros(
        (1, feature_extractor.n_samples),
        dtype=torch.long,
        device=device,
    )
    sample_mask[:, : min(waveform_length, feature_extractor.n_samples)] = 1
    return sample_mask[:, :: feature_extractor.hop_length]


def verify_frontend(
    waveform: np.ndarray,
    feature_extractor: WhisperFeatureExtractor,
    device: torch.device,
    tolerance: float = 2e-5,
) -> float:
    official = feature_extractor(
        waveform,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        device=str(device),
    ).input_features.to(device)
    candidate = differentiable_log_mel(
        torch.from_numpy(waveform).to(device).unsqueeze(0),
        feature_extractor,
        torch.float32,
    )
    difference = float((official - candidate).abs().max().item())
    if difference > tolerance:
        raise RuntimeError(
            f"Differentiable Whisper frontend mismatch: {difference:.3g} "
            f"> {tolerance:.3g}"
        )
    return difference


def target_decoder_tensors(
    processor: WhisperProcessor,
    target: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = processor.tokenizer
    prefix = list(tokenizer.prefix_tokens)
    encoded = tokenizer(
        target,
        add_special_tokens=True,
        return_attention_mask=False,
    ).input_ids
    if encoded[: len(prefix)] == prefix:
        target_ids = encoded[len(prefix) :]
        if target_ids and target_ids[-1] == tokenizer.eos_token_id:
            target_ids = target_ids[:-1]
    else:
        target_ids = tokenizer(
            target,
            add_special_tokens=False,
            return_attention_mask=False,
        ).input_ids
    if not target_ids:
        raise ValueError(f"Target produced no tokens: {target!r}")

    decoder_input_ids = prefix + target_ids
    labels = (
        [-100] * (len(prefix) - 1)
        + target_ids
        + [tokenizer.eos_token_id]
    )
    return (
        torch.tensor([decoder_input_ids], device=device),
        torch.tensor([labels], device=device),
    )


@torch.inference_mode()
def transcribe(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    waveform: np.ndarray,
    device: torch.device,
    model_dtype: torch.dtype,
) -> str:
    features = differentiable_log_mel(
        torch.from_numpy(waveform).to(device).unsqueeze(0),
        processor.feature_extractor,
        model_dtype,
    )
    attention_mask = feature_attention_mask(
        len(waveform), processor.feature_extractor, device
    )
    generated = model.generate(
        input_features=features,
        attention_mask=attention_mask,
        do_sample=False,
        num_beams=1,
        max_new_tokens=16,
        language="en",
        task="transcribe",
    )
    return normalize_text(
        processor.batch_decode(generated, skip_special_tokens=True)[0]
    )


@torch.inference_mode()
def target_loss(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    waveform: np.ndarray,
    target: str,
    device: torch.device,
    model_dtype: torch.dtype,
) -> float:
    features = differentiable_log_mel(
        torch.from_numpy(waveform).to(device).unsqueeze(0),
        processor.feature_extractor,
        model_dtype,
    )
    attention_mask = feature_attention_mask(
        len(waveform), processor.feature_extractor, device
    )
    decoder_input_ids, labels = target_decoder_tensors(
        processor, target, device
    )
    return float(
        model(
            input_features=features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            use_cache=False,
        ).loss.item()
    )


def pgd_target(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    waveform: np.ndarray,
    target: str,
    epsilon: float,
    steps: int,
    device: torch.device,
    model_dtype: torch.dtype,
    seed: int,
) -> tuple[np.ndarray, float]:
    original = torch.from_numpy(waveform).to(device).unsqueeze(0)
    generator = torch.Generator(device=device).manual_seed(seed)
    adversarial = original + torch.empty_like(original).uniform_(
        -epsilon, epsilon, generator=generator
    )
    adversarial = adversarial.clamp(-1.0, 1.0)
    momentum = torch.zeros_like(adversarial)
    alpha = 1.5 * epsilon / max(steps, 1)
    decoder_input_ids, labels = target_decoder_tensors(
        processor, target, device
    )
    attention_mask = feature_attention_mask(
        len(waveform), processor.feature_extractor, device
    )

    final_loss = float("nan")
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        features = differentiable_log_mel(
            adversarial, processor.feature_extractor, model_dtype
        )
        output = model(
            input_features=features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            use_cache=False,
        )
        final_loss = float(output.loss.detach().item())
        gradient = torch.autograd.grad(output.loss, adversarial)[0]
        gradient = gradient / gradient.abs().mean().clamp_min(1e-12)
        momentum = 0.9 * momentum + gradient
        with torch.no_grad():
            adversarial = adversarial - alpha * momentum.sign()
            delta = (adversarial - original).clamp(-epsilon, epsilon)
            adversarial = (original + delta).clamp(-1.0, 1.0)

    return adversarial.detach().cpu().numpy()[0], final_loss


def signal_metrics(
    original: np.ndarray, adversarial: np.ndarray
) -> tuple[float, float]:
    perturbation = adversarial - original
    linf = float(np.max(np.abs(perturbation)))
    signal_power = float(np.mean(original.astype(np.float64) ** 2))
    noise_power = float(np.mean(perturbation.astype(np.float64) ** 2))
    snr = (
        10.0 * math.log10(signal_power / noise_power)
        if noise_power > 0
        else float("inf")
    )
    return linf, snr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument(
        "--experiment",
        default="Full lexicon",
        help="Vocabulary experiment name in the Whisper baseline CSV.",
    )
    parser.add_argument(
        "--all-confirmed-failures",
        action="store_true",
        help=(
            "Evaluate every baseline failure that remains a failure when "
            "re-decoded by the differentiable Whisper implementation."
        ),
    )
    parser.add_argument(
        "--per-speaker",
        type=int,
        default=None,
        help="Select this many confirmed failures per test speaker.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.0001, 0.0002, 0.0005, 0.001],
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--random-controls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path(
            "analysis/results/whisper_baseline/closed_vocab_predictions.csv"
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
        default=Path("analysis/results/whisper_direct_attack_pilot"),
    )
    args = parser.parse_args()
    if args.per_speaker and args.all_confirmed_failures:
        parser.error(
            "--per-speaker and --all-confirmed-failures are mutually exclusive"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    lexicon = sorted(
        pd.read_csv(args.lexicon_csv)["target_word"]
        .astype(str)
        .map(normalize_text)
        .unique()
        .tolist()
    )
    baseline = pd.read_csv(args.baseline_csv)
    failed_candidates = baseline[
        (baseline["experiment"] == args.experiment)
        & (baseline["split"] == "test")
        & (baseline["mapped_correct"] == 0)
    ]
    if failed_candidates.empty:
        raise RuntimeError(
            f"No test failures found for experiment {args.experiment!r}"
        )
    if args.per_speaker:
        candidates = pd.concat(
            [
                group.head(args.candidate_limit)
                for _, group in failed_candidates.groupby(
                    "speaker", sort=True
                )
            ],
            ignore_index=True,
        )
    elif args.all_confirmed_failures:
        candidates = failed_candidates
    else:
        candidates = failed_candidates.head(args.candidate_limit)

    first_waveform = load_audio(candidates.iloc[0]["wav_head_path"])
    frontend_difference = verify_frontend(
        first_waveform, processor.feature_extractor, device
    )
    print(f"Frontend max absolute difference: {frontend_difference:.3g}")

    def evaluate_candidate(row: object) -> dict[str, object] | None:
        waveform = load_audio(row.wav_head_path)
        prediction = transcribe(
            model, processor, waveform, device, model_dtype
        )
        mapped, rank, distance = rank_prediction(
            prediction, normalize_text(row.target_word), lexicon
        )
        if mapped == normalize_text(row.target_word):
            return None
        return {
            "experiment": args.experiment,
            "speaker": row.speaker,
            "utt_id": row.utt_id,
            "target_word": normalize_text(row.target_word),
            "wav_head_path": row.wav_head_path,
            "baseline_raw_prediction": row.raw_prediction,
            "baseline_mapped_prediction": row.mapped_prediction,
            "original_raw_prediction": prediction,
            "original_mapped_prediction": mapped,
            "original_target_rank": rank,
            "original_target_distance": distance,
        }

    selected = []
    if args.per_speaker:
        for speaker, group in candidates.groupby("speaker", sort=True):
            speaker_selected = []
            for row in group.itertuples(index=False):
                evaluated = evaluate_candidate(row)
                if evaluated is not None:
                    speaker_selected.append(evaluated)
                if len(speaker_selected) >= args.per_speaker:
                    break
            if len(speaker_selected) < args.per_speaker:
                raise RuntimeError(
                    f"Found only {len(speaker_selected)} confirmed failures "
                    f"for {speaker} among {len(group)} candidates"
                )
            selected.extend(speaker_selected)
        required_items = args.per_speaker * candidates["speaker"].nunique()
    else:
        for row in candidates.itertuples(index=False):
            evaluated = evaluate_candidate(row)
            if evaluated is not None:
                selected.append(evaluated)
            if (
                not args.all_confirmed_failures
                and len(selected) >= args.max_items
            ):
                break
        required_items = args.max_items

    if args.all_confirmed_failures and not selected:
        raise RuntimeError(
            "No baseline failures remained failures under the differentiable "
            "Whisper implementation"
        )
    if not args.all_confirmed_failures and len(selected) < required_items:
        raise RuntimeError(
            f"Found only {len(selected)} direct-model failures among "
            f"{len(candidates)} candidates"
        )
    print(
        f"Selected {len(selected)} confirmed failures from "
        f"{len(failed_candidates)} baseline failures for {args.experiment}."
    )

    rows = []
    random_rows = []
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    for item_index, item in enumerate(selected):
        original = load_audio(item["wav_head_path"])
        original_loss = target_loss(
            model,
            processor,
            original,
            item["target_word"],
            device,
            model_dtype,
        )
        for epsilon in args.epsilons:
            adversarial, final_optimization_loss = pgd_target(
                model=model,
                processor=processor,
                waveform=original,
                target=item["target_word"],
                epsilon=epsilon,
                steps=args.steps,
                device=device,
                model_dtype=model_dtype,
                seed=args.seed + item_index,
            )
            assisted_prediction = transcribe(
                model, processor, adversarial, device, model_dtype
            )
            mapped, rank, distance = rank_prediction(
                assisted_prediction, item["target_word"], lexicon
            )
            measured_loss = target_loss(
                model,
                processor,
                adversarial,
                item["target_word"],
                device,
                model_dtype,
            )
            linf, snr = signal_metrics(original, adversarial)
            audio_path = audio_dir / (
                f"{item_index:03d}_{item['speaker']}_{item['utt_id']}_"
                f"{item['target_word']}_eps{epsilon:.5f}_k{args.steps}.wav"
            )
            sf.write(audio_path, adversarial, 16000, subtype="FLOAT")
            rows.append(
                {
                    **item,
                    "model": args.model,
                    "epsilon": epsilon,
                    "steps": args.steps,
                    "original_target_loss": original_loss,
                    "final_optimization_loss": final_optimization_loss,
                    "assisted_target_loss": measured_loss,
                    "assisted_raw_prediction": assisted_prediction,
                    "assisted_mapped_prediction": mapped,
                    "assisted_target_rank": rank,
                    "assisted_target_distance": distance,
                    "mapped_correct": int(mapped == item["target_word"]),
                    "rank_improved": int(rank < item["original_target_rank"]),
                    "linf": linf,
                    "snr_db": snr,
                    "assisted_audio": str(audio_path.resolve()),
                }
            )
            random_generator = np.random.default_rng(
                args.seed + 10000 * (item_index + 1) + int(epsilon * 1e7)
            )
            targeted_delta = adversarial - original
            for random_index in range(args.random_controls):
                random_delta = random_generator.permutation(targeted_delta)
                random_signs = random_generator.choice(
                    np.array([-1.0, 1.0], dtype=np.float32),
                    size=random_delta.shape,
                )
                random_delta = (random_delta * random_signs).astype(np.float32)
                random_waveform = np.clip(
                    original + random_delta, -1.0, 1.0
                )
                random_prediction = transcribe(
                    model, processor, random_waveform, device, model_dtype
                )
                random_mapped, random_rank, random_distance = rank_prediction(
                    random_prediction, item["target_word"], lexicon
                )
                random_linf, random_snr = signal_metrics(
                    original, random_waveform
                )
                random_rows.append(
                    {
                        "speaker": item["speaker"],
                        "utt_id": item["utt_id"],
                        "target_word": item["target_word"],
                        "model": args.model,
                        "epsilon": epsilon,
                        "steps": args.steps,
                        "random_index": random_index,
                        "original_target_rank": item["original_target_rank"],
                        "random_raw_prediction": random_prediction,
                        "random_mapped_prediction": random_mapped,
                        "random_target_rank": random_rank,
                        "random_target_distance": random_distance,
                        "mapped_correct": int(
                            random_mapped == item["target_word"]
                        ),
                        "rank_improved": int(
                            random_rank < item["original_target_rank"]
                        ),
                        "linf": random_linf,
                        "snr_db": random_snr,
                    }
                )
            print(
                f"[{item_index + 1}/{len(selected)}] "
                f"{item['target_word']} eps={epsilon:g}: "
                f"{item['original_raw_prediction']!r} -> "
                f"{assisted_prediction!r}, rank "
                f"{item['original_target_rank']} -> {rank}"
            )

    results = pd.DataFrame(rows)
    random_results = pd.DataFrame(random_rows)
    results.to_csv(args.output_dir / "results.csv", index=False)
    random_results.to_csv(
        args.output_dir / "random_controls.csv", index=False
    )
    random_summary = (
        random_results.groupby(["model", "epsilon", "steps"], as_index=False)
        .agg(
            random_trials=("mapped_correct", "size"),
            random_repairs=("mapped_correct", "sum"),
            random_repair_rate=("mapped_correct", "mean"),
            random_rank_improved=("rank_improved", "sum"),
            random_median_snr_db=("snr_db", "median"),
        )
    )
    summary = (
        results.groupby(["model", "epsilon", "steps"], as_index=False)
        .agg(
            n=("mapped_correct", "size"),
            repaired=("mapped_correct", "sum"),
            repair_rate=("mapped_correct", "mean"),
            rank_improved=("rank_improved", "sum"),
            mean_original_loss=("original_target_loss", "mean"),
            mean_assisted_loss=("assisted_target_loss", "mean"),
            median_snr_db=("snr_db", "median"),
        )
        .merge(random_summary, on=["model", "epsilon", "steps"])
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
