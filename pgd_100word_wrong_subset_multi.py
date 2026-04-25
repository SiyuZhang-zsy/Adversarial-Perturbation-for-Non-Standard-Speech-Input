from pathlib import Path
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# =========================================================
# Config
# =========================================================
CSV_PATH = Path(r"D:\audio_adv_experiment\datasets\torgo_100word_split_clean.csv")
#USE_SPLIT = "dev"   # 先跑 dev，之后改成 test
USE_SPLIT = None
OUT_PATH = Path(fr"D:\audio_adv_experiment\datasets\torgo_100word_wrong_subset_multi_{USE_SPLIT}_results.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/wav2vec2-base-960h"
DEBUG_HEAD = None

PARAMS = [
    (0.00008, 2),
    (0.00008, 3),
    (0.00010, 2),
    (0.00010, 3),
    (0.00012, 2),
    (0.00012, 3),
    (0.00014, 2),
    (0.00014, 3),
    (0.00016, 2),
    (0.00016, 3),
    (0.00018, 2),
    (0.00018, 3),
    (0.00020, 2),
    (0.00020, 3),
]

print("Using device:", DEVICE)

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(CSV_PATH)
df["target_word"] = df["target_word"].astype(str).str.strip().str.lower()
#df = df[df["split"] == USE_SPLIT].copy()
if USE_SPLIT is not None:
    df = df[df["split"] == USE_SPLIT].copy()

if DEBUG_HEAD is not None:
    df = df.head(DEBUG_HEAD).copy()

lexicon = sorted(df["target_word"].dropna().unique().tolist())

print("Rows before wrong-subset filter:", len(df))
print("Speakers:", sorted(df["speaker"].astype(str).unique().tolist()))
print("Lexicon size:", len(lexicon))
print("Lexicon preview:", lexicon[:20])

# =========================================================
# Load model
# =========================================================
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# =========================================================
# Helpers
# =========================================================
def normalize_text(s: str) -> str:
    return str(s).strip().lower()

def load_audio(path: str, target_sr: int = 16000):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return waveform.squeeze(0), sr

def normalize_waveform_for_wav2vec2(waveform_1d: torch.Tensor) -> torch.Tensor:
    mean = waveform_1d.mean()
    var = waveform_1d.var(unbiased=False)
    waveform_1d = (waveform_1d - mean) / torch.sqrt(var + 1e-7)
    return waveform_1d

def decode_waveform(waveform_1d: torch.Tensor) -> str:
    speech = waveform_1d.detach().cpu().numpy()
    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    pred_text = processor.batch_decode(pred_ids)[0]
    return normalize_text(pred_text)

def compute_ctc_loss(waveform_1d: torch.Tensor, target_text: str) -> torch.Tensor:
    x = normalize_waveform_for_wav2vec2(waveform_1d)
    input_values = x.unsqueeze(0)

    labels = processor.tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(DEVICE)

    outputs = model(input_values, labels=labels)
    return outputs.loss

def pgd_attack(waveform_1d: torch.Tensor, target_text: str, epsilon: float, steps: int):
    original = waveform_1d.detach().clone()
    adv = original.clone().detach()
    alpha = epsilon / steps

    for _ in range(steps):
        adv.requires_grad_(True)

        loss = compute_ctc_loss(adv, target_text)

        model.zero_grad()
        if adv.grad is not None:
            adv.grad.zero_()

        loss.backward()

        with torch.no_grad():
            grad_sign = adv.grad.sign()
            adv = adv - alpha * grad_sign
            delta = torch.clamp(adv - original, min=-epsilon, max=epsilon)
            adv = torch.clamp(original + delta, min=-1.0, max=1.0)

        adv = adv.detach()

    return adv

def levenshtein(a: str, b: str) -> int:
    a = normalize_text(a)
    b = normalize_text(b)

    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[len(a)][len(b)]

def nearest_words(pred: str, lexicon_words: list[str], topk: int = 10):
    scored = []
    for w in lexicon_words:
        d = levenshtein(pred, w)
        scored.append((w, d))
    scored.sort(key=lambda x: (x[1], x[0]))
    return scored[:topk]

def map_prediction_to_lexicon(raw_pred: str, lexicon_words: list[str]):
    ranked = nearest_words(raw_pred, lexicon_words, topk=len(lexicon_words))
    ranked_words = [w for w, _ in ranked]
    ranked_dists = [d for _, d in ranked]
    return ranked_words, ranked_dists

# =========================================================
# Baseline pass
# =========================================================
baseline_rows = []

for idx, row in df.iterrows():
    wav_path = row["wav_head_path"]
    target = normalize_text(row["target_word"])

    try:
        speech, sr = load_audio(wav_path)
        speech = speech.to(DEVICE)

        raw_pred = decode_waveform(speech)
        ranked_words, ranked_dists = map_prediction_to_lexicon(raw_pred, lexicon)

        mapped_pred = ranked_words[0]
        mapped_correct = int(mapped_pred == target)
        target_rank = ranked_words.index(target) + 1
        target_distance = ranked_dists[target_rank - 1]

        baseline_rows.append({
            "speaker": row["speaker"],
            "split": row["split"],
            "utt_id": row["utt_id"],
            "target_word": target,
            "wav_head_path": wav_path,
            "raw_prediction": raw_pred,
            "mapped_prediction": mapped_pred,
            "mapped_correct": mapped_correct,
            "target_rank": target_rank,
            "target_distance": target_distance,
        })
    except Exception as e:
        baseline_rows.append({
            "speaker": row["speaker"],
            "split": row["split"],
            "utt_id": row["utt_id"],
            "target_word": target,
            "wav_head_path": wav_path,
            "raw_prediction": "__ERROR__",
            "mapped_prediction": "__ERROR__",
            "mapped_correct": 0,
            "target_rank": -1,
            "target_distance": -1,
            "error_message": str(e),
        })

baseline_df = pd.DataFrame(baseline_rows)
wrong_df = baseline_df[baseline_df["mapped_correct"] == 0].copy()

print("Baseline total:", len(baseline_df))
print("Baseline wrong subset:", len(wrong_df))

# =========================================================
# PGD only on wrong subset
# =========================================================
results = []

for idx, row in wrong_df.iterrows():
    wav_path = row["wav_head_path"]
    target = normalize_text(row["target_word"])

    try:
        speech, sr = load_audio(wav_path)
        speech = speech.to(DEVICE)
        duration_sec = float(speech.shape[0] / 16000)

        baseline_raw = row["raw_prediction"]
        baseline_mapped = row["mapped_prediction"]
        baseline_target_rank = int(row["target_rank"])
        baseline_target_distance = float(row["target_distance"])

        for eps, steps in PARAMS:
            adv_wave = pgd_attack(
                waveform_1d=speech,
                target_text=target,
                epsilon=eps,
                steps=steps
            )

            adv_raw = decode_waveform(adv_wave)
            adv_ranked_words, adv_ranked_dists = map_prediction_to_lexicon(adv_raw, lexicon)

            adv_mapped = adv_ranked_words[0]
            adv_correct = int(adv_mapped == target)
            adv_target_rank = adv_ranked_words.index(target) + 1
            adv_target_distance = adv_ranked_dists[adv_target_rank - 1]

            results.append({
                "speaker": row["speaker"],
                "split": row["split"],
                "utt_id": row["utt_id"],
                "target_word": target,
                "epsilon": eps,
                "steps": steps,
                "raw_prediction": adv_raw,
                "mapped_prediction": adv_mapped,
                "mapped_correct": adv_correct,
                "target_rank": adv_target_rank,
                "top3_hit": int(adv_target_rank <= 3),
                "top5_hit": int(adv_target_rank <= 5),
                "top10_hit": int(adv_target_rank <= 10),
                "target_distance": adv_target_distance,
                "duration_sec": duration_sec,
                "wav_head_path": wav_path,
                "baseline_raw_prediction": baseline_raw,
                "baseline_mapped_prediction": baseline_mapped,
                "baseline_target_rank": baseline_target_rank,
                "baseline_target_distance": baseline_target_distance,
            })

        print(f"[{idx}] target={target} | base_raw={baseline_raw} | base_map={baseline_mapped} | base_rank={baseline_target_rank}")

    except Exception as e:
        print(f"[{idx}] ERROR on {wav_path}: {e}")

result_df = pd.DataFrame(results)
result_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("\nSaved to:", OUT_PATH)

if len(result_df) == 0:
    raise SystemExit("No results produced.")

result_df["rank_change"] = result_df["target_rank"] - result_df["baseline_target_rank"]
result_df["distance_change"] = result_df["target_distance"] - result_df["baseline_target_distance"]
result_df["repaired"] = (result_df["mapped_correct"] == 1).astype(int)

def change_label(x):
    if x < 0:
        return "improved"
    elif x > 0:
        return "worsened"
    else:
        return "same"

result_df["rank_change_label"] = result_df["rank_change"].apply(change_label)
result_df["distance_change_label"] = result_df["distance_change"].apply(change_label)

summary = (
    result_df.groupby(["epsilon", "steps"])
             .agg(
                 subset_size=("mapped_correct", "size"),
                 repair_rate=("repaired", "mean"),
                 mapped_top1_accuracy=("mapped_correct", "mean"),
                 top3_hit_rate=("top3_hit", "mean"),
                 top5_hit_rate=("top5_hit", "mean"),
                 mean_target_rank=("target_rank", "mean"),
                 mean_baseline_target_rank=("baseline_target_rank", "mean"),
                 mean_rank_change=("rank_change", "mean"),
                 mean_target_distance=("target_distance", "mean"),
                 mean_baseline_target_distance=("baseline_target_distance", "mean"),
                 mean_distance_change=("distance_change", "mean"),
             )
             .reset_index()
             .sort_values(["epsilon", "steps"])
)

rank_change_summary = (
    result_df.groupby(["epsilon", "steps", "rank_change_label"])
             .size()
             .reset_index(name="count")
             .sort_values(["epsilon", "steps", "rank_change_label"])
)

distance_change_summary = (
    result_df.groupby(["epsilon", "steps", "distance_change_label"])
             .size()
             .reset_index(name="count")
             .sort_values(["epsilon", "steps", "distance_change_label"])
)

print("\n=== 100-word wrong-subset multi summary ===")
print(summary)

print("\n=== 100-word rank change summary ===")
print(rank_change_summary)

print("\n=== 100-word distance change summary ===")
print(distance_change_summary)

# =========================================================
# Oracle / dynamic upper-bound analysis
# =========================================================
group_keys = ["speaker", "utt_id", "target_word", "split"]

oracle_df = (
    result_df.groupby(group_keys)
             .agg(
                 any_repaired=("mapped_correct", "max"),
                 best_rank=("target_rank", "min"),
                 baseline_rank=("baseline_target_rank", "first"),
                 best_distance=("target_distance", "min"),
                 baseline_distance=("baseline_target_distance", "first"),
             )
             .reset_index()
)

oracle_df["rank_improved"] = (oracle_df["best_rank"] < oracle_df["baseline_rank"]).astype(int)
oracle_df["distance_improved"] = (oracle_df["best_distance"] < oracle_df["baseline_distance"]).astype(int)

n_wrong = len(oracle_df)
n_repaired = int(oracle_df["any_repaired"].sum())

oracle_repair_rate = n_repaired / n_wrong if n_wrong > 0 else 0.0

baseline_correct_count = int(baseline_df["mapped_correct"].sum())
baseline_total_count = len(baseline_df)

oracle_total_correct = baseline_correct_count + n_repaired
oracle_overall_accuracy = oracle_total_correct / baseline_total_count if baseline_total_count > 0 else 0.0

print("\n=== Oracle / dynamic upper-bound summary ===")
print(f"Baseline total samples: {baseline_total_count}")
print(f"Baseline correct samples: {baseline_correct_count}")
print(f"Baseline wrong samples: {n_wrong}")
print(f"Wrong-subset repaired by >=1 parameter: {n_repaired}")
print(f"Oracle repair rate on wrong-subset: {oracle_repair_rate:.4f}")
print(f"Baseline overall mapped accuracy: {baseline_correct_count / baseline_total_count:.4f}")
print(f"Oracle overall mapped accuracy upper bound: {oracle_overall_accuracy:.4f}")

print("\n=== Oracle improvement counts ===")
print(f"Rank improved for >=1 parameter choice: {int(oracle_df['rank_improved'].sum())} / {n_wrong}")
print(f"Distance improved for >=1 parameter choice: {int(oracle_df['distance_improved'].sum())} / {n_wrong}")

oracle_df.to_csv(OUT_PATH.with_name(OUT_PATH.stem + "_oracle_summary.csv"), index=False, encoding="utf-8-sig")