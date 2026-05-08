"""
Stage 3: extract_neural_whisper
Extracts hidden-state representations from Whisper encoder.
Reads parameters from params.yaml.
Output: data/processed/features_whisper.npz

NOTE: This script requires a GPU. It was executed on Kaggle (NVIDIA Tesla T4).
      To reproduce locally, a CUDA-capable GPU is required.
      The output file features_whisper.npz is already provided in data/processed/.
"""

import re
import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
import textgrid
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel

# ── Load parameters ───────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

CORPUS_DIR  = Path(params["corpus"]["corpus_dir"])
META_PATH   = Path(params["corpus"]["metadata"])
MODEL_NAME  = params["whisper"]["model"]
LAYER_LOW   = params["whisper"]["layer_low"]
LAYER_HIGH  = params["whisper"]["layer_high"]
SAMPLE_RATE = params["whisper"]["sample_rate"]
HOP_LENGTH  = params["whisper"]["hop_length"]

OUT_PATH = Path("data/processed/features_whisper.npz")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Load metadata ─────────────────────────────────────────────────────────────
meta = pd.read_csv(META_PATH, sep=";")
meta.columns = meta.columns.str.strip()
spk_info = {}
for _, row in meta.iterrows():
    spk_info[row["spk"].strip().upper()] = {
        "l1":    "L1" if row["L1"].strip() == "fr" else "L2",
        "gender": row["Gender"].strip()
    }

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
model = model.to(DEVICE).eval()
print("Model loaded!")

# ── Collect phoneme tokens ────────────────────────────────────────────────────
records = []
for spk_dir in sorted(CORPUS_DIR.iterdir()):
    if not spk_dir.is_dir():
        continue
    spk_id = spk_dir.name.upper()
    if spk_id not in spk_info:
        continue
    l1     = spk_info[spk_id]["l1"]
    gender = spk_info[spk_id]["gender"]
    for tg_path in sorted(spk_dir.glob("*.TextGrid")):
        match = re.search(r"FRcorp(\d+)$", tg_path.stem)
        if not match:
            continue
        sent_id  = int(match.group(1))
        wav_path = tg_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        try:
            tg = textgrid.TextGrid.fromFile(str(tg_path))
        except Exception:
            continue
        phone_tier = None
        for tier in tg.tiers:
            if tier.name.lower() in ["phones","phone","phonemes","phoneme","segmentation"]:
                phone_tier = tier
                break
        if phone_tier is None:
            for tier in tg.tiers:
                if hasattr(tier, "intervals"):
                    phone_tier = tier
                    break
        if phone_tier is None:
            continue
        for interval in phone_tier.intervals:
            label = interval.mark.strip()
            if label in ["", "sp", "sil", "SIL", "SP"]:
                continue
            records.append({
                "speaker_id":  spk_id,
                "sentence_id": sent_id,
                "phoneme":     label,
                "onset":       interval.minTime,
                "offset":      interval.maxTime,
                "l1_status":   l1,
                "gender":      gender,
                "wav_path":    str(wav_path)
            })

print(f"Total tokens: {len(records)}")

# ── Extract features ──────────────────────────────────────────────────────────
vectors_low  = []
vectors_high = []
current_wav  = None
current_hidden = None

for row in tqdm(records, desc="Extracting Whisper features"):
    wav_path = row["wav_path"]
    if wav_path != current_wav:
        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(dim=0)
        inputs = processor(waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(DEVICE)
        with torch.no_grad():
            encoder_out = model.encoder(input_features, output_hidden_states=True)
        current_hidden = [h.squeeze(0).cpu().float().numpy()
                          for h in encoder_out.hidden_states]
        current_wav = wav_path

    onset_frame  = int(row["onset"]  * SAMPLE_RATE / HOP_LENGTH)
    offset_frame = int(row["offset"] * SAMPLE_RATE / HOP_LENGTH)
    n_frames     = current_hidden[0].shape[0]
    onset_frame  = min(onset_frame,  n_frames - 1)
    offset_frame = min(max(offset_frame, onset_frame + 1), n_frames)

    vectors_low.append(current_hidden[LAYER_LOW][onset_frame:offset_frame].mean(axis=0))
    vectors_high.append(current_hidden[LAYER_HIGH][onset_frame:offset_frame].mean(axis=0))

# ── Save ──────────────────────────────────────────────────────────────────────
vectors_low  = np.array(vectors_low,  dtype=np.float32)
vectors_high = np.array(vectors_high, dtype=np.float32)

np.savez_compressed(
    OUT_PATH,
    vectors_low=vectors_low,
    vectors_high=vectors_high,
    meta=np.array([list(r.values()) for r in records], dtype=object),
    meta_keys=np.array(list(records[0].keys()))
)

print(f"\n✅ Done! Saved to {OUT_PATH}")
print(f"   vectors_low  shape: {vectors_low.shape}")
print(f"   vectors_high shape: {vectors_high.shape}")