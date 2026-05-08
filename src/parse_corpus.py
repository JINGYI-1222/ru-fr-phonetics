"""
Stage 1: parse_corpus
Reads all TextGrid files and metadata_RUFR.csv.
Outputs: data/processed/phonemes.csv
One row per phoneme token.
"""

import os
import re
import csv
import pandas as pd
from pathlib import Path
import textgrid  # we'll install this next

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/raw/ru-fr_interference/ru-fr_interference/2")
CORPUS_DIR = RAW_DIR / "wav_et_textgrids" / "FRcorp_textgrids_only"
METADATA   = RAW_DIR / "metadata_RUFR.csv"
OUT_CSV    = Path("data/processed/phonemes.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── Load metadata ─────────────────────────────────────────────────────────────
meta = pd.read_csv(METADATA, sep=";")
meta.columns = meta.columns.str.strip()
# Build a dict: speaker_id -> {l1, gender}
spk_info = {}
for _, row in meta.iterrows():
    spk_info[row["spk"].strip().upper()] = {
        "l1":    "L1" if row["L1"].strip() == "fr" else "L2",
        "gender": row["Gender"].strip()
    }

# ── Parse TextGrids ───────────────────────────────────────────────────────────
records = []

for spk_dir in sorted(CORPUS_DIR.iterdir()):
    if not spk_dir.is_dir():
        continue
    spk_id = spk_dir.name.upper()
    if spk_id not in spk_info:
        print(f"  WARNING: {spk_id} not in metadata, skipping")
        continue

    l1     = spk_info[spk_id]["l1"]
    gender = spk_info[spk_id]["gender"]

    for tg_path in sorted(spk_dir.glob("*.TextGrid")):
        # Parse sentence ID and repetition from filename
        # e.g. ab_rus_list1_FRcorp3.TextGrid
        stem = tg_path.stem  # ab_rus_list1_FRcorp3
        match = re.search(r"FRcorp(\d+)$", stem)
        if not match:
            continue
        sent_id = int(match.group(1))
        rep_idx = 1  # one recording per sentence per speaker

        wav_path = tg_path.with_suffix(".wav")

        # Load TextGrid
        try:
            tg = textgrid.TextGrid.fromFile(str(tg_path))
        except Exception as e:
            print(f"  ERROR reading {tg_path.name}: {e}")
            continue

        # Find the phoneme tier (usually named 'phones' or similar)
        phone_tier = None
        for tier in tg.tiers:
            if tier.name.lower() in ["phones", "phone", "phonemes",
                                      "phoneme", "segmentation"]:
                phone_tier = tier
                break
        if phone_tier is None:
            # fallback: take first IntervalTier
            for tier in tg.tiers:
                if hasattr(tier, 'intervals'):
                    phone_tier = tier
                    break
        if phone_tier is None:
            print(f"  WARNING: no phone tier in {tg_path.name}")
            continue

        for interval in phone_tier.intervals:
            label = interval.mark.strip()
            if label == "" or label in ["", "sp", "sil", "SIL", "SP"]:
                continue
            records.append({
                "speaker_id":  spk_id,
                "sentence_id": sent_id,
                "rep_idx":     rep_idx,
                "phoneme":     label,
                "onset":       round(interval.minTime, 6),
                "offset":      round(interval.maxTime, 6),
                "duration_ms": round((interval.maxTime - interval.minTime) * 1000, 2),
                "l1_status":   l1,
                "gender":      gender,
                "wav_path":    str(wav_path)
            })

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"\n✅ Done! {len(df)} phoneme tokens saved to {OUT_CSV}")
print(f"   Speakers : {df['speaker_id'].nunique()}")
print(f"   Sentences: {df['sentence_id'].nunique()}")
print(f"   Phonemes : {df['phoneme'].nunique()} unique labels")
print(df.head())