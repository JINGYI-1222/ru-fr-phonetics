"""
Stage 2: extract_acoustics
For each phoneme token in phonemes.csv, extract:
  - F1, F2, F3 at midpoint (all tokens)
  - F1, F2, F3 at 25% and 75% for long vowels (duration > 80ms)
  - f0 mean (voiced segments)
  - Spectral Centre of Gravity (fricatives only)
Reads parameters from params.yaml.
Output: data/processed/features_acoustic.csv
"""

import parselmouth
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# ── Load parameters ───────────────────────────────────────────────────────────
with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

MAX_FORMANT_F = params["acoustics"]["max_formant_female"]
MAX_FORMANT_M = params["acoustics"]["max_formant_male"]
N_FORMANTS    = params["acoustics"]["n_formants"]
WINDOW_LENGTH = params["acoustics"]["window_length"]
TIME_STEP     = params["acoustics"]["time_step"]

# ── French fricatives (for SCG) ───────────────────────────────────────────────
FRICATIVES = {"f", "v", "s", "z", "ʃ", "ʒ"}

# ── French oral vowels (for trajectory analysis) ──────────────────────────────
VOWELS = {"i", "e", "ɛ", "a", "ɑ", "o", "ɔ", "u", "y", "ø", "œ", "ə",
          "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"}

LONG_VOWEL_THRESHOLD_MS = 80.0

# ── Load phonemes ─────────────────────────────────────────────────────────────
PHONEMES_CSV = Path("data/processed/phonemes.csv")
OUT_CSV      = Path("data/processed/features_acoustic.csv")

df = pd.read_csv(PHONEMES_CSV)
print(f"Loaded {len(df)} phoneme tokens")

# ── Extract features ──────────────────────────────────────────────────────────
results = []
current_wav   = None
current_sound = None

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting acoustics"):
    wav_path = row["wav_path"]
    gender   = row["gender"]
    max_f    = MAX_FORMANT_F if gender == "f" else MAX_FORMANT_M

    # Load sound only when wav file changes
    if wav_path != current_wav:
        try:
            current_sound = parselmouth.Sound(wav_path)
            current_wav   = wav_path
        except Exception as e:
            print(f"  ERROR loading {wav_path}: {e}")
            current_sound = None
            current_wav   = None

    midpoint = (row["onset"] + row["offset"]) / 2
    phoneme  = row["phoneme"]

    # Initialise all feature columns to NaN
    record = {**row,
              "F1": np.nan, "F2": np.nan, "F3": np.nan,
              "F1_25": np.nan, "F2_25": np.nan, "F3_25": np.nan,
              "F1_75": np.nan, "F2_75": np.nan, "F3_75": np.nan,
              "f0": np.nan, "SCG": np.nan}

    if current_sound is None:
        results.append(record)
        continue

    # ── F1, F2, F3 at midpoint ────────────────────────────────────────────────
    try:
        formants = current_sound.to_formant_burg(
            time_step=TIME_STEP,
            max_number_of_formants=N_FORMANTS,
            maximum_formant=max_f,
            window_length=WINDOW_LENGTH,
            pre_emphasis_from=50
        )
        record["F1"] = formants.get_value_at_time(1, midpoint)
        record["F2"] = formants.get_value_at_time(2, midpoint)
        record["F3"] = formants.get_value_at_time(3, midpoint)
    except Exception:
        pass

    # ── Trajectory measurements for long vowels (duration > 80ms) ────────────
    if phoneme in VOWELS and row["duration_ms"] > LONG_VOWEL_THRESHOLD_MS:
        t25 = row["onset"] + 0.25 * (row["offset"] - row["onset"])
        t75 = row["onset"] + 0.75 * (row["offset"] - row["onset"])
        try:
            record["F1_25"] = formants.get_value_at_time(1, t25)
            record["F2_25"] = formants.get_value_at_time(2, t25)
            record["F3_25"] = formants.get_value_at_time(3, t25)
            record["F1_75"] = formants.get_value_at_time(1, t75)
            record["F2_75"] = formants.get_value_at_time(2, t75)
            record["F3_75"] = formants.get_value_at_time(3, t75)
        except Exception:
            pass

    # ── f0 (voiced segments only) ─────────────────────────────────────────────
    try:
        pitch = current_sound.to_pitch()
        f0_val = pitch.get_value_at_time(midpoint)
        if f0_val and not np.isnan(f0_val):
            record["f0"] = f0_val
    except Exception:
        pass

    # ── Spectral Centre of Gravity (fricatives only) ──────────────────────────
    if phoneme in FRICATIVES:
        try:
            segment = current_sound.extract_part(
                from_time=row["onset"],
                to_time=row["offset"],
                preserve_times=False
            )
            spectrum = segment.to_spectrum()
            record["SCG"] = spectrum.get_centre_of_gravity()
        except Exception:
            pass

    results.append(record)

# ── Save ──────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv(OUT_CSV, index=False)

total = len(out_df)
vowel_mask = out_df["phoneme"].isin(VOWELS)
fric_mask  = out_df["phoneme"].isin(FRICATIVES)

print(f"\n✅ Done! Saved to {OUT_CSV}")
print(f"   Total tokens : {total}")

# ── Overall missing value summary ─────────────────────────────────────────────
print(f"\n── Overall missing value summary ──────────────────────")
for col in ["F1", "F2", "F3", "f0", "SCG"]:
    n = out_df[col].isna().sum()
    print(f"   {col:5s} missing: {n:5d} ({100*n/total:.1f}%)")

# ── f0 missing by phoneme class ───────────────────────────────────────────────
print(f"\n── f0 missing by phoneme class ────────────────────────")
print(f"   Vowels     : {out_df.loc[vowel_mask,  'f0'].isna().sum()} / {vowel_mask.sum()}")
print(f"   Fricatives : {out_df.loc[fric_mask,   'f0'].isna().sum()} / {fric_mask.sum()}")
other_mask = ~vowel_mask & ~fric_mask
print(f"   Other cons : {out_df.loc[other_mask,  'f0'].isna().sum()} / {other_mask.sum()}")

# ── f0 missing by group (L1 x Gender) ────────────────────────────────────────
print(f"\n── f0 missing by group (L1 x Gender) ─────────────────")
for (l1, gen), sub in out_df.groupby(["l1_status", "gender"]):
    n = sub["f0"].isna().sum()
    print(f"   {l1}/{gen}: {n} / {len(sub)} ({100*n/len(sub):.1f}%)")

# ── SCG missing by group (L1 x Gender) ───────────────────────────────────────
print(f"\n── SCG missing by group (L1 x Gender) ────────────────")
for (l1, gen), sub in out_df.groupby(["l1_status", "gender"]):
    n = sub["SCG"].isna().sum()
    print(f"   {l1}/{gen}: {n} / {len(sub)} ({100*n/len(sub):.1f}%)")

# ── Trajectory coverage ───────────────────────────────────────────────────────
print(f"\n── Trajectory (25%/75%) coverage ──────────────────────")
long_vowels = out_df[vowel_mask & (out_df["duration_ms"] > LONG_VOWEL_THRESHOLD_MS)]
print(f"   Long vowels (>{LONG_VOWEL_THRESHOLD_MS}ms): {len(long_vowels)}")
print(f"   F1_25 available: {long_vowels['F1_25'].notna().sum()}")
print(f"   F1_75 available: {long_vowels['F1_75'].notna().sum()}")