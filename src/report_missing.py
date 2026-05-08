"""
Quick script to report missing value statistics from features_acoustic.csv
"""
import pandas as pd
from pathlib import Path

VOWELS     = {"i", "e", "ɛ", "a", "ɑ", "o", "ɔ", "u", "y", "ø", "œ", "ə",
              "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"}
FRICATIVES = {"f", "v", "s", "z", "ʃ", "ʒ"}

df         = pd.read_csv("data/processed/features_acoustic.csv")
total      = len(df)
vowel_mask = df["phoneme"].isin(VOWELS)
fric_mask  = df["phoneme"].isin(FRICATIVES)
other_mask = ~vowel_mask & ~fric_mask

print(f"Total tokens: {total}")

print(f"\n── Overall missing value summary ──────────────────────")
for col in ["F1", "F2", "F3", "f0", "SCG"]:
    n = df[col].isna().sum()
    print(f"   {col:5s} missing: {n:5d} ({100*n/total:.1f}%)")

print(f"\n── f0 missing by phoneme class ────────────────────────")
for label, mask in [("Vowels", vowel_mask), ("Fricatives", fric_mask), ("Other cons", other_mask)]:
    n   = df.loc[mask, "f0"].isna().sum()
    tot = mask.sum()
    print(f"   {label:12s}: {n} / {tot} ({100*n/tot:.1f}%)")

print(f"\n── f0 missing by group (L1 x Gender) ─────────────────")
for (l1, gen), sub in df.groupby(["l1_status", "gender"]):
    n = sub["f0"].isna().sum()
    print(f"   {l1}/{gen}: {n} / {len(sub)} ({100*n/len(sub):.1f}%)")

print(f"\n── SCG missing within fricatives ───────────────────────")
fric_df = df[fric_mask]
n_missing = fric_df["SCG"].isna().sum()
n_total   = len(fric_df)
print(f"   Fricative tokens total : {n_total}")
print(f"   SCG available          : {fric_df['SCG'].notna().sum()} ({100*fric_df['SCG'].notna().sum()/n_total:.1f}%)")
print(f"   SCG missing            : {n_missing} ({100*n_missing/n_total:.1f}%)")

print(f"\n── SCG missing by group (within fricatives) ───────────")
for (l1, gen), sub in fric_df.groupby(["l1_status", "gender"]):
    n = sub["SCG"].isna().sum()
    print(f"   {l1}/{gen}: {n} / {len(sub)} ({100*n/len(sub):.1f}%)")

print(f"\n── Trajectory coverage ─────────────────────────────────")
long_vowels = df[vowel_mask & (df["duration_ms"] > 80.0)]
print(f"   Long vowels (>80ms)  : {len(long_vowels)}")
print(f"   F1_25 available      : {long_vowels['F1_25'].notna().sum()}")
print(f"   F1_75 available      : {long_vowels['F1_75'].notna().sum()}")