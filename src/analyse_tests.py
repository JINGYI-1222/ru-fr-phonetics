"""
analyse_tests.py
§6 Statistical Tests
6.1 L1 vs L2 on acoustic features (t-test / Mann-Whitney + BH FDR)
6.2 Gender differences (Lobanov normalised, paired test)
6.3 L1 vs L2 on neural representations (permutation test + BH FDR)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = Path("results/figures")
TAB_DIR = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── French oral vowels ────────────────────────────────────────────────────────
VOWELS = {"i", "e", "ɛ", "a", "ɑ", "o", "ɔ", "u", "y", "ø", "œ", "ə",
          "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df     = pd.read_csv("data/processed/features_acoustic_norm.csv")
vowels = df[df["phoneme"].isin(VOWELS)].copy()

w_raw     = np.load("data/processed/features_whisper.npz", allow_pickle=True)
x_raw     = np.load("data/processed/features_xlsr.npz",   allow_pickle=True)
meta_keys = list(w_raw["meta_keys"])
meta_df   = pd.DataFrame(w_raw["meta"], columns=meta_keys)

# ─────────────────────────────────────────────────────────────────────────────
# §6.1 L1 vs L2 on acoustic features
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§6.1 L1 vs L2 on acoustic features")
print("="*60)

test_rows = []
qq_phonemes = ["i", "a", "y", "u", "ɛ"]  # subset for Q-Q plots

# ── Q-Q plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(qq_phonemes), 4, figsize=(16, 4*len(qq_phonemes)))
for row_idx, phoneme in enumerate(qq_phonemes):
    sub = vowels[vowels["phoneme"] == phoneme]
    l1  = sub[sub["l1_status"] == "L1"]["F1_lob"].dropna()
    l2  = sub[sub["l1_status"] == "L2"]["F1_lob"].dropna()
    for col_idx, (vals, label) in enumerate([(l1, "L1 F1"), (l2, "L2 F1")]):
        ax = axes[row_idx, col_idx]
        stats.probplot(vals, dist="norm", plot=ax)
        ax.set_title(f"/{phoneme}/ {label}", fontsize=9)
    for col_idx, (vals, label) in enumerate([(l1, "L1 F2"), (l2, "L2 F2")], start=2):
        f2_vals = sub[sub["l1_status"] == ("L1" if col_idx == 2 else "L2")]["F2_lob"].dropna()
        ax = axes[row_idx, col_idx]
        stats.probplot(f2_vals, dist="norm", plot=ax)
        ax.set_title(f"/{phoneme}/ {'L1' if col_idx==2 else 'L2'} F2", fontsize=9)

plt.suptitle("Figure 7: Q-Q plots for normality check (selected vowels)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig7_qq_plots.png", dpi=150)
plt.close()
print("   Saved → results/figures/fig7_qq_plots.png")

# ── Tests per phoneme ─────────────────────────────────────────────────────────
for phoneme, grp in vowels.groupby("phoneme"):
    for formant in ["F1_lob", "F2_lob"]:
        l1_vals = grp[grp["l1_status"] == "L1"][formant].dropna()
        l2_vals = grp[grp["l1_status"] == "L2"][formant].dropna()
        if len(l1_vals) < 5 or len(l2_vals) < 5:
            continue

        # Normality
        _, p_sw_l1 = shapiro(l1_vals) if len(l1_vals) <= 5000 else (None, 0.0)
        _, p_sw_l2 = shapiro(l2_vals) if len(l2_vals) <= 5000 else (None, 0.0)
        normal = (p_sw_l1 > 0.05) and (p_sw_l2 > 0.05)

        # Homogeneity of variances
        _, p_lev = levene(l1_vals, l2_vals)
        equal_var = p_lev > 0.05

        # Test
        if normal:
            stat, p_val = ttest_ind(l1_vals, l2_vals, equal_var=equal_var)
            test_name = "t-test" if equal_var else "Welch t-test"
        else:
            stat, p_val = mannwhitneyu(l1_vals, l2_vals, alternative="two-sided")
            test_name = "Mann-Whitney U"

        # Effect size (Cohen's d)
        pooled_sd = np.sqrt((l1_vals.std()**2 + l2_vals.std()**2) / 2)
        cohens_d  = (l1_vals.mean() - l2_vals.mean()) / pooled_sd if pooled_sd > 0 else 0

        test_rows.append({
            "phoneme":   phoneme,
            "formant":   formant,
            "n_L1":      len(l1_vals),
            "n_L2":      len(l2_vals),
            "mean_L1":   round(l1_vals.mean(), 3),
            "mean_L2":   round(l2_vals.mean(), 3),
            "p_sw_L1":   round(p_sw_l1, 4),
            "p_sw_L2":   round(p_sw_l2, 4),
            "normal":    normal,
            "p_levene":  round(p_lev, 4),
            "equal_var": equal_var,
            "test":      test_name,
            "stat":      round(stat, 4),
            "p_raw":     round(p_val, 6),
            "cohens_d":  round(cohens_d, 3)
        })

test_df = pd.DataFrame(test_rows)

# BH FDR correction
_, p_fdr, _, _ = multipletests(test_df["p_raw"], method="fdr_bh")
test_df["p_fdr"] = p_fdr.round(6)
test_df["sig_fdr"] = test_df["p_fdr"].apply(
    lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
)

test_df.to_csv(TAB_DIR / "acoustic_tests_L1L2.csv", index=False)
print("\n── Results (FDR corrected) ──────────────────────────────")
print(test_df[["phoneme","formant","test","p_raw","p_fdr","sig_fdr","cohens_d"]].to_string(index=False))
print(f"\n   Saved → results/tables/acoustic_tests_L1L2.csv")

# ─────────────────────────────────────────────────────────────────────────────
# §6.2 Gender differences (after Lobanov normalisation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§6.2 Gender differences (Lobanov normalised)")
print("="*60)

# Paired test at speaker level: per-speaker mean F1 and F2
spk_means = vowels.groupby(["speaker_id", "gender"])[["F1_lob", "F2_lob"]].mean().reset_index()
male   = spk_means[spk_means["gender"] == "m"]
female = spk_means[spk_means["gender"] == "f"]

gender_rows = []
for formant in ["F1_lob", "F2_lob"]:
    m_vals = male[formant].values
    f_vals = female[formant].values
    # Wilcoxon signed-rank (unpaired groups, use Mann-Whitney)
    stat, p_val = mannwhitneyu(m_vals, f_vals, alternative="two-sided")
    gender_rows.append({
        "formant":  formant,
        "mean_m":   round(m_vals.mean(), 3),
        "mean_f":   round(f_vals.mean(), 3),
        "test":     "Mann-Whitney U",
        "stat":     round(stat, 4),
        "p":        round(p_val, 6)
    })
    print(f"   {formant}: mean_m={m_vals.mean():.3f}  mean_f={f_vals.mean():.3f}  p={p_val:.4f}")

gender_df = pd.DataFrame(gender_rows)
gender_df.to_csv(TAB_DIR / "gender_tests.csv", index=False)
print(f"   Saved → results/tables/gender_tests.csv")

# ─────────────────────────────────────────────────────────────────────────────
# §6.3 L1 vs L2 on neural representations (permutation test)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§6.3 L1 vs L2 on neural representations (permutation test)")
print("="*60)

B = 5000  # number of permutations

def cosine_distance(a, b):
    """Cosine distance between two vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return 1 - np.dot(a_norm, b_norm)

neural_configs = [
    (w_raw["vectors_low"],  "Whisper L4"),
    (w_raw["vectors_high"], "Whisper L20"),
    (x_raw["vectors_low"],  "XLS-R L4"),
    (x_raw["vectors_mid"],  "XLS-R L12"),
    (x_raw["vectors_high"], "XLS-R L20"),
]

perm_rows = []
rng = np.random.default_rng(42)

for vectors, model_name in neural_configs:
    print(f"\n   Model: {model_name}")
    model_rows = []

    for phoneme in tqdm(sorted(VOWELS), desc=f"  {model_name}"):
        mask   = meta_df["phoneme"] == phoneme
        vecs   = vectors[mask]
        labels = meta_df["l1_status"][mask].values

        l1_vecs = vecs[labels == "L1"]
        l2_vecs = vecs[labels == "L2"]

        if len(l1_vecs) < 3 or len(l2_vecs) < 3:
            continue

        # Observed centroid distance
        l1_centroid = l1_vecs.mean(axis=0)
        l2_centroid = l2_vecs.mean(axis=0)
        obs_dist = cosine_distance(l1_centroid, l2_centroid)

        # Permutation null distribution
        all_vecs = np.concatenate([l1_vecs, l2_vecs])
        n_l1     = len(l1_vecs)
        null_dists = []
        for _ in range(B):
            perm = rng.permutation(len(all_vecs))
            perm_l1 = all_vecs[perm[:n_l1]].mean(axis=0)
            perm_l2 = all_vecs[perm[n_l1:]].mean(axis=0)
            null_dists.append(cosine_distance(perm_l1, perm_l2))

        p_val = np.mean(np.array(null_dists) >= obs_dist)

        model_rows.append({
            "model":    model_name,
            "phoneme":  phoneme,
            "n_L1":     len(l1_vecs),
            "n_L2":     len(l2_vecs),
            "obs_dist": round(obs_dist, 6),
            "p_raw":    round(p_val, 6)
        })

    # BH FDR correction per model
    if model_rows:
        model_df = pd.DataFrame(model_rows)
        _, p_fdr, _, _ = multipletests(model_df["p_raw"], method="fdr_bh")
        model_df["p_fdr"] = p_fdr.round(6)
        model_df["sig_fdr"] = model_df["p_fdr"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        )
        perm_rows.append(model_df)
        print(model_df[["phoneme","obs_dist","p_raw","p_fdr","sig_fdr"]].to_string(index=False))

perm_df = pd.concat(perm_rows, ignore_index=True)
perm_df.to_csv(TAB_DIR / "neural_tests_L1L2.csv", index=False)
print(f"\n   Saved → results/tables/neural_tests_L1L2.csv")

print("\n✅ §6 Statistical tests complete!")