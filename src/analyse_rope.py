"""
analyse_rope.py
§8 Confidence Intervals and ROPE
8.1 Bootstrap CIs on acoustic L1/L2 contrasts (per phoneme)
8.2 Bootstrap CIs on neural cosine distances (per phoneme)
8.3 ROPE definition (acoustic + neural)
8.4 ROPE classification + forest plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = Path("results/figures")
TAB_DIR = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── French oral vowels ────────────────────────────────────────────────────────
VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df     = pd.read_csv("data/processed/features_acoustic_norm.csv")
vowels = df[df["phoneme"].isin(VOWELS)].copy()

w_raw     = np.load("data/processed/features_whisper.npz", allow_pickle=True)
x_raw     = np.load("data/processed/features_xlsr.npz",   allow_pickle=True)
meta_keys = list(w_raw["meta_keys"])
meta_df   = pd.DataFrame(w_raw["meta"], columns=meta_keys)

B   = 2000
rng = np.random.default_rng(42)

def cosine_dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return 1 - np.dot(a, b)

# ─────────────────────────────────────────────────────────────────────────────
# §8.1 BOOTSTRAP CIs ON ACOUSTIC L1/L2 CONTRASTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§8.1 Bootstrap CIs — Acoustic L1/L2 contrasts")
print("="*60)

ac_ci_rows = []

for phoneme in tqdm(VOWELS, desc="Acoustic bootstrap"):
    sub = vowels[vowels["phoneme"] == phoneme].copy()
    if len(sub) < 10:
        continue

    # Speaker-level bootstrap within this phoneme only
    spks = sub["speaker_id"].values
    unique_spks = np.unique(spks)

    for formant in ["F1_lob", "F2_lob"]:
        vals    = sub[formant].values
        labels  = sub["l1_status"].values
        l1_vals = vals[labels == "L1"]
        l2_vals = vals[labels == "L2"]
        if len(l1_vals) < 3 or len(l2_vals) < 3:
            continue

        obs_diff = np.nanmean(l1_vals) - np.nanmean(l2_vals)

        boot_diffs = []
        for _ in range(B):
            boot_spk = rng.choice(unique_spks, size=len(unique_spks), replace=True)
            idx_list = [np.where(spks == s)[0] for s in boot_spk]
            boot_idx = np.concatenate(idx_list)
            bl = labels[boot_idx]
            bv = vals[boot_idx]
            b_l1 = np.nanmean(bv[bl == "L1"]) if (bl == "L1").sum() > 0 else np.nan
            b_l2 = np.nanmean(bv[bl == "L2"]) if (bl == "L2").sum() > 0 else np.nan
            if not np.isnan(b_l1) and not np.isnan(b_l2):
                boot_diffs.append(b_l1 - b_l2)

        if len(boot_diffs) < 100:
            ci = [np.nan, np.nan]
        else:
            ci = np.percentile(boot_diffs, [2.5, 97.5])

        ac_ci_rows.append({
            "phoneme":       phoneme,
            "formant":       formant,
            "obs_diff":      round(obs_diff, 4),
            "ci_low":        round(ci[0], 4) if not np.isnan(ci[0]) else np.nan,
            "ci_high":       round(ci[1], 4) if not np.isnan(ci[1]) else np.nan,
            "excludes_zero": int(ci[0] > 0 or ci[1] < 0) if not np.isnan(ci[0]) else 0
        })
        print(f"   /{phoneme}/ {formant}: diff={obs_diff:.4f} CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

ac_ci_df = pd.DataFrame(ac_ci_rows)
ac_ci_df.to_csv(TAB_DIR / "rope_acoustic_ci.csv", index=False)
print(f"   Saved → results/tables/rope_acoustic_ci.csv")

# ─────────────────────────────────────────────────────────────────────────────
# §8.2 BOOTSTRAP CIs ON NEURAL COSINE DISTANCES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§8.2 Bootstrap CIs — Neural cosine distances")
print("="*60)

neural_configs = [
    (w_raw["vectors_low"],  "Whisper_L4"),
    (w_raw["vectors_high"], "Whisper_L20"),
    (x_raw["vectors_low"],  "XLSR_L4"),
    (x_raw["vectors_mid"],  "XLSR_L12"),
    (x_raw["vectors_high"], "XLSR_L20"),
]

ne_ci_rows = []

for vectors, model_name in neural_configs:
    print(f"\n   {model_name}:")
    for phoneme in tqdm(VOWELS, desc=f"  {model_name}", leave=False):
        # Work only on tokens for this phoneme
        mask   = (meta_df["phoneme"] == phoneme).values
        vecs   = vectors[mask]
        labels = meta_df["l1_status"].values[mask]
        spks   = meta_df["speaker_id"].values[mask]

        l1_idx = labels == "L1"
        l2_idx = labels == "L2"
        if l1_idx.sum() < 3 or l2_idx.sum() < 3:
            continue

        c1 = vecs[l1_idx].mean(axis=0)
        c2 = vecs[l2_idx].mean(axis=0)
        obs_dist = cosine_dist(c1, c2)

        # Bootstrap within this phoneme only
        unique_spks = np.unique(spks)
        boot_dists  = []
        for _ in range(B):
            boot_spk = rng.choice(unique_spks, size=len(unique_spks), replace=True)
            idx_list = [np.where(spks == s)[0] for s in boot_spk]
            boot_idx = np.concatenate(idx_list)
            bl = labels[boot_idx]
            bv = vecs[boot_idx]
            if (bl == "L1").sum() < 2 or (bl == "L2").sum() < 2:
                continue
            bc1 = bv[bl == "L1"].mean(axis=0)
            bc2 = bv[bl == "L2"].mean(axis=0)
            boot_dists.append(cosine_dist(bc1, bc2))

        if len(boot_dists) < 100:
            continue
        ci = np.percentile(boot_dists, [2.5, 97.5])
        ne_ci_rows.append({
            "model":    model_name,
            "phoneme":  phoneme,
            "obs_dist": round(obs_dist, 6),
            "ci_low":   round(ci[0], 6),
            "ci_high":  round(ci[1], 6),
        })
        print(f"   /{phoneme}/: dist={obs_dist:.4f} CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

ne_ci_df = pd.DataFrame(ne_ci_rows)
ne_ci_df.to_csv(TAB_DIR / "rope_neural_ci.csv", index=False)
print(f"   Saved → results/tables/rope_neural_ci.csv")

# ─────────────────────────────────────────────────────────────────────────────
# §8.3 ROPE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§8.3 ROPE definition")
print("="*60)

AC_ROPE_LOW  = -0.1
AC_ROPE_HIGH =  0.1
print(f"   Acoustic ROPE: [{AC_ROPE_LOW}, {AC_ROPE_HIGH}] Lobanov units")

# Neural noise floor: mean intra-speaker cosine distance per phoneme
print("\n   Computing neural noise floor...")
noise_rows = []
for vectors, model_name in neural_configs:
    intra_dists = []
    for phoneme in VOWELS:
        ph_mask = (meta_df["phoneme"] == phoneme).values
        ph_vecs = vectors[ph_mask]
        ph_spks = meta_df["speaker_id"].values[ph_mask]
        for spk in np.unique(ph_spks):
            spk_mask = ph_spks == spk
            sv = ph_vecs[spk_mask]
            if len(sv) < 2:
                continue
            norms = np.linalg.norm(sv, axis=1, keepdims=True)
            norms[norms == 0] = 1
            sv_norm = sv / norms
            sim_mat = sv_norm @ sv_norm.T
            n = len(sv_norm)
            for i in range(n):
                for j in range(i+1, n):
                    intra_dists.append(1 - sim_mat[i, j])

    delta0 = np.mean(intra_dists)
    noise_rows.append({"model": model_name, "delta0": round(delta0, 6)})
    print(f"   {model_name}: delta0 = {delta0:.6f}")

noise_df = pd.DataFrame(noise_rows)
noise_df.to_csv(TAB_DIR / "rope_neural_noise_floor.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# §8.4 ROPE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§8.4 ROPE classification")
print("="*60)

def classify_rope(ci_low, ci_high, rope_low, rope_high):
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "Indeterminate"
    if ci_low >= rope_low and ci_high <= rope_high:
        return "Equivalent"
    elif ci_low > rope_high or ci_high < rope_low:
        return "Non-equivalent"
    else:
        return "Indeterminate"

# Acoustic
ac_ci_df["rope_class"] = ac_ci_df.apply(
    lambda r: classify_rope(r["ci_low"], r["ci_high"], AC_ROPE_LOW, AC_ROPE_HIGH), axis=1)
ac_ci_df.to_csv(TAB_DIR / "rope_acoustic_classified.csv", index=False)
print("\n── Acoustic ROPE classification ──────────────────────")
print(ac_ci_df[["phoneme","formant","obs_diff","ci_low","ci_high","rope_class"]].to_string(index=False))

# Neural
ne_ci_df = ne_ci_df.merge(noise_df, on="model")
ne_ci_df["rope_class"] = ne_ci_df.apply(
    lambda r: classify_rope(r["ci_low"], r["ci_high"], 0, r["delta0"]), axis=1)
ne_ci_df.to_csv(TAB_DIR / "rope_neural_classified.csv", index=False)
print("\n── Neural ROPE classification ────────────────────────")
print(ne_ci_df[["model","phoneme","obs_dist","ci_low","ci_high","delta0","rope_class"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9: Forest plot — Acoustic
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 9: Forest plot — Acoustic...")

COLOR_MAP = {
    "Equivalent":     "#4CAF50",
    "Non-equivalent": "#F44336",
    "Indeterminate":  "#FF9800"
}

for formant in ["F1_lob", "F2_lob"]:
    sub = ac_ci_df[ac_ci_df["formant"] == formant].dropna(subset=["ci_low","ci_high"])
    sub = sub.sort_values("obs_diff").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, row in sub.iterrows():
        color = COLOR_MAP[row["rope_class"]]
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, lw=2)
        ax.scatter(row["obs_diff"], i, color=color, zorder=5, s=50)
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.axvspan(AC_ROPE_LOW, AC_ROPE_HIGH, alpha=0.1, color="gray")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f"/{r['phoneme']}/" for _, r in sub.iterrows()], fontsize=10)
    ax.set_xlabel(f"L1 − L2 difference ({formant.replace('_lob','')} Lobanov)", fontsize=11)
    ax.set_title(f"Figure 9: Forest Plot — Acoustic {formant.replace('_lob','')} L1/L2", fontsize=11)
    handles = [mpatches.Patch(color=c, label=l) for l, c in COLOR_MAP.items()]
    handles.append(mpatches.Patch(color="gray", alpha=0.3, label="ROPE"))
    ax.legend(handles=handles, fontsize=9)
    plt.tight_layout()
    fname = f"fig9_forest_acoustic_{formant.replace('_lob','')}.png"
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()
    print(f"   Saved → results/figures/{fname}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10: Forest plot — Neural
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 10: Forest plot — Neural...")

for model_name in ne_ci_df["model"].unique():
    sub    = ne_ci_df[ne_ci_df["model"] == model_name].copy()
    delta0 = sub["delta0"].iloc[0]
    sub    = sub.sort_values("obs_dist").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, row in sub.iterrows():
        color = COLOR_MAP[row["rope_class"]]
        ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=color, lw=2)
        ax.scatter(row["obs_dist"], i, color=color, zorder=5, s=50)
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.axvspan(0, delta0, alpha=0.1, color="gray", label=f"ROPE [0, {delta0:.4f}]")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f"/{r['phoneme']}/" for _, r in sub.iterrows()], fontsize=10)
    ax.set_xlabel("Cosine distance (L1 vs L2 centroid)", fontsize=11)
    ax.set_title(f"Figure 10: Forest Plot — {model_name} L1/L2", fontsize=11)
    handles = [mpatches.Patch(color=c, label=l) for l, c in COLOR_MAP.items()]
    handles.append(mpatches.Patch(color="gray", alpha=0.3, label=f"ROPE [0, δ₀={delta0:.4f}]"))
    ax.legend(handles=handles, fontsize=9)
    plt.tight_layout()
    fname = f"fig10_forest_{model_name}.png"
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()
    print(f"   Saved → results/figures/{fname}")

print("\n✅ §8 ROPE analysis complete!")