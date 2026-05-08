"""
analyse_descriptive.py
§5 Descriptive Statistics
- Table: mean, median, SD, IQR, CV of F1/F2 per phoneme x 4 groups
- Variance decomposition of F1 per phoneme
- Figure 1: Vowel chart
- Figure 2: Box plots F1/F2 by phoneme, L1 status and gender
- Figure 3: Strip/violin plots intra-speaker variability
- Figure 4: PCA projections (by phoneme, L1 status, gender)
- Figure 5: UMAP projections (by phoneme, L1 status, gender)
- Between-class variance ratio
- Cosine similarity within/between phoneme
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
 
# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = Path("results/figures")
TAB_DIR = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
 
# ── French oral vowels ────────────────────────────────────────────────────────
VOWELS = {"i", "e", "ɛ", "a", "ɑ", "o", "ɔ", "u", "y", "ø", "œ", "ə",
          "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"}
 
# ── Load acoustic data ────────────────────────────────────────────────────────
print("Loading data...")
df     = pd.read_csv("data/processed/features_acoustic_norm.csv")
vowels = df[df["phoneme"].isin(VOWELS)].copy()
vowels["group"] = vowels["l1_status"] + "/" + vowels["gender"]
print(f"Total tokens: {len(df)}, Vowel tokens: {len(vowels)}")
 
# ── Load neural representations ───────────────────────────────────────────────
w_low  = np.load("data/processed/features_whisper_pca_low.npz")
w_high = np.load("data/processed/features_whisper_pca_high.npz")
x_low  = np.load("data/processed/features_xlsr_pca_low.npz")
x_mid  = np.load("data/processed/features_xlsr_pca_mid.npz")
x_high = np.load("data/processed/features_xlsr_pca_high.npz")
 
w_raw     = np.load("data/processed/features_whisper.npz", allow_pickle=True)
x_raw     = np.load("data/processed/features_xlsr.npz",   allow_pickle=True)
meta_keys = list(w_raw["meta_keys"])
meta_df   = pd.DataFrame(w_raw["meta"], columns=meta_keys)
meta_df["group"] = meta_df["l1_status"] + "/" + meta_df["gender"]
 
# ─────────────────────────────────────────────────────────────────────────────
# §5.1 TABLE: Descriptive statistics per phoneme x 4 groups
# ─────────────────────────────────────────────────────────────────────────────
print("\n§5.1 Descriptive statistics table (4 groups)...")
 
rows = []
for phoneme, grp in vowels.groupby("phoneme"):
    for group in ["L1/f", "L1/m", "L2/f", "L2/m"]:
        sub = grp[grp["group"] == group]
        for formant in ["F1_lob", "F2_lob"]:
            vals = sub[formant].dropna()
            if len(vals) < 3:
                continue
            rows.append({
                "phoneme":  phoneme,
                "group":    group,
                "formant":  formant,
                "n":        len(vals),
                "mean":     round(vals.mean(), 3),
                "median":   round(vals.median(), 3),
                "sd":       round(vals.std(), 3),
                "iqr":      round(vals.quantile(0.75) - vals.quantile(0.25), 3),
                "cv":       round(vals.std() / abs(vals.mean()), 3) if vals.mean() != 0 else np.nan
            })
 
stats_df = pd.DataFrame(rows)
stats_df.to_csv(TAB_DIR / "descriptive_acoustic.csv", index=False)
print(f"   Saved → results/tables/descriptive_acoustic.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# §5.1 VARIANCE DECOMPOSITION of F1 per phoneme
# ─────────────────────────────────────────────────────────────────────────────
print("\n§5.1 Variance decomposition of F1...")
 
var_rows = []
for phoneme, grp in vowels.groupby("phoneme"):
    vals = grp[["speaker_id", "F1_lob"]].dropna()
    if len(vals) < 10:
        continue
    total_var = vals["F1_lob"].var()
    inter_var = vals.groupby("speaker_id")["F1_lob"].mean().var()
    intra_var = vals.groupby("speaker_id")["F1_lob"].var().mean()
    resid_var = max(total_var - inter_var - intra_var, 0)
    var_rows.append({
        "phoneme":   phoneme,
        "total_var": round(total_var, 4),
        "inter_spk": round(inter_var, 4),
        "intra_spk": round(intra_var, 4),
        "residual":  round(resid_var, 4),
        "pct_inter": round(100*inter_var/total_var, 1) if total_var > 0 else 0
    })
 
var_df = pd.DataFrame(var_rows)
var_df.to_csv(TAB_DIR / "variance_decomposition.csv", index=False)
print(f"   Saved → results/tables/variance_decomposition.csv")
print(var_df.to_string(index=False))
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Vowel chart
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 1: Vowel chart...")
 
GROUP_COLORS = {
    "L1/f": "#2196F3",
    "L1/m": "#1565C0",
    "L2/f": "#F44336",
    "L2/m": "#B71C1C",
}
 
fig, ax = plt.subplots(figsize=(11, 9))
 
for group, sub in vowels.groupby("group"):
    color = GROUP_COLORS.get(group, "gray")
    for phoneme, pgrp in sub.groupby("phoneme"):
        f1_vals = pgrp["F1_lob"].dropna().values
        f2_vals = pgrp["F2_lob"].dropna().values
        if len(f1_vals) < 3:
            continue
        f1_mean = f1_vals.mean()
        f2_mean = f2_vals.mean()
        if len(f1_vals) > 5:
            try:
                cov = np.cov(f1_vals, f2_vals)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                if np.all(eigenvalues > 0) and np.all(np.isfinite(eigenvalues)):
                    order        = eigenvalues.argsort()[::-1]
                    eigenvalues  = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]
                    angle  = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                    width  = 2 * 1.96 * np.sqrt(eigenvalues[0])
                    height = 2 * 1.96 * np.sqrt(eigenvalues[1])
                    if width < 1.2 and height < 1.2:
                        ellipse = mpatches.Ellipse(
                            (f2_mean, f1_mean), width, height,
                            angle=angle, fill=False,
                            edgecolor=color, alpha=0.5, linewidth=1.2
                        )
                        ax.add_patch(ellipse)
            except Exception:
                pass
        ax.annotate(phoneme, (f2_mean, f1_mean),
                    fontsize=10, ha="center", va="center",
                    color=color, fontweight="bold")
 
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_xlabel("F2 (Lobanov normalised)", fontsize=12)
ax.set_ylabel("F1 (Lobanov normalised)", fontsize=12)
ax.set_title("Figure 1: French Vowel Space (Lobanov normalised)\nper-phoneme centroids with 95% confidence ellipses", fontsize=12)
legend_handles = [mpatches.Patch(color=c, label=l)
                  for l, c in GROUP_COLORS.items()]
ax.legend(handles=legend_handles, loc="lower right", fontsize=10)
ax.set_xlim(2, -2)
ax.set_ylim(2, -2)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_vowel_chart.png", dpi=150)
plt.close()
print(f"   Saved → results/figures/fig1_vowel_chart.png")
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Box plots F1/F2 by phoneme, L1 status AND gender
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 2: Box plots...")
 
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
palette = {"L1/f": "#2196F3", "L1/m": "#1565C0",
           "L2/f": "#F44336", "L2/m": "#B71C1C"}
 
for ax, formant in zip(axes, ["F1_lob", "F2_lob"]):
    sns.boxplot(
        data=vowels, x="phoneme", y=formant,
        hue="group", palette=palette,
        ax=ax, width=0.7
    )
    ax.set_title(f"{formant.replace('_lob','')} by phoneme and group", fontsize=11)
    ax.set_xlabel("Phoneme", fontsize=10)
    ax.set_ylabel(f"{formant.replace('_lob','')} (Lobanov)", fontsize=10)
    ax.legend(title="Group", fontsize=9, loc="upper right")
 
plt.suptitle("Figure 2: F1 and F2 by phoneme and speaker group", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_boxplots_formants.png", dpi=150)
plt.close()
print(f"   Saved → results/figures/fig2_boxplots_formants.png")
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Strip/violin plots — intra-speaker variability
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 3: Strip/violin plots...")
 
selected_vowels = ["i", "e", "a", "u", "y", "ɛ"]
sub_vowels = vowels[vowels["phoneme"].isin(selected_vowels)].copy()
 
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, formant in zip(axes, ["F1_lob", "F2_lob"]):
    sns.violinplot(
        data=sub_vowels, x="phoneme", y=formant,
        hue="l1_status",
        palette={"L1": "#2196F3", "L2": "#F44336"},
        ax=ax, split=True, inner="box", alpha=0.7
    )
    sns.stripplot(
        data=sub_vowels, x="phoneme", y=formant,
        hue="l1_status",
        palette={"L1": "#1565C0", "L2": "#B71C1C"},
        ax=ax, dodge=True, size=2, alpha=0.4, legend=False
    )
    ax.set_title(f"{formant.replace('_lob','')} intra-speaker variability", fontsize=11)
    ax.set_xlabel("Phoneme", fontsize=10)
    ax.set_ylabel(f"{formant.replace('_lob','')} (Lobanov)", fontsize=10)
    ax.legend(title="L1 status", fontsize=9)
 
plt.suptitle("Figure 3: Intra-speaker variability (selected vowels)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_violin_strip.png", dpi=150)
plt.close()
print(f"   Saved → results/figures/fig3_violin_strip.png")
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: PCA projections — by phoneme, L1 status, gender
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 4: PCA projections...")
 
COLORS_PHONEME = plt.cm.tab20.colors
phoneme_list  = sorted(meta_df["phoneme"].unique())
phoneme_color = {p: COLORS_PHONEME[i % 20] for i, p in enumerate(phoneme_list)}
 
pca_configs = [
    (w_low["pca_2d"],  "Whisper L4"),
    (w_high["pca_2d"], "Whisper L20"),
    (x_low["pca_2d"],  "XLS-R L4"),
    (x_mid["pca_2d"],  "XLS-R L12"),
    (x_high["pca_2d"], "XLS-R L20"),
]
 
for coords, model_name in pca_configs:
    mask = (
        (np.abs(coords[:, 0] - coords[:, 0].mean()) < 3 * coords[:, 0].std()) &
        (np.abs(coords[:, 1] - coords[:, 1].mean()) < 3 * coords[:, 1].std())
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
    # By phoneme
    colors_ph = [phoneme_color.get(p, "gray") for p in meta_df["phoneme"][mask]]
    axes[0].scatter(coords[mask, 0], coords[mask, 1],
                    c=colors_ph, alpha=0.3, s=5, rasterized=True)
    axes[0].set_title("By phoneme", fontsize=10)
 
    # By L1 status
    colors_l1 = meta_df["l1_status"][mask].map({"L1": "#2196F3", "L2": "#F44336"}).values
    axes[1].scatter(coords[mask, 0], coords[mask, 1],
                    c=colors_l1, alpha=0.3, s=5, rasterized=True)
    axes[1].set_title("By L1 status", fontsize=10)
    axes[1].legend(handles=[mpatches.Patch(color="#2196F3", label="L1"),
                              mpatches.Patch(color="#F44336", label="L2")], fontsize=8)
 
    # By gender
    colors_gen = meta_df["gender"][mask].map({"f": "#E91E63", "m": "#4CAF50"}).values
    axes[2].scatter(coords[mask, 0], coords[mask, 1],
                    c=colors_gen, alpha=0.3, s=5, rasterized=True)
    axes[2].set_title("By gender", fontsize=10)
    axes[2].legend(handles=[mpatches.Patch(color="#E91E63", label="Female"),
                              mpatches.Patch(color="#4CAF50", label="Male")], fontsize=8)
 
    for ax in axes:
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
 
    plt.suptitle(f"Figure 4: PCA projections — {model_name}", fontsize=12)
    plt.tight_layout()
    fname = f"fig4_pca_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()
    print(f"   Saved → results/figures/{fname}")
 
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: UMAP projections — by phoneme, L1 status, gender
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 5: UMAP projections...")
 
umap_configs = [
    (w_low["umap_2d"],  "Whisper L4"),
    (w_high["umap_2d"], "Whisper L20"),
    (x_low["umap_2d"],  "XLS-R L4"),
    (x_mid["umap_2d"],  "XLS-R L12"),
    (x_high["umap_2d"], "XLS-R L20"),
]
 
for coords, model_name in umap_configs:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
    # By phoneme
    colors_ph = [phoneme_color.get(p, "gray") for p in meta_df["phoneme"]]
    axes[0].scatter(coords[:, 0], coords[:, 1],
                    c=colors_ph, alpha=0.3, s=5, rasterized=True)
    axes[0].set_title("By phoneme", fontsize=10)
 
    # By L1 status
    colors_l1 = meta_df["l1_status"].map({"L1": "#2196F3", "L2": "#F44336"}).values
    axes[1].scatter(coords[:, 0], coords[:, 1],
                    c=colors_l1, alpha=0.3, s=5, rasterized=True)
    axes[1].set_title("By L1 status", fontsize=10)
    axes[1].legend(handles=[mpatches.Patch(color="#2196F3", label="L1"),
                              mpatches.Patch(color="#F44336", label="L2")], fontsize=8)
 
    # By gender
    colors_gen = meta_df["gender"].map({"f": "#E91E63", "m": "#4CAF50"}).values
    axes[2].scatter(coords[:, 0], coords[:, 1],
                    c=colors_gen, alpha=0.3, s=5, rasterized=True)
    axes[2].set_title("By gender", fontsize=10)
    axes[2].legend(handles=[mpatches.Patch(color="#E91E63", label="Female"),
                              mpatches.Patch(color="#4CAF50", label="Male")], fontsize=8)
 
    for ax in axes:
        ax.set_xlabel("UMAP1", fontsize=9)
        ax.set_ylabel("UMAP2", fontsize=9)
 
    plt.suptitle(f"Figure 5: UMAP projections — {model_name}", fontsize=12)
    plt.tight_layout()
    fname = f"fig5_umap_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()
    print(f"   Saved → results/figures/{fname}")
 
# ─────────────────────────────────────────────────────────────────────────────
# §5.2 Between-class variance ratio
# ─────────────────────────────────────────────────────────────────────────────
print("\n§5.2 Between-class variance ratio...")
 
bcv_rows = []
neural_configs = [
    (w_low["pca_2d"],   w_low["umap_2d"],   "Whisper L4"),
    (w_high["pca_2d"],  w_high["umap_2d"],  "Whisper L20"),
    (x_low["pca_2d"],   x_low["umap_2d"],   "XLS-R L4"),
    (x_mid["pca_2d"],   x_mid["umap_2d"],   "XLS-R L12"),
    (x_high["pca_2d"],  x_high["umap_2d"],  "XLS-R L20"),
]
 
for pca_coords, umap_coords, name in neural_configs:
    for coords, method in [(pca_coords, "PCA"), (umap_coords, "UMAP")]:
        phonemes = meta_df["phoneme"].values
        total_var = coords.var(axis=0).sum()
        centroids = np.array([coords[phonemes == p].mean(axis=0)
                               for p in np.unique(phonemes)])
        between_var = centroids.var(axis=0).sum()
        ratio = between_var / total_var if total_var > 0 else 0
        bcv_rows.append({
            "model":       name,
            "method":      method,
            "between_var": round(between_var, 4),
            "total_var":   round(total_var, 4),
            "ratio":       round(ratio, 4)
        })
        print(f"   {name:12s} {method:4s}  ratio={ratio:.4f}")
 
bcv_df = pd.DataFrame(bcv_rows)
bcv_df.to_csv(TAB_DIR / "between_class_variance.csv", index=False)
print(f"   Saved → results/tables/between_class_variance.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# §5.2 Cosine similarity within/between phoneme
# ─────────────────────────────────────────────────────────────────────────────
print("\n§5.2 Cosine similarity...")
 
cos_rows = []
raw_configs = [
    (w_raw["vectors_low"],  "Whisper L4"),
    (w_raw["vectors_high"], "Whisper L20"),
    (x_raw["vectors_low"],  "XLS-R L4"),
    (x_raw["vectors_mid"],  "XLS-R L12"),
    (x_raw["vectors_high"], "XLS-R L20"),
]
 
N_SAMPLE = 3000
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(meta_df), size=min(N_SAMPLE, len(meta_df)), replace=False)
 
for vectors, name in raw_configs:
    vecs   = vectors[sample_idx]
    phones = meta_df["phoneme"].values[sample_idx]
 
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs_norm = vecs / norms
 
    sim_matrix = vecs_norm @ vecs_norm.T
 
    within_sims  = []
    between_sims = []
 
    for i in range(len(phones)):
        for j in range(i + 1, len(phones)):
            s = sim_matrix[i, j]
            if phones[i] == phones[j]:
                within_sims.append(s)
            else:
                between_sims.append(s)
 
    within_mean  = np.mean(within_sims)
    between_mean = np.mean(between_sims)
    ratio = within_mean / between_mean if between_mean != 0 else 0
 
    cos_rows.append({
        "model":       name,
        "within_sim":  round(within_mean, 4),
        "between_sim": round(between_mean, 4),
        "ratio":       round(ratio, 4)
    })
    print(f"   {name:12s}  within={within_mean:.4f}  between={between_mean:.4f}  ratio={ratio:.4f}")
 
cos_df = pd.DataFrame(cos_rows)
cos_df.to_csv(TAB_DIR / "cosine_similarity.csv", index=False)
print(f"   Saved → results/tables/cosine_similarity.csv")
 
print("\n✅ §5 Descriptive statistics complete!")