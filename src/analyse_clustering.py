"""
analyse_clustering.py
§9 Hierarchical Clustering
9.1 Clustering of French oral vowels (acoustic + neural)
9.2 Consonants vs vowels
9.3 Speaker clustering
9.4 Number of clusters (silhouette + dendrogram)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = Path("results/figures")
TAB_DIR = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Phoneme sets ──────────────────────────────────────────────────────────────
VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]

# Ground truth: front/back
FRONT_BACK = {
    "i": "front", "e": "front", "ɛ": "front", "y": "front", "ø": "front",
    "a": "front", "ɑ": "back",  "o": "back",  "u": "back",  "ə": "central"
}

# Ground truth: high/mid/low
HEIGHT = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",  "ɛ": "mid", "ə": "mid",
    "a": "low",  "ɑ": "low"
}

# Consonants for §9.2
CONSONANTS = ["p", "t", "k", "b", "d", "g", "s", "z", "f", "v", "m", "n", "l", "R"]
CONSONANT_CLASS = {
    "p": "stop", "t": "stop", "k": "stop",
    "b": "stop", "d": "stop", "g": "stop",
    "s": "fricative", "z": "fricative", "f": "fricative", "v": "fricative",
    "m": "nasal", "n": "nasal",
    "l": "lateral", "R": "rhotic"
}
CV_CLASS = {p: "consonant" for p in CONSONANTS}
CV_CLASS.update({v: "vowel" for v in VOWELS})

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df     = pd.read_csv("data/processed/features_acoustic_norm.csv")
vowels = df[df["phoneme"].isin(VOWELS)].copy()
consos = df[df["phoneme"].isin(CONSONANTS)].copy()

w_raw     = np.load("data/processed/features_whisper.npz", allow_pickle=True)
x_raw     = np.load("data/processed/features_xlsr.npz",   allow_pickle=True)
meta_keys = list(w_raw["meta_keys"])
meta_df   = pd.DataFrame(w_raw["meta"], columns=meta_keys)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_dendrogram(Z, labels, title, fname, color_map=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=labels, ax=ax,
               leaf_font_size=12, leaf_rotation=0,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Distance", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150)
    plt.close()
    print(f"   Saved → results/figures/{fname}")

def compute_ari(labels_pred, labels_true_dict, phonemes):
    """Compute ARI between predicted clusters and ground truth."""
    true = [labels_true_dict[p] for p in phonemes]
    unique_true = list(set(true))
    true_int = [unique_true.index(t) for t in true]
    return adjusted_rand_score(true_int, labels_pred)

def silhouette(dist_matrix, labels):
    """Silhouette score from precomputed distance matrix."""
    if len(set(labels)) < 2:
        return np.nan
    return silhouette_score(dist_matrix, labels, metric="precomputed")

# ─────────────────────────────────────────────────────────────────────────────
# §9.1 VOWEL CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§9.1 Vowel clustering")
print("="*60)

ari_rows = []

# ── Acoustic: mean F1_lob, F2_lob per phoneme ────────────────────────────────
print("\n── Acoustic ──────────────────────────────────────────")
ac_means = {}
for p in VOWELS:
    sub = vowels[vowels["phoneme"] == p][["F1_lob", "F2_lob"]].dropna()
    if len(sub) >= 3:
        ac_means[p] = sub.values.mean(axis=0)

valid_vowels = [p for p in VOWELS if p in ac_means]
ac_matrix    = np.array([ac_means[p] for p in valid_vowels])

# Ward's linkage with Euclidean distance
ac_dist = squareform(pdist(ac_matrix, metric="euclidean"))
Z_ac    = linkage(squareform(ac_dist), method="ward", optimal_ordering=True)

# Dendrogramm
plot_dendrogram(Z_ac, [f"/{p}/" for p in valid_vowels],
                "Figure 11a: Vowel Dendrogram — Acoustic (F1/F2)",
                "fig11a_dendrogram_acoustic.png")

# ARI for k=2 (front/back) and k=3 (high/mid/low)
for k, gt_dict, gt_name in [
    (2, FRONT_BACK, "front_back"),
    (3, HEIGHT,     "height")
]:
    pred = fcluster(Z_ac, k, criterion="maxclust")
    ari  = compute_ari(pred, gt_dict, valid_vowels)
    sil  = silhouette(ac_dist, pred)
    print(f"   Acoustic k={k} ({gt_name}): ARI={ari:.4f}  sil={sil:.4f}")
    ari_rows.append({
        "representation": "Acoustic",
        "ground_truth": gt_name,
        "k": k,
        "ARI": round(ari, 4),
        "silhouette": round(sil, 4)
    })

# ── Neural vowel clustering ───────────────────────────────────────────────────
neural_configs = [
    (w_raw["vectors_low"],  "Whisper_L4"),
    (w_raw["vectors_high"], "Whisper_L20"),
    (x_raw["vectors_low"],  "XLSR_L4"),
    (x_raw["vectors_mid"],  "XLSR_L12"),
    (x_raw["vectors_high"], "XLSR_L20"),
]

for vectors, model_name in neural_configs:
    print(f"\n── {model_name} ──────────────────────────────────────")
    ne_means = {}
    for p in VOWELS:
        mask = (meta_df["phoneme"] == p).values
        vecs = vectors[mask]
        if len(vecs) >= 3:
            c = vecs.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-10)
            ne_means[p] = c

    valid_ne = [p for p in VOWELS if p in ne_means]
    ne_matrix = np.array([ne_means[p] for p in valid_ne])

    # Cosine distance matrix
    ne_dist = np.zeros((len(valid_ne), len(valid_ne)))
    for i in range(len(valid_ne)):
        for j in range(len(valid_ne)):
            ne_dist[i, j] = 1 - np.dot(ne_matrix[i], ne_matrix[j])
    np.fill_diagonal(ne_dist, 0)

    Z_ne = linkage(squareform(ne_dist), method="ward", optimal_ordering=True)

    plot_dendrogram(Z_ne, [f"/{p}/" for p in valid_ne],
                    f"Figure 11: Vowel Dendrogram — {model_name}",
                    f"fig11_dendrogram_{model_name}.png")

    for k, gt_dict, gt_name in [
        (2, FRONT_BACK, "front_back"),
        (3, HEIGHT,     "height")
    ]:
        pred = fcluster(Z_ne, k, criterion="maxclust")
        ari  = compute_ari(pred, gt_dict, valid_ne)
        sil  = silhouette(ne_dist, pred)
        print(f"   {model_name} k={k} ({gt_name}): ARI={ari:.4f}  sil={sil:.4f}")
        ari_rows.append({
            "representation": model_name,
            "ground_truth":   gt_name,
            "k":              k,
            "ARI":            round(ari, 4),
            "silhouette":     round(sil, 4)
        })

ari_df = pd.DataFrame(ari_rows)
ari_df.to_csv(TAB_DIR / "clustering_ari_vowels.csv", index=False)
print(f"\n   Saved → results/tables/clustering_ari_vowels.csv")
print(ari_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# §9.2 CONSONANTS VS VOWELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§9.2 Consonants vs Vowels")
print("="*60)

ALL_PHONES = VOWELS + CONSONANTS
cv_ari_rows = []

# Acoustic: F1, F2, duration, SCG
print("\n── Acoustic (F1, F2, duration, SCG) ─────────────────")
all_df   = df[df["phoneme"].isin(ALL_PHONES)].copy()
ac_cv_means = {}
for p in ALL_PHONES:
    sub = all_df[all_df["phoneme"] == p]
    feats = []
    for col in ["F1_lob", "F2_lob", "duration_ms", "SCG"]:
        if col in sub.columns:
            val = sub[col].dropna().mean()
            feats.append(val if not np.isnan(val) else 0.0)
        else:
            feats.append(0.0)
    if len(sub) >= 3:
        ac_cv_means[p] = np.array(feats)

valid_cv = [p for p in ALL_PHONES if p in ac_cv_means]
ac_cv_matrix = np.array([ac_cv_means[p] for p in valid_cv])
ac_cv_matrix = StandardScaler().fit_transform(ac_cv_matrix)

ac_cv_dist = squareform(pdist(ac_cv_matrix, metric="euclidean"))
Z_cv_ac    = linkage(squareform(ac_cv_dist), method="ward", optimal_ordering=True)

plot_dendrogram(Z_cv_ac, [f"/{p}/" for p in valid_cv],
                "Figure 12a: Consonants+Vowels Dendrogram — Acoustic",
                "fig12a_dendrogram_cv_acoustic.png")

pred_cv = fcluster(Z_cv_ac, 2, criterion="maxclust")
gt_cv   = {p: CV_CLASS.get(p, "vowel") for p in valid_cv}
ari_cv  = compute_ari(pred_cv, gt_cv, valid_cv)
print(f"   Acoustic C/V ARI = {ari_cv:.4f}")
cv_ari_rows.append({"representation": "Acoustic", "ARI_CV": round(ari_cv, 4)})

# Neural C/V
for vectors, model_name in neural_configs:
    ne_cv_means = {}
    for p in ALL_PHONES:
        mask = (meta_df["phoneme"] == p).values
        vecs = vectors[mask]
        if len(vecs) >= 3:
            c = vecs.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-10)
            ne_cv_means[p] = c

    valid_cv_ne = [p for p in ALL_PHONES if p in ne_cv_means]
    ne_cv_matrix = np.array([ne_cv_means[p] for p in valid_cv_ne])

    ne_cv_dist = np.zeros((len(valid_cv_ne), len(valid_cv_ne)))
    for i in range(len(valid_cv_ne)):
        for j in range(len(valid_cv_ne)):
            ne_cv_dist[i, j] = 1 - np.dot(ne_cv_matrix[i], ne_cv_matrix[j])
    np.fill_diagonal(ne_cv_dist, 0)

    Z_cv_ne = linkage(squareform(ne_cv_dist), method="ward", optimal_ordering=True)

    plot_dendrogram(Z_cv_ne, [f"/{p}/" for p in valid_cv_ne],
                    f"Figure 12: C/V Dendrogram — {model_name}",
                    f"fig12_dendrogram_cv_{model_name}.png")

    pred_cv_ne = fcluster(Z_cv_ne, 2, criterion="maxclust")
    gt_cv_ne   = {p: CV_CLASS.get(p, "vowel") for p in valid_cv_ne}
    ari_cv_ne  = compute_ari(pred_cv_ne, gt_cv_ne, valid_cv_ne)
    print(f"   {model_name} C/V ARI = {ari_cv_ne:.4f}")
    cv_ari_rows.append({"representation": model_name, "ARI_CV": round(ari_cv_ne, 4)})

cv_ari_df = pd.DataFrame(cv_ari_rows)
cv_ari_df.to_csv(TAB_DIR / "clustering_ari_cv.csv", index=False)
print(f"   Saved → results/tables/clustering_ari_cv.csv")

# ─────────────────────────────────────────────────────────────────────────────
# §9.3 SPEAKER CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§9.3 Speaker clustering")
print("="*60)

speakers   = sorted(meta_df["speaker_id"].unique())
spk_meta   = meta_df.drop_duplicates("speaker_id").set_index("speaker_id")
spk_l1     = [spk_meta.loc[s, "l1_status"] for s in speakers]
spk_gender = [spk_meta.loc[s, "gender"]    for s in speakers]

spk_ari_rows = []

# Acoustic speaker vectors: concatenate per-phoneme mean F1/F2
print("\n── Acoustic speaker clustering ───────────────────────")
spk_ac_vecs = []
for spk in speakers:
    feats = []
    for p in VOWELS:
        sub = vowels[(vowels["speaker_id"] == spk) & (vowels["phoneme"] == p)]
        f1 = sub["F1_lob"].mean() if len(sub) > 0 else 0.0
        f2 = sub["F2_lob"].mean() if len(sub) > 0 else 0.0
        feats.extend([f1, f2])
    spk_ac_vecs.append(feats)

spk_ac_matrix = StandardScaler().fit_transform(np.array(spk_ac_vecs))
spk_ac_dist   = squareform(pdist(spk_ac_matrix, metric="euclidean"))
Z_spk_ac      = linkage(squareform(spk_ac_dist), method="ward", optimal_ordering=True)

plot_dendrogram(Z_spk_ac, speakers,
                "Figure 13a: Speaker Dendrogram — Acoustic",
                "fig13a_dendrogram_speakers_acoustic.png")

for k, gt_list, gt_name in [
    (2, spk_l1,     "L1_status"),
    (2, spk_gender, "gender")
]:
    pred = fcluster(Z_spk_ac, k, criterion="maxclust")
    unique_gt = list(set(gt_list))
    gt_int    = [unique_gt.index(g) for g in gt_list]
    ari = adjusted_rand_score(gt_int, pred)
    print(f"   Acoustic speakers k={k} ({gt_name}): ARI={ari:.4f}")
    spk_ari_rows.append({
        "representation": "Acoustic",
        "ground_truth":   gt_name,
        "k":              k,
        "ARI":            round(ari, 4)
    })

# Neural speaker clustering
for vectors, model_name in neural_configs:
    spk_ne_vecs = []
    for spk in speakers:
        feats = []
        for p in VOWELS:
            mask = ((meta_df["speaker_id"] == spk) &
                    (meta_df["phoneme"] == p)).values
            vecs = vectors[mask]
            if len(vecs) > 0:
                c = vecs.mean(axis=0)
            else:
                c = np.zeros(vectors.shape[1])
            feats.append(c)
        spk_ne_vecs.append(np.concatenate(feats))

    spk_ne_matrix = StandardScaler().fit_transform(np.array(spk_ne_vecs))
    spk_ne_dist   = squareform(pdist(spk_ne_matrix, metric="euclidean"))
    Z_spk_ne      = linkage(squareform(spk_ne_dist), method="ward",
                             optimal_ordering=True)

    plot_dendrogram(Z_spk_ne, speakers,
                    f"Figure 13: Speaker Dendrogram — {model_name}",
                    f"fig13_dendrogram_speakers_{model_name}.png")

    for k, gt_list, gt_name in [
        (2, spk_l1,     "L1_status"),
        (2, spk_gender, "gender")
    ]:
        pred = fcluster(Z_spk_ne, k, criterion="maxclust")
        unique_gt = list(set(gt_list))
        gt_int    = [unique_gt.index(g) for g in gt_list]
        ari = adjusted_rand_score(gt_int, pred)
        print(f"   {model_name} speakers k={k} ({gt_name}): ARI={ari:.4f}")
        spk_ari_rows.append({
            "representation": model_name,
            "ground_truth":   gt_name,
            "k":              k,
            "ARI":            round(ari, 4)
        })

spk_ari_df = pd.DataFrame(spk_ari_rows)
spk_ari_df.to_csv(TAB_DIR / "clustering_ari_speakers.csv", index=False)
print(f"\n   Saved → results/tables/clustering_ari_speakers.csv")
print(spk_ari_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# §9.4 NUMBER OF CLUSTERS — Silhouette + dendrogram
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§9.4 Number of clusters — Silhouette analysis")
print("="*60)

sil_rows = []

# Acoustic vowels
for k in range(2, 6):
    pred = fcluster(Z_ac, k, criterion="maxclust")
    sil  = silhouette(ac_dist, pred) if len(set(pred)) > 1 else np.nan
    sil_rows.append({"representation": "Acoustic", "k": k,
                     "silhouette": round(sil, 4) if not np.isnan(sil) else np.nan})
    print(f"   Acoustic vowels k={k}: sil={sil:.4f}")

# Neural vowels (best model: XLSR_L12)
ne_means_best = {}
for p in VOWELS:
    mask = (meta_df["phoneme"] == p).values
    vecs = x_raw["vectors_mid"][mask]
    if len(vecs) >= 3:
        c = vecs.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-10)
        ne_means_best[p] = c

valid_best = [p for p in VOWELS if p in ne_means_best]
ne_best    = np.array([ne_means_best[p] for p in valid_best])
ne_best_dist = np.zeros((len(valid_best), len(valid_best)))
for i in range(len(valid_best)):
    for j in range(len(valid_best)):
        ne_best_dist[i, j] = 1 - np.dot(ne_best[i], ne_best[j])
np.fill_diagonal(ne_best_dist, 0)
Z_best = linkage(squareform(ne_best_dist), method="ward", optimal_ordering=True)

for k in range(2, 6):
    pred = fcluster(Z_best, k, criterion="maxclust")
    sil  = silhouette(ne_best_dist, pred) if len(set(pred)) > 1 else np.nan
    sil_rows.append({"representation": "XLSR_L12", "k": k,
                     "silhouette": round(sil, 4) if not np.isnan(sil) else np.nan})
    print(f"   XLSR_L12 vowels k={k}: sil={sil:.4f}")

sil_df = pd.DataFrame(sil_rows)
sil_df.to_csv(TAB_DIR / "clustering_silhouette.csv", index=False)
print(f"   Saved → results/tables/clustering_silhouette.csv")

# Silhouette plot
fig, ax = plt.subplots(figsize=(8, 4))
for rep in sil_df["representation"].unique():
    sub = sil_df[sil_df["representation"] == rep]
    ax.plot(sub["k"], sub["silhouette"], marker="o", label=rep)
ax.set_xlabel("Number of clusters k", fontsize=11)
ax.set_ylabel("Silhouette score", fontsize=11)
ax.set_title("Figure 14: Silhouette scores by k", fontsize=12)
ax.legend(fontsize=9)
ax.set_xticks(range(2, 6))
plt.tight_layout()
plt.savefig(FIG_DIR / "fig14_silhouette.png", dpi=150)
plt.close()
print(f"   Saved → results/figures/fig14_silhouette.png")

print("\n✅ §9 Clustering analysis complete!")