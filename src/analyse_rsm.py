"""
analyse_rsm.py
§5.3 Cross-Representation Comparison
- Compute pairwise RSM for acoustic, Whisper, and XLS-R representations
- Compare RSMs using Mantel test (rank correlation between upper triangles)
Output: results/tables/mantel_results.csv
        results/figures/fig6_rsm.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = Path("results/figures")
TAB_DIR = Path("results/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/processed/features_acoustic_norm.csv")

w_raw = np.load("data/processed/features_whisper.npz", allow_pickle=True)
x_raw = np.load("data/processed/features_xlsr.npz",   allow_pickle=True)
meta_keys = list(w_raw["meta_keys"])
meta_df   = pd.DataFrame(w_raw["meta"], columns=meta_keys)

# ── Subsample for efficiency ──────────────────────────────────────────────────
# RSM is N×N — use max 1000 tokens for tractability
N_SAMPLE = 1000
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(meta_df), size=min(N_SAMPLE, len(meta_df)), replace=False)
print(f"Using {len(sample_idx)} tokens for RSM computation")

# ── Acoustic features (F1_lob, F2_lob) ───────────────────────────────────────
# Match acoustic df to neural meta by wav_path + onset
acoustic_sub = df.iloc[sample_idx][["F1_lob", "F2_lob"]].fillna(0).values

# ── Neural features ───────────────────────────────────────────────────────────
w_low  = w_raw["vectors_low"][sample_idx]
w_high = w_raw["vectors_high"][sample_idx]
x_low  = x_raw["vectors_low"][sample_idx]
x_mid  = x_raw["vectors_mid"][sample_idx]
x_high = x_raw["vectors_high"][sample_idx]

# ── RSM computation ───────────────────────────────────────────────────────────
def compute_rsm_acoustic(features):
    """Negative Euclidean distance for acoustic features."""
    dist = cdist(features, features, metric="euclidean")
    return -dist

def compute_rsm_neural(features):
    """Cosine similarity for neural features."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features_norm = features / norms
    return features_norm @ features_norm.T

def upper_triangle(matrix):
    """Extract upper triangle (excluding diagonal)."""
    idx = np.triu_indices(len(matrix), k=1)
    return matrix[idx]

print("\nComputing RSMs...")
rsm_acoustic = compute_rsm_acoustic(acoustic_sub)
rsm_w_low    = compute_rsm_neural(w_low)
rsm_w_high   = compute_rsm_neural(w_high)
rsm_x_low    = compute_rsm_neural(x_low)
rsm_x_mid    = compute_rsm_neural(x_mid)
rsm_x_high   = compute_rsm_neural(x_high)
print("RSMs computed!")

# ── Mantel test ───────────────────────────────────────────────────────────────
print("\nRunning Mantel tests...")

def mantel_test(rsm1, rsm2, n_permutations=1000, rng=None):
    """Mantel test: Spearman rank correlation between upper triangles."""
    if rng is None:
        rng = np.random.default_rng(42)
    v1 = upper_triangle(rsm1)
    v2 = upper_triangle(rsm2)
    observed_r, _ = spearmanr(v1, v2)
    # Permutation test
    perm_r = []
    idx = np.arange(len(rsm1))
    for _ in range(n_permutations):
        perm = rng.permutation(idx)
        rsm1_perm = rsm1[perm][:, perm]
        v1_perm = upper_triangle(rsm1_perm)
        r, _ = spearmanr(v1_perm, v2)
        perm_r.append(r)
    p_value = np.mean(np.abs(perm_r) >= np.abs(observed_r))
    return observed_r, p_value

rng_mantel = np.random.default_rng(42)

comparisons = [
    ("Acoustic", "Whisper L4",  rsm_acoustic, rsm_w_low),
    ("Acoustic", "Whisper L20", rsm_acoustic, rsm_w_high),
    ("Acoustic", "XLS-R L4",   rsm_acoustic, rsm_x_low),
    ("Acoustic", "XLS-R L12",  rsm_acoustic, rsm_x_mid),
    ("Acoustic", "XLS-R L20",  rsm_acoustic, rsm_x_high),
    ("Whisper L4",  "Whisper L20", rsm_w_low,  rsm_w_high),
    ("XLS-R L4",    "XLS-R L12",  rsm_x_low,  rsm_x_mid),
    ("XLS-R L12",   "XLS-R L20",  rsm_x_mid,  rsm_x_high),
    ("Whisper L20", "XLS-R L12",  rsm_w_high, rsm_x_mid),
]

mantel_rows = []
for name1, name2, rsm1, rsm2 in comparisons:
    r, p = mantel_test(rsm1, rsm2, rng=rng_mantel)
    mantel_rows.append({
        "RSM1": name1, "RSM2": name2,
        "r":    round(r, 4),
        "p":    round(p, 4),
        "sig":  "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    })
    print(f"   {name1:12s} vs {name2:12s}  r={r:.4f}  p={p:.4f}")

mantel_df = pd.DataFrame(mantel_rows)
mantel_df.to_csv(TAB_DIR / "mantel_results.csv", index=False)
print(f"   Saved → results/tables/mantel_results.csv")

# ── Figure 6: RSM heatmaps ────────────────────────────────────────────────────
print("\nFigure 6: RSM heatmaps...")

# Sort tokens by phoneme for clearer visualisation
phonemes = meta_df["phoneme"].values[sample_idx]
sort_idx  = np.argsort(phonemes)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
rsm_configs = [
    (rsm_acoustic[sort_idx][:, sort_idx], "Acoustic (−Euclidean)"),
    (rsm_w_low[sort_idx][:, sort_idx],    "Whisper L4 (cosine)"),
    (rsm_w_high[sort_idx][:, sort_idx],   "Whisper L20 (cosine)"),
    (rsm_x_low[sort_idx][:, sort_idx],    "XLS-R L4 (cosine)"),
    (rsm_x_mid[sort_idx][:, sort_idx],    "XLS-R L12 (cosine)"),
    (rsm_x_high[sort_idx][:, sort_idx],   "XLS-R L20 (cosine)"),
]

for ax, (rsm, title) in zip(axes.flat, rsm_configs):
    im = ax.imshow(rsm, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Token index (sorted by phoneme)", fontsize=9)
    ax.set_ylabel("Token index (sorted by phoneme)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Figure 6: Representational Similarity Matrices", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_rsm.png", dpi=150)
plt.close()
print(f"   Saved → results/figures/fig6_rsm.png")

print("\n✅ §5.3 RSM analysis complete!")