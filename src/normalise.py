"""
Stage 5: normalise
1. Lobanov normalisation of F1/F2 (per speaker, vowels only)
2. PCA reduction of Whisper and XLS-R representations
3. UMAP reduction of Whisper and XLS-R representations
Reads parameters from params.yaml.
Outputs:
  - data/processed/features_acoustic_norm.csv
  - data/processed/features_whisper_pca.npz
  - data/processed/features_xlsr_pca.npz
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# ── Load parameters ───────────────────────────────────────────────────────────
with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

PCA_VIS     = params["normalise"]["pca_components_visual"]
PCA_CLUSTER = params["normalise"]["pca_components_cluster"]
UMAP_N      = params["normalise"]["umap_n_components"]
UMAP_NN     = params["normalise"]["umap_n_neighbors"]
UMAP_DIST   = params["normalise"]["umap_min_dist"]

# ── Paths ─────────────────────────────────────────────────────────────────────
ACOUSTIC_CSV    = Path("data/processed/features_acoustic.csv")
WHISPER_NPZ     = Path("data/processed/features_whisper.npz")
XLSR_NPZ        = Path("data/processed/features_xlsr.npz")

OUT_ACOUSTIC    = Path("data/processed/features_acoustic_norm.csv")
OUT_WHISPER_PCA = Path("data/processed/features_whisper_pca.npz")
OUT_XLSR_PCA    = Path("data/processed/features_xlsr_pca.npz")

# ── French oral vowels ────────────────────────────────────────────────────────
VOWELS = {"i", "e", "ɛ", "a", "ɑ", "o", "ɔ", "u", "y", "ø", "œ", "ə",
          "ɛ̃", "ɑ̃", "ɔ̃", "œ̃"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOBANOV NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Lobanov normalisation")
print("=" * 60)

df = pd.read_csv(ACOUSTIC_CSV)

# Lobanov uses vowel tokens only for computing speaker means/SDs
vowel_mask = df["phoneme"].isin(VOWELS)

df["F1_lob"] = np.nan
df["F2_lob"] = np.nan

for spk, grp in df[vowel_mask].groupby("speaker_id"):
    for formant in ["F1", "F2"]:
        mu = grp[formant].mean()
        sd = grp[formant].std()
        if sd > 0:
            norm_col = f"{formant}_lob"
            df.loc[grp.index, norm_col] = (grp[formant] - mu) / sd

# Report
n_vowels   = vowel_mask.sum()
n_lob_ok   = df.loc[vowel_mask, "F1_lob"].notna().sum()
print(f"   Vowel tokens          : {n_vowels}")
print(f"   F1_lob computed       : {n_lob_ok} ({100*n_lob_ok/n_vowels:.1f}%)")
print(f"   Speakers normalised   : {df[vowel_mask]['speaker_id'].nunique()}")

df.to_csv(OUT_ACOUSTIC, index=False)
print(f"   Saved → {OUT_ACOUSTIC}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PCA ON WHISPER REPRESENTATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: PCA on Whisper representations")
print("=" * 60)

w = np.load(WHISPER_NPZ, allow_pickle=True)
meta_keys = list(w["meta_keys"])
meta_df   = pd.DataFrame(w["meta"], columns=meta_keys)

for layer_name, vectors in [("low", w["vectors_low"]),
                              ("high", w["vectors_high"])]:
    print(f"\n   Layer: {layer_name}  shape: {vectors.shape}")

    # Standardise before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(vectors)

    # PCA for visualisation (d=2)
    pca_vis = PCA(n_components=PCA_VIS, random_state=42)
    X_pca_vis = pca_vis.fit_transform(X_scaled)
    var_vis = pca_vis.explained_variance_ratio_.sum()
    print(f"   PCA {PCA_VIS}D  variance explained: {100*var_vis:.1f}%")

    # PCA for clustering (d=50)
    n_comp = min(PCA_CLUSTER, vectors.shape[1], vectors.shape[0])
    pca_cls = PCA(n_components=n_comp, random_state=42)
    X_pca_cls = pca_cls.fit_transform(X_scaled)
    var_cls = pca_cls.explained_variance_ratio_.sum()
    print(f"   PCA {n_comp}D  variance explained: {100*var_cls:.1f}%")

    # UMAP for visualisation (d=2)
    print(f"   Running UMAP...")
    reducer = umap.UMAP(n_components=UMAP_N,
                        n_neighbors=UMAP_NN,
                        min_dist=UMAP_DIST,
                        random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    print(f"   UMAP done. shape: {X_umap.shape}")

    np.savez_compressed(
        str(OUT_WHISPER_PCA).replace(".npz", f"_{layer_name}.npz"),
        pca_2d=X_pca_vis,
        pca_50d=X_pca_cls,
        umap_2d=X_umap,
        explained_variance_vis=pca_vis.explained_variance_ratio_,
        explained_variance_cls=pca_cls.explained_variance_ratio_
    )
    print(f"   Saved → {str(OUT_WHISPER_PCA).replace('.npz', f'_{layer_name}.npz')}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA ON XLS-R REPRESENTATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: PCA on XLS-R representations")
print("=" * 60)

x = np.load(XLSR_NPZ, allow_pickle=True)

for layer_name, vectors in [("low",  x["vectors_low"]),
                              ("mid",  x["vectors_mid"]),
                              ("high", x["vectors_high"])]:
    print(f"\n   Layer: {layer_name}  shape: {vectors.shape}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(vectors)

    pca_vis = PCA(n_components=PCA_VIS, random_state=42)
    X_pca_vis = pca_vis.fit_transform(X_scaled)
    var_vis = pca_vis.explained_variance_ratio_.sum()
    print(f"   PCA {PCA_VIS}D  variance explained: {100*var_vis:.1f}%")

    n_comp = min(PCA_CLUSTER, vectors.shape[1], vectors.shape[0])
    pca_cls = PCA(n_components=n_comp, random_state=42)
    X_pca_cls = pca_cls.fit_transform(X_scaled)
    var_cls = pca_cls.explained_variance_ratio_.sum()
    print(f"   PCA {n_comp}D  variance explained: {100*var_cls:.1f}%")

    print(f"   Running UMAP...")
    reducer = umap.UMAP(n_components=UMAP_N,
                        n_neighbors=UMAP_NN,
                        min_dist=UMAP_DIST,
                        random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    print(f"   UMAP done. shape: {X_umap.shape}")

    np.savez_compressed(
        str(OUT_XLSR_PCA).replace(".npz", f"_{layer_name}.npz"),
        pca_2d=X_pca_vis,
        pca_50d=X_pca_cls,
        umap_2d=X_umap,
        explained_variance_vis=pca_vis.explained_variance_ratio_,
        explained_variance_cls=pca_cls.explained_variance_ratio_
    )
    print(f"   Saved → {str(OUT_XLSR_PCA).replace('.npz', f'_{layer_name}.npz')}")

print("\n✅ Normalisation complete!")