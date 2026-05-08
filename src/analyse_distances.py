"""
analyse_distances.py
§6.2 Inter-phoneme Distances
- Acoustic distance matrix (Euclidean + Mahalanobis)
- Neural distance matrices (cosine)
- Mantel test between distance structures
- Bootstrap CI on selected phoneme pairs
- Nearest-centroid classifier (LOSO CV) with per-class F1
- McNemar test L1 vs L2
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
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
 
vowel_mask  = meta_df["phoneme"].isin(VOWELS)
meta_vowels = meta_df[vowel_mask].reset_index(drop=True)
 
w_low  = w_raw["vectors_low"][vowel_mask]
w_high = w_raw["vectors_high"][vowel_mask]
x_low  = x_raw["vectors_low"][vowel_mask]
x_mid  = x_raw["vectors_mid"][vowel_mask]
x_high = x_raw["vectors_high"][vowel_mask]
 
min_tokens = 10
valid_phonemes = [p for p in VOWELS
                  if (vowels["phoneme"] == p).sum() >= min_tokens
                  and (meta_vowels["phoneme"] == p).sum() >= min_tokens]
print(f"Valid phonemes: {valid_phonemes}")
 
# ─────────────────────────────────────────────────────────────────────────────
# §6.2.1 ACOUSTIC DISTANCE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§6.2.1 Acoustic distance matrix")
print("="*60)
 
ac_centroids = {}
for p in valid_phonemes:
    sub = vowels[vowels["phoneme"] == p][["F1_lob", "F2_lob"]].dropna()
    ac_centroids[p] = sub.values.mean(axis=0)
 
n = len(valid_phonemes)
 
D_euclidean = np.zeros((n, n))
for i, p in enumerate(valid_phonemes):
    for j, q in enumerate(valid_phonemes):
        D_euclidean[i, j] = np.linalg.norm(ac_centroids[p] - ac_centroids[q])
 
all_vecs = np.concatenate([
    vowels[vowels["phoneme"] == p][["F1_lob", "F2_lob"]].dropna().values
    for p in valid_phonemes
])
pooled_cov     = np.cov(all_vecs.T) + np.eye(2) * 1e-6
pooled_cov_inv = np.linalg.inv(pooled_cov)
 
D_mahal = np.zeros((n, n))
for i, p in enumerate(valid_phonemes):
    for j, q in enumerate(valid_phonemes):
        diff = ac_centroids[p] - ac_centroids[q]
        D_mahal[i, j] = np.sqrt(diff @ pooled_cov_inv @ diff)
 
pd.DataFrame(D_euclidean, index=valid_phonemes, columns=valid_phonemes).to_csv(
    TAB_DIR / "distance_acoustic_euclidean.csv")
pd.DataFrame(D_mahal, index=valid_phonemes, columns=valid_phonemes).to_csv(
    TAB_DIR / "distance_acoustic_mahal.csv")
print(f"   Saved acoustic distance matrices")
 
# ─────────────────────────────────────────────────────────────────────────────
# §6.2.2 NEURAL DISTANCE MATRICES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§6.2.2 Neural distance matrices")
print("="*60)
 
def cosine_dist_matrix(vectors, meta, phonemes):
    centroids = {}
    for p in phonemes:
        mask = meta["phoneme"] == p
        vecs = vectors[mask]
        if len(vecs) == 0:
            continue
        c = vecs.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-10)
        centroids[p] = c
    n = len(phonemes)
    D = np.zeros((n, n))
    for i, p in enumerate(phonemes):
        for j, q in enumerate(phonemes):
            if p in centroids and q in centroids:
                D[i, j] = 1 - np.dot(centroids[p], centroids[q])
    return D, centroids
 
neural_configs = [
    (w_low,  "Whisper_L4"),
    (w_high, "Whisper_L20"),
    (x_low,  "XLSR_L4"),
    (x_mid,  "XLSR_L12"),
    (x_high, "XLSR_L20"),
]
 
neural_distances = {}
neural_centroids = {}
for vectors, name in neural_configs:
    D, centroids = cosine_dist_matrix(vectors, meta_vowels, valid_phonemes)
    neural_distances[name] = D
    neural_centroids[name] = centroids
    pd.DataFrame(D, index=valid_phonemes, columns=valid_phonemes).to_csv(
        TAB_DIR / f"distance_{name}.csv")
    print(f"   Saved {name} distance matrix")
 
# ─────────────────────────────────────────────────────────────────────────────
# §6.2.3 MANTEL TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§6.2.3 Mantel test between distance structures")
print("="*60)
 
def upper_tri(matrix):
    idx = np.triu_indices(len(matrix), k=1)
    return matrix[idx]
 
def mantel_test(D1, D2, n_perm=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    v1 = upper_tri(D1)
    v2 = upper_tri(D2)
    r_obs, _ = spearmanr(v1, v2)
    perm_r = []
    idx = np.arange(len(D1))
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        D1_perm = D1[perm][:, perm]
        r, _ = spearmanr(upper_tri(D1_perm), v2)
        perm_r.append(r)
    p = np.mean(np.abs(perm_r) >= np.abs(r_obs))
    return round(r_obs, 4), round(p, 4)
 
rng = np.random.default_rng(42)
mantel_rows = []
 
for name, D_neural in neural_distances.items():
    for ac_name, D_ac in [("Euclidean", D_euclidean), ("Mahalanobis", D_mahal)]:
        r, p = mantel_test(D_ac, D_neural, rng=rng)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        mantel_rows.append({"RSM1": ac_name, "RSM2": name, "r": r, "p": p, "sig": sig})
        print(f"   {ac_name:12s} vs {name:12s}  r={r:.4f}  p={p:.4f}  {sig}")
 
pairs = [
    ("Whisper_L4", "Whisper_L20"),
    ("XLSR_L4",    "XLSR_L12"),
    ("XLSR_L12",   "XLSR_L20"),
    ("Whisper_L20","XLSR_L12"),
]
for n1, n2 in pairs:
    r, p = mantel_test(neural_distances[n1], neural_distances[n2], rng=rng)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    mantel_rows.append({"RSM1": n1, "RSM2": n2, "r": r, "p": p, "sig": sig})
    print(f"   {n1:12s} vs {n2:12s}  r={r:.4f}  p={p:.4f}  {sig}")
 
mantel_df = pd.DataFrame(mantel_rows)
mantel_df.to_csv(TAB_DIR / "mantel_phoneme_distances.csv", index=False)
print(f"   Saved → results/tables/mantel_phoneme_distances.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# §6.2.4 BOOTSTRAP CI
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§6.2.4 Bootstrap CI on selected phoneme pairs")
print("="*60)
 
PAIRS = [("e", "ɛ"), ("o", "ɔ"), ("y", "u")]
B     = 2000
rng_b = np.random.default_rng(42)
 
boot_rows = []
for p1, p2 in PAIRS:
    if p1 not in valid_phonemes or p2 not in valid_phonemes:
        print(f"   Skipping {p1}-{p2}: insufficient tokens")
        continue
 
    speakers = vowels["speaker_id"].unique()
    ac_dists = []
    for _ in range(B):
        boot_spk = rng_b.choice(speakers, size=len(speakers), replace=True)
        boot_df  = pd.concat([vowels[vowels["speaker_id"] == s] for s in boot_spk])
        c1 = boot_df[boot_df["phoneme"] == p1][["F1_lob","F2_lob"]].dropna().values.mean(axis=0)
        c2 = boot_df[boot_df["phoneme"] == p2][["F1_lob","F2_lob"]].dropna().values.mean(axis=0)
        ac_dists.append(np.linalg.norm(c1 - c2))
    ac_ci = np.percentile(ac_dists, [2.5, 97.5])
    boot_rows.append({
        "pair": f"/{p1}/–/{p2}/", "rep": "Acoustic (Euclidean)",
        "mean_dist": round(np.mean(ac_dists), 4),
        "ci_low": round(ac_ci[0], 4), "ci_high": round(ac_ci[1], 4)
    })
    print(f"   /{p1}/–/{p2}/ Acoustic: {np.mean(ac_dists):.4f} [{ac_ci[0]:.4f}, {ac_ci[1]:.4f}]")
 
    speakers_n = meta_vowels["speaker_id"].unique()
    for vectors, name in neural_configs:
        neural_dists = []
        for _ in range(B):
            boot_spk = rng_b.choice(speakers_n, size=len(speakers_n), replace=True)
            idx_list = [np.where(meta_vowels["speaker_id"] == s)[0] for s in boot_spk]
            boot_idx  = np.concatenate(idx_list)
            boot_meta = meta_vowels.iloc[boot_idx]
            boot_vecs = vectors[boot_idx]
            m1 = boot_meta["phoneme"] == p1
            m2 = boot_meta["phoneme"] == p2
            if m1.sum() < 2 or m2.sum() < 2:
                continue
            c1 = boot_vecs[m1].mean(axis=0)
            c2 = boot_vecs[m2].mean(axis=0)
            c1 /= np.linalg.norm(c1) + 1e-10
            c2 /= np.linalg.norm(c2) + 1e-10
            neural_dists.append(1 - np.dot(c1, c2))
        if neural_dists:
            ci = np.percentile(neural_dists, [2.5, 97.5])
            boot_rows.append({
                "pair": f"/{p1}/–/{p2}/", "rep": name,
                "mean_dist": round(np.mean(neural_dists), 4),
                "ci_low": round(ci[0], 4), "ci_high": round(ci[1], 4)
            })
            print(f"   /{p1}/–/{p2}/ {name}: {np.mean(neural_dists):.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
 
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(TAB_DIR / "bootstrap_ci_phoneme_pairs.csv", index=False)
print(f"   Saved → results/tables/bootstrap_ci_phoneme_pairs.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# §6.2.5 NEAREST-CENTROID CLASSIFIER (LOSO CV)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§6.2.5 Nearest-centroid classifier (LOSO CV)")
print("="*60)
 
def nearest_centroid_loso(features, labels, speakers):
    y_true, y_pred = [], []
    for test_spk in tqdm(np.unique(speakers), desc="  LOSO", leave=False):
        train_mask = speakers != test_spk
        test_mask  = speakers == test_spk
        train_f, train_l = features[train_mask], labels[train_mask]
        test_f,  test_l  = features[test_mask],  labels[test_mask]
        centroids = {}
        for lbl in np.unique(train_l):
            c = train_f[train_l == lbl].mean(axis=0)
            centroids[lbl] = c / (np.linalg.norm(c) + 1e-10)
        for i in range(len(test_f)):
            feat = test_f[i] / (np.linalg.norm(test_f[i]) + 1e-10)
            best_lbl = max(centroids, key=lambda lbl: np.dot(feat, centroids[lbl]))
            y_true.append(test_l[i])
            y_pred.append(best_lbl)
    return np.array(y_true), np.array(y_pred)
 
def nearest_centroid_loso_acoustic(features, labels, speakers):
    y_true, y_pred = [], []
    for test_spk in tqdm(np.unique(speakers), desc="  LOSO", leave=False):
        train_mask = speakers != test_spk
        test_mask  = speakers == test_spk
        train_f, train_l = features[train_mask], labels[train_mask]
        test_f,  test_l  = features[test_mask],  labels[test_mask]
        centroids = {lbl: train_f[train_l == lbl].mean(axis=0)
                     for lbl in np.unique(train_l)}
        for i in range(len(test_f)):
            dists = {lbl: np.linalg.norm(test_f[i] - c) for lbl, c in centroids.items()}
            y_true.append(test_l[i])
            y_pred.append(min(dists, key=dists.get))
    return np.array(y_true), np.array(y_pred)
 
ac_feats  = vowels[["F1_lob","F2_lob"]].fillna(0).values
ac_labels = vowels["phoneme"].values
ac_spk    = vowels["speaker_id"].values
ne_labels = meta_vowels["phoneme"].values
ne_spk    = meta_vowels["speaker_id"].values
 
clf_rows     = []
clf_per_rows = []
 
# Acoustic
print("\n   Acoustic classifier...")
y_true, y_pred = nearest_centroid_loso_acoustic(ac_feats, ac_labels, ac_spk)
acc      = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
f1_per   = f1_score(y_true, y_pred, average=None, labels=valid_phonemes, zero_division=0)
row = {"rep": "Acoustic", "accuracy": round(acc,4), "macro_f1": round(f1_macro,4)}
for p, f in zip(valid_phonemes, f1_per):
    row[f"f1_{p}"] = round(f, 4)
clf_rows.append(row)
print(f"   Acoustic: acc={acc:.4f}  macro_F1={f1_macro:.4f}")
print(f"   Per-class F1: {dict(zip(valid_phonemes, f1_per.round(3)))}")
 
# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=valid_phonemes)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=valid_phonemes,
            yticklabels=valid_phonemes, cmap="Blues", ax=ax)
ax.set_title("Figure 8a: Confusion Matrix — Acoustic (LOSO)", fontsize=12)
ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("True", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig8a_confusion_acoustic.png", dpi=150)
plt.close()
 
# Neural
for vectors, name in neural_configs:
    print(f"\n   {name}...")
    y_true, y_pred = nearest_centroid_loso(vectors, ne_labels, ne_spk)
    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per   = f1_score(y_true, y_pred, average=None, labels=valid_phonemes, zero_division=0)
    row = {"rep": name, "accuracy": round(acc,4), "macro_f1": round(f1_macro,4)}
    for p, f in zip(valid_phonemes, f1_per):
        row[f"f1_{p}"] = round(f, 4)
    clf_rows.append(row)
    print(f"   {name}: acc={acc:.4f}  macro_F1={f1_macro:.4f}")
    print(f"   Per-class F1: {dict(zip(valid_phonemes, f1_per.round(3)))}")
 
    cm = confusion_matrix(y_true, y_pred, labels=valid_phonemes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=valid_phonemes,
                yticklabels=valid_phonemes, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {name} (LOSO)", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"fig8_confusion_{name}.png", dpi=150)
    plt.close()
 
clf_df = pd.DataFrame(clf_rows)
clf_df.to_csv(TAB_DIR / "classifier_results.csv", index=False)
print(f"\n   Saved → results/tables/classifier_results.csv")
print(clf_df[["rep","accuracy","macro_f1"]].to_string(index=False))
 
# ─────────────────────────────────────────────────────────────────────────────
# §6.2.6 McNemar test L1 vs L2
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§6.2.6 McNemar test: L1 vs L2 classification accuracy")
print("="*60)
 
def get_correctness(features, labels, speakers, l1_status, is_acoustic=False):
    correct_l1, correct_l2 = [], []
    for test_spk in np.unique(speakers):
        train_mask = speakers != test_spk
        test_mask  = speakers == test_spk
        train_f, train_l = features[train_mask], labels[train_mask]
        test_f,  test_l  = features[test_mask],  labels[test_mask]
        test_l1 = l1_status[test_mask]
        if is_acoustic:
            centroids = {lbl: train_f[train_l == lbl].mean(axis=0)
                         for lbl in np.unique(train_l)}
            for i in range(len(test_f)):
                dists = {lbl: np.linalg.norm(test_f[i] - c) for lbl, c in centroids.items()}
                pred  = min(dists, key=dists.get)
                correct = int(pred == test_l[i])
                (correct_l1 if test_l1[i] == "L1" else correct_l2).append(correct)
        else:
            centroids = {}
            for lbl in np.unique(train_l):
                c = train_f[train_l == lbl].mean(axis=0)
                centroids[lbl] = c / (np.linalg.norm(c) + 1e-10)
            for i in range(len(test_f)):
                feat = test_f[i] / (np.linalg.norm(test_f[i]) + 1e-10)
                pred = max(centroids, key=lambda lbl: np.dot(feat, centroids[lbl]))
                correct = int(pred == test_l[i])
                (correct_l1 if test_l1[i] == "L1" else correct_l2).append(correct)
    return np.array(correct_l1), np.array(correct_l2)
 
ac_l1_status = vowels["l1_status"].values
ne_l1_status = meta_vowels["l1_status"].values
 
ac_c_l1, ac_c_l2 = get_correctness(ac_feats, ac_labels, ac_spk,
                                     ac_l1_status, is_acoustic=True)
print(f"\n   Acoustic: L1 acc={ac_c_l1.mean():.4f}  L2 acc={ac_c_l2.mean():.4f}")
 
mcnemar_rows = []
for vectors, name in tqdm(neural_configs, desc="McNemar tests"):
    ne_c_l1, ne_c_l2 = get_correctness(vectors, ne_labels, ne_spk,
                                         ne_l1_status, is_acoustic=False)
    n = min(len(ac_c_l1), len(ne_c_l1))
    table_l1 = np.array([
        [(ac_c_l1[:n] == 1) & (ne_c_l1[:n] == 1),
         (ac_c_l1[:n] == 1) & (ne_c_l1[:n] == 0)],
        [(ac_c_l1[:n] == 0) & (ne_c_l1[:n] == 1),
         (ac_c_l1[:n] == 0) & (ne_c_l1[:n] == 0)]
    ]).sum(axis=2)
    result_l1 = mcnemar(table_l1, exact=False)
 
    n = min(len(ac_c_l2), len(ne_c_l2))
    table_l2 = np.array([
        [(ac_c_l2[:n] == 1) & (ne_c_l2[:n] == 1),
         (ac_c_l2[:n] == 1) & (ne_c_l2[:n] == 0)],
        [(ac_c_l2[:n] == 0) & (ne_c_l2[:n] == 1),
         (ac_c_l2[:n] == 0) & (ne_c_l2[:n] == 0)]
    ]).sum(axis=2)
    result_l2 = mcnemar(table_l2, exact=False)
 
    mcnemar_rows.append({
        "neural_rep":   name,
        "L1_acc":       round(ne_c_l1.mean(), 4),
        "L2_acc":       round(ne_c_l2.mean(), 4),
        "mcnemar_p_L1": round(result_l1.pvalue, 4),
        "mcnemar_p_L2": round(result_l2.pvalue, 4),
    })
    print(f"   {name}: L1={ne_c_l1.mean():.4f} L2={ne_c_l2.mean():.4f} "
          f"McNemar_L1 p={result_l1.pvalue:.4f} McNemar_L2 p={result_l2.pvalue:.4f}")
 
mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv(TAB_DIR / "mcnemar_results.csv", index=False)
print(f"   Saved → results/tables/mcnemar_results.csv")
 
print("\n✅ §6.2 Inter-phoneme distances complete!")