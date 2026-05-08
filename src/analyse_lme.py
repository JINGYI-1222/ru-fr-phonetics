"""
analyse_lme.py
§7 Linear Mixed-Effects Models
7.1 Acoustic features (F1, F2)
7.2 Neural representations (PCA-projected, d=5, standardised)
7.3 Model building: null → main → full → extended → random slope
7.4 Compare variance explained (marginal/conditional R2) across representations
"""
 
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from scipy.stats import chi2
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
# ── Output directories ────────────────────────────────────────────────────────
TAB_DIR = Path("results/tables")
TAB_DIR.mkdir(parents=True, exist_ok=True)
 
# ── French oral vowels ────────────────────────────────────────────────────────
VOWELS = {"i", "e", "ɛ", "a", "ɑ", "o", "ɔ", "u", "y", "ø", "œ", "ə"}
 
# ── Vowel height categories ───────────────────────────────────────────────────
VOWEL_HEIGHT = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",
    "ɛ": "mid",  "œ": "mid",  "ɔ": "mid",
    "a": "low",  "ɑ": "low",  "ə": "mid"
}
 
# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df     = pd.read_csv("data/processed/features_acoustic_norm.csv")
vowels = df[df["phoneme"].isin(VOWELS)].copy()
 
vowels["L2"]   = (vowels["l1_status"] == "L2").astype(int)
vowels["Male"] = (vowels["gender"] == "m").astype(int)
vowels["vowel_height"] = vowels["phoneme"].map(VOWEL_HEIGHT)
 
w_raw     = np.load("data/processed/features_whisper.npz", allow_pickle=True)
x_raw     = np.load("data/processed/features_xlsr.npz",   allow_pickle=True)
meta_keys = list(w_raw["meta_keys"])
meta_df   = pd.DataFrame(w_raw["meta"], columns=meta_keys)
meta_df["L2"]   = (meta_df["l1_status"] == "L2").astype(int)
meta_df["Male"] = (meta_df["gender"] == "m").astype(int)
meta_df["vowel_height"] = meta_df["phoneme"].map(VOWEL_HEIGHT)
 
print(f"Acoustic vowel tokens: {len(vowels)}")
print(f"Neural tokens: {len(meta_df)}")
 
# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_icc(model_null):
    """ICC = var_random / (var_random + var_residual)"""
    var_u   = float(model_null.cov_re.iloc[0, 0])
    var_eps = float(model_null.scale)
    icc     = var_u / (var_u + var_eps) if (var_u + var_eps) > 0 else 0
    return icc, var_u, var_eps
 
def marginal_r2(model_full, var_u_null, var_eps_null):
    """Nakagawa & Schielzeth (2013) marginal R2."""
    var_fixed = model_full.fittedvalues.var()
    try:
        var_u   = float(model_full.cov_re.iloc[0, 0])
        var_eps = float(model_full.scale)
    except Exception:
        var_u   = var_u_null
        var_eps = var_eps_null
    denom = var_fixed + var_u + var_eps
    r2m = var_fixed / denom if denom > 0 else 0
    r2c = (var_fixed + var_u) / denom if denom > 0 else 0
    return round(r2m, 4), round(r2c, 4)
 
def fit_model_sequence(data, response, group_col="speaker_id"):
    """Fit null → main → full → extended → random slope models."""
    results = {}
    data = data.copy().dropna(subset=[response, "L2", "Male",
                                       "vowel_height", group_col])
    if len(data) < 30:
        print(f"   Skipping {response}: too few observations ({len(data)})")
        return results
 
    var_u_null   = 0
    var_eps_null = 0
 
    # 1. Null model
    try:
        md0 = MixedLM.from_formula(f"{response} ~ 1", data, groups=data[group_col])
        mf0 = md0.fit(reml=False, method="powell")
        icc, var_u_null, var_eps_null = compute_icc(mf0)
        results["null"] = {
            "aic": round(mf0.aic, 2), "bic": round(mf0.bic, 2),
            "loglik": round(mf0.llf, 4),
            "icc": round(icc, 4),
            "var_u": round(var_u_null, 4),
            "var_eps": round(var_eps_null, 4)
        }
        print(f"   Null:      AIC={mf0.aic:.1f}  ICC={icc:.4f}")
    except Exception as e:
        print(f"   Null model failed: {e}")
        return results
 
    # 2. Main effects model
    try:
        md1 = MixedLM.from_formula(f"{response} ~ L2 + Male", data, groups=data[group_col])
        mf1 = md1.fit(reml=False, method="powell")
        lrt1 = 2 * (mf1.llf - mf0.llf)
        p1   = 1 - chi2.cdf(lrt1, df=2)
        r2m, r2c = marginal_r2(mf1, var_u_null, var_eps_null)
        results["main"] = {
            "aic": round(mf1.aic, 2), "bic": round(mf1.bic, 2),
            "loglik": round(mf1.llf, 4),
            "lrt_vs_null": round(lrt1, 3),
            "p_lrt": round(p1, 6),
            "beta_L2":   round(float(mf1.params.get("L2",   np.nan)), 4),
            "beta_Male": round(float(mf1.params.get("Male", np.nan)), 4),
            "p_L2":   round(float(mf1.pvalues.get("L2",   np.nan)), 6),
            "p_Male": round(float(mf1.pvalues.get("Male", np.nan)), 6),
            "r2m": r2m, "r2c": r2c
        }
        print(f"   Main:      AIC={mf1.aic:.1f}  R2m={r2m:.4f}  β_L2={mf1.params.get('L2',np.nan):.4f}  p_L2={mf1.pvalues.get('L2',np.nan):.4f}")
    except Exception as e:
        print(f"   Main model failed: {e}")
 
    # 3. Full model (L2 x Male interaction)
    try:
        md2 = MixedLM.from_formula(f"{response} ~ L2 * Male", data, groups=data[group_col])
        mf2 = md2.fit(reml=False, method="powell")
        lrt2 = 2 * (mf2.llf - mf1.llf) if "main" in results else np.nan
        p2   = 1 - chi2.cdf(lrt2, df=1) if not np.isnan(lrt2) else np.nan
        r2m, r2c = marginal_r2(mf2, var_u_null, var_eps_null)
        results["full"] = {
            "aic": round(mf2.aic, 2), "bic": round(mf2.bic, 2),
            "loglik": round(mf2.llf, 4),
            "lrt_vs_main": round(lrt2, 3),
            "p_lrt": round(p2, 6),
            "beta_L2xMale": round(float(mf2.params.get("L2:Male", np.nan)), 4),
            "p_L2xMale":   round(float(mf2.pvalues.get("L2:Male", np.nan)), 6),
            "r2m": r2m, "r2c": r2c
        }
        print(f"   Full:      AIC={mf2.aic:.1f}  R2m={r2m:.4f}  β_L2xMale={mf2.params.get('L2:Male',np.nan):.4f}  p={p2:.4f}")
    except Exception as e:
        print(f"   Full model failed: {e}")
 
    # 4. Extended model (+ vowel height)
    try:
        md3 = MixedLM.from_formula(
            f"{response} ~ L2 * Male + C(vowel_height)", data, groups=data[group_col])
        mf3 = md3.fit(reml=False, method="powell")
        lrt3 = 2 * (mf3.llf - mf2.llf) if "full" in results else np.nan
        r2m, r2c = marginal_r2(mf3, var_u_null, var_eps_null)
        results["extended"] = {
            "aic": round(mf3.aic, 2), "bic": round(mf3.bic, 2),
            "loglik": round(mf3.llf, 4),
            "lrt_vs_full": round(lrt3, 3),
            "r2m": r2m, "r2c": r2c
        }
        print(f"   Extended:  AIC={mf3.aic:.1f}  R2m={r2m:.4f}")
    except Exception as e:
        print(f"   Extended model failed: {e}")
 
    # 5. Random slope model
    try:
        md4 = MixedLM.from_formula(
            f"{response} ~ L2 * Male", data,
            groups=data[group_col], re_formula="~L2")
        mf4 = md4.fit(reml=False, method="powell")
        lrt4 = 2 * (mf4.llf - mf2.llf) if "full" in results else np.nan
        p4   = 1 - chi2.cdf(lrt4, df=2) if not np.isnan(lrt4) else np.nan
        results["random_slope"] = {
            "aic": round(mf4.aic, 2), "bic": round(mf4.bic, 2),
            "loglik": round(mf4.llf, 4),
            "lrt_vs_full": round(lrt4, 3),
            "p_lrt": round(p4, 6),
        }
        print(f"   RandSlope: AIC={mf4.aic:.1f}  LRT={lrt4:.3f}  p={p4:.4f}")
    except Exception as e:
        print(f"   Random slope failed: {e}")
 
    return results
 
# ─────────────────────────────────────────────────────────────────────────────
# §7.1 ACOUSTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§7.1 Acoustic LME models")
print("="*60)
 
acoustic_results = {}
for formant in ["F1_lob", "F2_lob"]:
    print(f"\n── {formant} ──────────────────────────────")
    results = fit_model_sequence(vowels, formant)
    acoustic_results[formant] = results
 
rows = []
for formant, res in acoustic_results.items():
    for model_name, vals in res.items():
        row = {"formant": formant, "model": model_name}
        row.update(vals)
        rows.append(row)
pd.DataFrame(rows).to_csv(TAB_DIR / "lme_acoustic_results.csv", index=False)
print(f"\n   Saved → results/tables/lme_acoustic_results.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# §7.2 NEURAL LME MODELS (PCA d=5, standardised)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§7.2 Neural LME models (PCA d=5, standardised)")
print("="*60)
 
neural_configs = [
    (w_raw["vectors_low"],  "Whisper_L4"),
    (w_raw["vectors_high"], "Whisper_L20"),
    (x_raw["vectors_low"],  "XLSR_L4"),
    (x_raw["vectors_mid"],  "XLSR_L12"),
    (x_raw["vectors_high"], "XLSR_L20"),
]
 
neural_results = {}
 
for vectors, model_name in neural_configs:
    print(f"\n── {model_name} ──────────────────────────────")
 
    # PCA d=5 + standardise PC scores
    scaler1  = StandardScaler()
    X_scaled = scaler1.fit_transform(vectors)
    pca      = PCA(n_components=5, random_state=42)
    X_pca    = pca.fit_transform(X_scaled)
    X_pca    = StandardScaler().fit_transform(X_pca)  # standardise PC scores
    var_expl = pca.explained_variance_ratio_.sum()
    print(f"   Variance explained by 5 PCs: {var_expl*100:.1f}%")
 
    model_results = {}
    for pc_idx in range(5):
        pc_col = f"PC{pc_idx+1}"
        neural_data = meta_df.copy()
        neural_data[pc_col] = X_pca[:, pc_idx]
        print(f"\n   {pc_col}:")
        res = fit_model_sequence(neural_data, pc_col)
        model_results[pc_col] = res
 
    neural_results[model_name] = model_results
 
rows = []
for model_name, pcs in neural_results.items():
    for pc, res in pcs.items():
        for model_step, vals in res.items():
            row = {"model": model_name, "pc": pc, "step": model_step}
            row.update(vals)
            rows.append(row)
pd.DataFrame(rows).to_csv(TAB_DIR / "lme_neural_results.csv", index=False)
print(f"\n   Saved → results/tables/lme_neural_results.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# §7.3 ICC COMPARISON: F1 of /a/ vs Whisper PC1 of /a/
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§7.3 ICC comparison: /a/ acoustic vs Whisper PC1")
print("="*60)
 
# Acoustic ICC for /a/ F1
a_data = vowels[vowels["phoneme"] == "a"].copy()
try:
    md_a = MixedLM.from_formula("F1_lob ~ 1", a_data, groups=a_data["speaker_id"])
    mf_a = md_a.fit(reml=False, method="powell")
    icc_a, var_u_a, var_eps_a = compute_icc(mf_a)
    print(f"   /a/ F1_lob ICC = {icc_a:.4f}  (var_u={var_u_a:.4f}, var_eps={var_eps_a:.4f})")
except Exception as e:
    print(f"   /a/ F1 ICC failed: {e}")
 
# Whisper PC1 ICC for /a/
a_mask   = meta_df["phoneme"] == "a"
a_neural = meta_df[a_mask].copy()
w_low_a  = w_raw["vectors_low"][a_mask]
scaler_a = StandardScaler()
X_a      = scaler_a.fit_transform(w_low_a)
pca_a    = PCA(n_components=5, random_state=42)
X_a_pca  = pca_a.fit_transform(X_a)
X_a_pca  = StandardScaler().fit_transform(X_a_pca)
a_neural["PC1"] = X_a_pca[:, 0]
 
try:
    md_wh = MixedLM.from_formula("PC1 ~ 1", a_neural, groups=a_neural["speaker_id"])
    mf_wh = md_wh.fit(reml=False, method="powell")
    icc_wh, var_u_wh, var_eps_wh = compute_icc(mf_wh)
    print(f"   /a/ Whisper L4 PC1 ICC = {icc_wh:.4f}  (var_u={var_u_wh:.4f}, var_eps={var_eps_wh:.4f})")
except Exception as e:
    print(f"   Whisper ICC failed: {e}")
 
# ─────────────────────────────────────────────────────────────────────────────
# §7.4 COMPARE MARGINAL R2 ACROSS REPRESENTATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("§7.4 Marginal R2 comparison across representations")
print("="*60)
 
r2_rows = []
 
# Acoustic
for formant in ["F1_lob", "F2_lob"]:
    res = acoustic_results.get(formant, {})
    if "main" in res:
        r2_rows.append({
            "representation": f"Acoustic_{formant}",
            "r2m_main": res["main"].get("r2m", np.nan),
            "r2c_main": res["main"].get("r2c", np.nan),
            "r2m_full": res.get("full", {}).get("r2m", np.nan),
            "r2c_full": res.get("full", {}).get("r2c", np.nan),
            "beta_L2":  res["main"].get("beta_L2", np.nan),
            "p_L2":     res["main"].get("p_L2", np.nan),
            "icc":      res.get("null", {}).get("icc", np.nan)
        })
 
# Neural PC1 only
for model_name, pcs in neural_results.items():
    res = pcs.get("PC1", {})
    if "main" in res:
        r2_rows.append({
            "representation": f"{model_name}_PC1",
            "r2m_main": res["main"].get("r2m", np.nan),
            "r2c_main": res["main"].get("r2c", np.nan),
            "r2m_full": res.get("full", {}).get("r2m", np.nan),
            "r2c_full": res.get("full", {}).get("r2c", np.nan),
            "beta_L2":  res["main"].get("beta_L2", np.nan),
            "p_L2":     res["main"].get("p_L2", np.nan),
            "icc":      res.get("null", {}).get("icc", np.nan)
        })
 
r2_df = pd.DataFrame(r2_rows)
r2_df.to_csv(TAB_DIR / "lme_r2_comparison.csv", index=False)
print(r2_df.to_string(index=False))
print(f"\n   Saved → results/tables/lme_r2_comparison.csv")
 
print("\n✅ §7 LME analysis complete!")