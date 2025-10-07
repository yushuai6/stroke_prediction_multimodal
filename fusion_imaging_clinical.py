# Late-stacking (clinical + imaging):
# A) Clinical + primary imaging fused prob (weighted if available, else LR, else mean)
# B) Clinical + MR-only (DWI/ADC/FLAIR) fused prob
# C) Clinical + NCCT prob
# D) Clinical + individual imaging modalities (with SHAP & pretty labels)
#
# Features
# - Robust ID alignment (strict, relaxed, normalized) + reports
# - Config-driven thresholding (knee/youden/target_point/target_sensitivity/constrained_sens)
# - Baselines: Imaging-only & Clinical-only
# - Separate artifacts per fusion variant
# - SHAP plots for clinical+per-modality stacker (TEST/EXTERNAL, if available)

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

# ---------- Project imports ----------
import config as cfg  # expects fusion_outdir, clinical_xlsx, threshold knobs, etc.


# =========================
# Utilities
# =========================

def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _onehot())
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop", sparse_threshold=0.0)


def _coerce_mixed_types(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        s = Xc[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sn = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() and (sn.notna().sum() / max(s.notna().sum(), 1)) >= 0.7:
                Xc[c] = sn
            else:
                Xc[c] = s.astype(str)
    return Xc


def _normalize_id(s: pd.Series) -> pd.Series:
    z = s.astype(str).str.strip()
    z = z.str.replace(r"\s+", " ", regex=True)  # collapse
    z = z.str.replace(" ", "", regex=False)     # remove
    z = z.str.replace(r"[-_]", "", regex=True)  # strip - _
    z = z.str.upper()
    return z


def _write_thresh_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return acc, sens, spec


def _pick_threshold_knee(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    dist = np.sqrt((fpr - 0.0) ** 2 + (1.0 - tpr) ** 2)
    i = int(np.argmin(dist))
    return float(thr[i]), float(tpr[i]), float(1.0 - fpr[i])


def _pick_threshold_youden(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(1.0 - fpr[i])


def _pick_threshold_target_point(y_true, y_prob, target_tpr=0.8, target_fpr=0.25):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    dist = np.sqrt((fpr - float(target_fpr)) ** 2 + (tpr - float(target_tpr)) ** 2)
    i = int(np.argmin(dist))
    return float(thr[i]), float(tpr[i]), float(1.0 - fpr[i])


def _pick_threshold_for_sensitivity(y_true, y_prob, target_sens=0.70):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    idxs = np.where(tpr >= float(target_sens))[0]
    i = int(idxs[0]) if len(idxs) else int(np.argmax(tpr))
    return float(thr[i]), float(tpr[i]), float(1.0 - fpr[i])


def _pick_threshold_constrained_sens(y_true, y_prob, max_fpr=0.25):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    mask = fpr <= float(max_fpr) + 1e-12
    if np.any(mask):
        idxs = np.where(mask)[0]
        order = np.lexsort((fpr[idxs], -tpr[idxs]))
        best = idxs[order][0]
        return float(thr[best]), float(tpr[best]), float(1.0 - fpr[best])
    dif = np.abs(fpr - float(max_fpr))
    best = int(np.lexsort((dif, -tpr))[0])
    return float(thr[best]), float(tpr[best]), float(1.0 - fpr[best])


def _choose_threshold(y_true, y_prob, mode="knee", target_sens=0.75, target_tpr=0.8, target_fpr=0.35):
    mode = (mode or "knee").lower()
    if mode == "knee":
        return _pick_threshold_knee(y_true, y_prob)
    if mode == "youden":
        return _pick_threshold_youden(y_true, y_prob)
    if mode == "target_point":
        return _pick_threshold_target_point(y_true, y_prob, target_tpr, target_fpr)
    if mode == "target_sensitivity":
        return _pick_threshold_for_sensitivity(y_true, y_prob, target_sens)
    if mode == "constrained_sens":
        return _pick_threshold_constrained_sens(y_true, y_prob, max_fpr=target_fpr)
    return _pick_threshold_knee(y_true, y_prob)


def _metrics_and_plots(y_true, y_prob, outdir, prefix, threshold_for_report=None, target_sens=None, mode_used=None):
    os.makedirs(outdir, exist_ok=True)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    acc05, sens05, spec05 = _write_thresh_metrics(y_true, y_prob, 0.5)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    with open(os.path.join(outdir, f"{prefix}_metrics.txt"), "w") as f:
        f.write(f"AUC: {auc:.4f}\nPR-AUC: {ap:.4f}\nBrier: {brier:.4f}\n")
        f.write(f"Accuracy@0.5: {acc05:.4f}\nSensitivity@0.5: {sens05:.4f}\nSpecificity@0.5: {spec05:.4f}\n")
        if threshold_for_report is not None:
            accT, sensT, specT = _write_thresh_metrics(y_true, y_prob, threshold_for_report)
            tag = f"T={threshold_for_report:.4f}"
            f.write("---\n")
            if mode_used is not None:
                f.write(f"ThresholdMode: {mode_used}\n")
            if target_sens is not None:
                f.write(f"TargetSensitivity: {target_sens:.2f}\n")
            f.write(f"Accuracy@{tag}: {accT:.4f}\nSensitivity@{tag}: {sensT:.4f}\nSpecificity@{tag}: {specT:.4f}\n")

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {prefix}")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_roc.png"), dpi=200); plt.close()

    # PR
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR - {prefix}")
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_pr.png"), dpi=200); plt.close()

    return {"auc": auc, "pr_auc": ap, "brier": brier}


# ---------- Pretty names for SHAP ----------
def _pretty_feature_names(cols):
    """Map raw column names to display names for SHAP plots."""
    out = []
    for c in cols:
        l = c.lower()
        if c == "clin_prob":
            out.append("Clinical")
        elif "adc" in l:
            out.append("ADC")
        elif "dwi" in l:
            out.append("DWI")
        elif "flair" in l:
            out.append("T2-FLAIR")
        elif "noncontrast" in l or "ncct" in l or "non-contrast" in l or "non_contrast" in l:
            out.append("Non-contrast")
        else:
            out.append(c.replace("_prob_cal", "").replace("_", " ").title())
    return out


# =========================
# Clinical probabilities (VAL/TEST/EXTERNAL) from Excel
# =========================

def get_clinical_val_test_probs(xlsx_path: str):
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]

    label_col = getattr(cfg, "clinical_label_col", None) or ("90dmRS" if "90dmRS" in df.columns else None)
    if label_col is None:
        raise ValueError("Expected label column not found (default '90dmRS'). Set cfg.clinical_label_col.")

    id_candidates_cfg = list(getattr(cfg, "clinical_id_candidates", []))
    default_ids = ["patient", "Identity", "姓名", "拼音", "case_id", "ID", "id"]
    id_candidates = [c for c in (id_candidates_cfg + default_ids) if c in df.columns]
    if not id_candidates:
        raise ValueError("No suitable ID column found. Provide cfg.clinical_id_candidates or include e.g. 'patient'.")
    id_col = id_candidates[0]

    if "split" not in df.columns:
        raise ValueError("Expected split column 'split' not found in clinical Excel.")
    split = df["split"].astype(str).str.strip().str.lower()

    mrs_bad_threshold = int(getattr(cfg, "mrs_bad_threshold", 3))
    y = (pd.to_numeric(df[label_col], errors="coerce") >= mrs_bad_threshold).astype(int)

    drop_cols = {id_col, "split", label_col}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = _coerce_mixed_types(X)

    pre = _build_preprocessor(X)

    mask_train = split.eq("train")
    mask_val = split.eq("val")
    mask_test = split.eq("test")
    mask_external = split.eq("external")

    X_train, y_train = X.loc[mask_train], y.loc[mask_train]
    X_val, y_val = X.loc[mask_val], y.loc[mask_val]
    X_test, y_test = X.loc[mask_test], y.loc[mask_test]
    X_external, y_external = X.loc[mask_external], y.loc[mask_external]

    ids_val = df.loc[mask_val, id_col].astype(str).tolist()
    ids_test = df.loc[mask_test, id_col].astype(str).tolist()
    ids_external = df.loc[mask_external, id_col].astype(str).tolist()

    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", n_jobs=-1, random_state=42
        ))
    ])
    rf.fit(X_train, y_train)
    p_val = rf.predict_proba(X_val)[:, 1]

    rf_trval = Pipeline(steps=rf.steps)
    rf_trval.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    p_test = rf_trval.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])
    p_external = rf_trval.predict_proba(X_external)[:, 1] if len(X_external) else np.array([])

    df_val = pd.DataFrame({"case_id": ids_val, "label": y_val.values.astype(int), "clin_prob": p_val})
    df_test = pd.DataFrame({"case_id": ids_test, "label": y_test.values.astype(int), "clin_prob": p_test})
    df_external = pd.DataFrame({"case_id": ids_external, "label": y_external.values.astype(int), "clin_prob": p_external})

    return df_val, df_test, df_external


# =========================
# Alignment helpers
# =========================

def _align_on_case_and_label(left: pd.DataFrame, right: pd.DataFrame, label_col="label"):
    try:
        return left.merge(right, on=["case_id", label_col], how="inner", validate="one_to_one")
    except Exception:
        return left.merge(right, on=["case_id", label_col], how="inner")


def _align_on_case_then_reconcile(left: pd.DataFrame, right: pd.DataFrame, outdir: str, side_tag: str):
    m = left.merge(right, on=["case_id"], how="inner", suffixes=("_img", "_clin"))
    report = []
    if "label_img" in m.columns and "label_clin" in m.columns:
        neq = m["label_img"] != m["label_clin"]
        n_neq = int(neq.sum())
        if n_neq > 0:
            report.append(f"[{side_tag}] Label mismatch on {n_neq} cases (keeping clinical label).")
            bad = m.loc[neq, ["case_id", "label_img", "label_clin"]]
            bad.to_csv(os.path.join(outdir, f"{side_tag}_label_mismatch.csv"), index=False)
        m["label"] = m["label_clin"]
    elif "label_clin" in m.columns:
        m["label"] = m["label_clin"]
    elif "label_img" in m.columns:
        m["label"] = m["label_img"]
    else:
        raise RuntimeError(f"[{side_tag}] No label column to reconcile.")
    drop_cols = [c for c in ["label_img", "label_clin"] if c in m.columns]
    m = m.drop(columns=drop_cols)
    return m, report


def _normalized_merge(img: pd.DataFrame, clin: pd.DataFrame, outdir: str, side_tag: str):
    img = img.copy()
    clin = clin.copy()
    img["_key"] = _normalize_id(img["case_id"])
    clin["_key"] = _normalize_id(clin["case_id"])
    m = img.merge(clin.rename(columns={"case_id": "case_id_clin"}), on="_key", how="inner", suffixes=("_img", "_clin"))
    if len(m):
        m["case_id"] = m["case_id_img"]
        m = m.drop(columns=[c for c in ["_key", "case_id_clin", "case_id_img"] if c in m.columns])
    report = []
    if "label_img" in m.columns and "label_clin" in m.columns:
        neq = m["label_img"] != m["label_clin"]
        n_neq = int(neq.sum())
        if n_neq > 0:
            report.append(f"[{side_tag}] (normalized) Label mismatch on {n_neq} cases (keeping clinical label).")
            bad = m.loc[neq, ["case_id", "label_img", "label_clin"]]
            bad.to_csv(os.path.join(outdir, f"{side_tag}_normalized_label_mismatch.csv"), index=False)
        m["label"] = m["label_clin"]
        m = m.drop(columns=["label_img", "label_clin"])
    return m, report


# =========================
# Imaging subset helpers (MR & NCCT)
# =========================

def _find_prob_cols(df_cols, token_list):
    """
    Find columns whose name contains any token (case-insensitive) AND endswith '_prob_cal'.
    Returns a list (possibly empty).
    """
    tokens = [t.lower() for t in token_list]
    out = []
    for c in df_cols:
        cl = c.lower()
        if not cl.endswith("_prob_cal"):
            continue
        if any(t in cl for t in tokens):
            out.append(c)
    return out


def _get_mr_prob_series(df):
    mr_fused_col = getattr(cfg, "mr_fused_col", None)
    if mr_fused_col and mr_fused_col in df.columns:
        return df[mr_fused_col].astype(float), mr_fused_col, [mr_fused_col]
    mr_modalities = list(getattr(cfg, "mr_modalities", ["DWI", "ADC", "FLAIR"]))
    mr_cols = _find_prob_cols(df.columns, mr_modalities)
    if not mr_cols:
        raise RuntimeError(
            "No MR probability columns found. "
            "Set cfg.mr_fused_col to an existing fused MR column, or "
            "ensure per-modality cols like DWI/ADC/FLAIR end with '_prob_cal'."
        )
    return df[mr_cols].mean(axis=1), "mr_agg_prob", mr_cols


def _get_ncct_prob_series(df):
    ncct_col = getattr(cfg, "ncct_col", None)
    if ncct_col and ncct_col in df.columns:
        return df[ncct_col].astype(float), ncct_col
    candidates = _find_prob_cols(df.columns, [getattr(cfg, "ncct_token", "NCCT")])
    if not candidates:
        raise RuntimeError(
            "No NCCT probability column found. "
            "Set cfg.ncct_col to the exact column (e.g., 'NCCT_prob_cal'), "
            "or ensure the NCCT column name contains 'NCCT' and ends with '_prob_cal'."
        )
    return df[candidates[0]].astype(float), candidates[0]


def _collect_shared_permodality_cols(*dfs):
    """
    Intersect of columns that end with '_prob_cal' across provided dataframes.
    Ensures we only use features present in all splits.
    """
    sets = []
    for d in dfs:
        sets.append(set([c for c in d.columns if c.endswith("_prob_cal")]))
    if not sets:
        return []
    shared = set.intersection(*sets)
    return sorted(shared)


# =========================
# Main late-stacking
# =========================

def main():
    outdir = getattr(cfg, "fusion_outdir", "fusion_out")
    os.makedirs(outdir, exist_ok=True)

    # 1) Load imaging fused predictions (from fusion_imaging.py)
    val_img = pd.read_csv(os.path.join(outdir, "val_predictions.csv"))
    test_img = pd.read_csv(os.path.join(outdir, "test_predictions.csv"))
    ext_path = os.path.join(outdir, "external_predictions.csv")
    external_test_img = pd.read_csv(ext_path) if os.path.exists(ext_path) else pd.DataFrame(columns=["case_id", "label"])

    # Choose primary imaging fused column for the original CLIN+IMG stack
    use_col = None
    weighted_val_path = os.path.join(outdir, "val_predictions_weighted.csv")
    weighted_test_path = os.path.join(outdir, "test_predictions_weighted.csv")
    weighted_external_path = os.path.join(outdir, "external_predictions_weighted.csv")

    if os.path.exists(weighted_val_path) and os.path.exists(weighted_test_path):
        val_w = pd.read_csv(weighted_val_path)
        test_w = pd.read_csv(weighted_test_path)
        val_img = val_img.merge(val_w[["case_id", "fused_prob_weighted"]], on="case_id", how="left")
        test_img = test_img.merge(test_w[["case_id", "fused_prob_weighted"]], on="case_id", how="left")
        if os.path.exists(weighted_external_path) and not external_test_img.empty:
            external_w = pd.read_csv(weighted_external_path)
            external_test_img = external_test_img.merge(
                external_w[["case_id", "fused_prob_weighted"]], on="case_id", how="left")
        use_col = "fused_prob_weighted"
    elif "fused_prob_LR" in val_img.columns and "fused_prob_LR" in test_img.columns:
        use_col = "fused_prob_LR"
    else:
        per_mod_cols_val = [c for c in val_img.columns if c.endswith("_prob_cal")]
        per_mod_cols_test = [c for c in test_img.columns if c.endswith("_prob_cal")]
        per_mod_cols_external = [c for c in external_test_img.columns if c.endswith("_prob_cal")]
        if not per_mod_cols_val or not per_mod_cols_test:
            raise RuntimeError("No imaging probability columns found. Run fusion_imaging.py first.")
        use_col = "imaging_mean_prob"
        val_img[use_col] = val_img[per_mod_cols_val].mean(axis=1)
        test_img[use_col] = test_img[per_mod_cols_test].mean(axis=1)
        if not external_test_img.empty and per_mod_cols_external:
            external_test_img[use_col] = external_test_img[per_mod_cols_external].mean(axis=1)

    # 2) Build clinical VAL/TEST(/EXTERNAL) probabilities from Excel
    clinical_xlsx = getattr(cfg, "clinical_xlsx", "stroke_multimodal_clinicalrecords_20250927.xlsx")
    val_clin, test_clin, external_clin = get_clinical_val_test_probs(clinical_xlsx)

    # Deduplicate
    for df in (val_img, test_img, external_test_img, val_clin, test_clin, external_clin):
        if not df.empty and "case_id" in df.columns:
            df.drop_duplicates(subset=["case_id"], inplace=True)

    # 3) Robust ID alignment + report
    report_lines = [f"Alignment report generated {datetime.now().isoformat()}"]

    # VAL
    val_try1 = _align_on_case_and_label(val_img.copy(), val_clin.copy())
    if len(val_try1):
        val = val_try1; report_lines.append(f"[VAL] strict (case_id,label) matched: {len(val)}")
    else:
        val_try2, rep2 = _align_on_case_then_reconcile(val_img.copy(), val_clin.copy(), outdir, "VAL")
        if len(val_try2):
            val = val_try2; report_lines.append(f"[VAL] relaxed (case_id only) matched: {len(val)}"); report_lines.extend(rep2)
        else:
            val_norm, repN = _normalized_merge(val_img.copy(), val_clin.copy(), outdir, "VAL")
            val = val_norm; report_lines.append(f"[VAL] normalized-key matched: {len(val)}"); report_lines.extend(repN)

    # TEST
    test_try1 = _align_on_case_and_label(test_img.copy(), test_clin.copy())
    if len(test_try1):
        test = test_try1; report_lines.append(f"[TEST] strict (case_id,label) matched: {len(test)}")
    else:
        test_try2, rep2t = _align_on_case_then_reconcile(test_img.copy(), test_clin.copy(), outdir, "TEST")
        if len(test_try2):
            test = test_try2; report_lines.append(f"[TEST] relaxed (case_id only) matched: {len(test)}"); report_lines.extend(rep2t)
        else:
            test_norm, repNt = _normalized_merge(test_img.copy(), test_clin.copy(), outdir, "TEST")
            test = test_norm; report_lines.append(f"[TEST] normalized-key matched: {len(test)}"); report_lines.extend(repNt)

    # EXTERNAL (optional)
    external_test = pd.DataFrame()
    if not external_test_img.empty or not external_clin.empty:
        external_try1 = _align_on_case_and_label(external_test_img.copy(), external_clin.copy())
        if len(external_try1):
            external_test = external_try1; report_lines.append(f"[EXTERNAL] strict matched: {len(external_test)}")
        else:
            external_try2, rep2e = _align_on_case_then_reconcile(external_test_img.copy(), external_clin.copy(), outdir, "EXTERNAL")
            if len(external_try2):
                external_test = external_try2; report_lines.append(f"[EXTERNAL] relaxed matched: {len(external_test)}"); report_lines.extend(rep2e)
            else:
                external_norm, repNe = _normalized_merge(external_test_img.copy(), external_clin.copy(), outdir, "EXTERNAL")
                external_test = external_norm; report_lines.append(f"[EXTERNAL] normalized-key matched: {len(external_test)}"); report_lines.extend(repNe)

    # Unmatched tracking
    def _unmatched(left, right, tag):
        if left.empty or right.empty: return
        a = set(left["case_id"]); b = set(right["case_id"]); inter = a & b
        u_left = a - inter; u_right = b - inter
        if u_left:
            pd.DataFrame({f"unmatched_{tag.lower()}_img": sorted(u_left)}).to_csv(
                os.path.join(outdir, f"{tag}_unmatched_from_imaging.csv"), index=False)
            report_lines.append(f"[{tag}] imaging unmatched: {len(u_left)}")
        if u_right:
            pd.DataFrame({f"unmatched_{tag.lower()}_clin": sorted(u_right)}).to_csv(
                os.path.join(outdir, f"{tag}_unmatched_from_clinical.csv"), index=False)
            report_lines.append(f"[{tag}] clinical unmatched: {len(u_right)}")

    _unmatched(val_img, val_clin, "VAL")
    _unmatched(test_img, test_clin, "TEST")
    if not external_test_img.empty or not external_clin.empty:
        _unmatched(external_test_img, external_clin, "EXTERNAL")

    with open(os.path.join(outdir, "alignment_report.txt"), "w") as f:
        f.write("\n".join(report_lines) + "\n")

    if len(val) == 0 or len(test) == 0:
        raise RuntimeError("Case alignment failed for VAL or TEST. See alignment_report.txt and *_unmatched_*.csv")
    # External is optional

    # ============ Existing CLIN+IMG (primary) ============

    def dump_block(y, p, prefix, thr=None, mode=None, target_sens=None):
        _metrics_and_plots(y, p, outdir, prefix, threshold_for_report=thr, target_sens=target_sens, mode_used=mode)
        auc = roc_auc_score(y, p); ap = average_precision_score(y, p); b = brier_score_loss(y, p)
        if thr is not None:
            accT, sensT, specT = _write_thresh_metrics(y, p, thr)
        else:
            accT = sensT = specT = np.nan
        return {"auc": auc, "ap": ap, "brier": b, "acc@T": accT, "sens@T": sensT, "spec@T": specT}

    # Prepare arrays for original stack
    X_val_base = val[[use_col, "clin_prob"]].values
    y_val = val["label"].values.astype(int)
    X_test_base = test[[use_col, "clin_prob"]].values
    y_test = test["label"].values.astype(int)

    scaler_base = StandardScaler().fit(X_val_base)
    meta_base = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42).fit(
        scaler_base.transform(X_val_base), y_val)
    p_val_base = meta_base.predict_proba(scaler_base.transform(X_val_base))[:, 1]
    p_test_base = meta_base.predict_proba(scaler_base.transform(X_test_base))[:, 1]

    mode = str(getattr(cfg, 'fusion_threshold_mode', 'target_point')).lower()
    target_sens = float(getattr(cfg, 'fusion_target_sensitivity', 0.75))
    target_tpr = float(getattr(cfg, 'fusion_target_tpr', 0.80))
    target_fpr = float(getattr(cfg, 'fusion_target_fpr', 0.35))
    thr_base, _, _ = _choose_threshold(y_val, p_val_base, mode=mode,
                                       target_sens=target_sens, target_tpr=target_tpr, target_fpr=target_fpr)

    res_val_base = dump_block(y_val, p_val_base, "val_fused_CLINICAL+IMG", thr_base, mode, target_sens)
    res_test_base = dump_block(y_test, p_test_base, "test_fused_CLINICAL+IMG", thr_base, mode, target_sens)

    # External (optional)
    if not external_test.empty:
        X_external_base = external_test[[use_col, "clin_prob"]].values
        y_external = external_test["label"].values.astype(int)
        p_external_base = meta_base.predict_proba(scaler_base.transform(X_external_base))[:, 1]
        res_external_base = dump_block(y_external, p_external_base, "external_fused_CLINICAL+IMG",
                                       thr_base, mode, target_sens)
    else:
        y_external = np.array([]); p_external_base = np.array([]); res_external_base = None

    # Baselines
    res_val_img = dump_block(y_val, val[use_col].values, "val_IMG_only")
    res_test_img = dump_block(y_test, test[use_col].values, "test_IMG_only")
    if not external_test.empty:
        res_external_img = dump_block(y_external, external_test[use_col].values, "external_IMG_only")
    res_val_clin = dump_block(y_val, val["clin_prob"].values, "val_CLIN_only")
    res_test_clin = dump_block(y_test, test["clin_prob"].values, "test_CLIN_only")
    if not external_test.empty:
        res_external_clin = dump_block(y_external, external_test["clin_prob"].values, "external_CLIN_only")

    # Save per-case for base
    val_out = val.assign(stacked_prob=p_val_base, img_prob=val[use_col], clin_prob=val["clin_prob"])
    test_out = test.assign(stacked_prob=p_test_base, img_prob=test[use_col], clin_prob=test["clin_prob"])
    val_out.to_csv(os.path.join(outdir, "val_predictions_clinical_plus_img.csv"), index=False)
    test_out.to_csv(os.path.join(outdir, "test_predictions_clinical_plus_img.csv"), index=False)
    if not external_test.empty:
        external_out = external_test.assign(stacked_prob=p_external_base, img_prob=external_test[use_col],
                                            clin_prob=external_test["clin_prob"])
        external_out.to_csv(os.path.join(outdir, "external_predictions_clinical_plus_img.csv"), index=False)

    joblib.dump(
        {
            "scaler": scaler_base,
            "meta_lr": meta_base,
            "imaging_prob_key": use_col,
            "threshold_mode": mode,
            "chosen_threshold": float(thr_base),
            "target_sensitivity": target_sens,
            "target_tpr": target_tpr,
            "target_fpr": target_fpr,
            "columns": ["img_prob", "clin_prob"],
            "coef": meta_base.coef_.ravel().tolist(),
            "intercept": float(meta_base.intercept_.ravel()[0]),
        },
        os.path.join(outdir, "fusion_model_CLINICAL_PLUS_IMG.joblib")
    )

    # ============ MR + Clinical ============
    mr_val_series, mr_val_key, mr_val_cols = _get_mr_prob_series(val)
    mr_test_series, _, _ = _get_mr_prob_series(test)
    mr_external_series = None
    if not external_test.empty:
        try:
            mr_external_series, _, _ = _get_mr_prob_series(external_test)
        except Exception:
            pass  # allow missing MR on external

    X_val_mr = np.vstack([mr_val_series.values, val["clin_prob"].values]).T
    X_test_mr = np.vstack([mr_test_series.values, test["clin_prob"].values]).T
    scaler_mr = StandardScaler().fit(X_val_mr)
    meta_mr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42).fit(
        scaler_mr.transform(X_val_mr), y_val)
    p_val_mr = meta_mr.predict_proba(scaler_mr.transform(X_val_mr))[:, 1]
    p_test_mr = meta_mr.predict_proba(scaler_mr.transform(X_test_mr))[:, 1]
    thr_mr, _, _ = _choose_threshold(y_val, p_val_mr, mode=mode,
                                     target_sens=target_sens, target_tpr=target_tpr, target_fpr=target_fpr)

    _metrics_and_plots(y_val, p_val_mr, outdir, "val_fused_MRplusCLIN",
                       threshold_for_report=thr_mr, target_sens=target_sens, mode_used=mode)
    _metrics_and_plots(y_test, p_test_mr, outdir, "test_fused_MRplusCLIN",
                       threshold_for_report=thr_mr, target_sens=target_sens, mode_used=mode)
    if not external_test.empty and mr_external_series is not None:
        X_external_mr = np.vstack([mr_external_series.values, external_test["clin_prob"].values]).T
        p_external_mr = meta_mr.predict_proba(scaler_mr.transform(X_external_mr))[:, 1]
        _metrics_and_plots(y_external, p_external_mr, outdir, "external_fused_MRplusCLIN",
                           threshold_for_report=thr_mr, target_sens=target_sens, mode_used=mode)

    # Persist + per-case
    val.assign(mr_prob=mr_val_series.values, stacked_prob=p_val_mr).to_csv(
        os.path.join(outdir, "val_predictions_MR_plus_clinical.csv"), index=False)
    test.assign(mr_prob=mr_test_series.values, stacked_prob=p_test_mr).to_csv(
        os.path.join(outdir, "test_predictions_MR_plus_clinical.csv"), index=False)
    if not external_test.empty and mr_external_series is not None:
        external_test.assign(mr_prob=mr_external_series.values).to_csv(
            os.path.join(outdir, "external_predictions_MR_plus_clinical.csv"), index=False)

    joblib.dump(
        {
            "scaler": scaler_mr,
            "meta_lr": meta_mr,
            "mr_prob_key": mr_val_key,
            "mr_component_cols": mr_val_cols,
            "threshold_mode": mode,
            "chosen_threshold": float(thr_mr),
            "target_sensitivity": target_sens,
            "target_tpr": target_tpr,
            "target_fpr": target_fpr,
            "columns": ["mr_prob", "clin_prob"],
            "coef": meta_mr.coef_.ravel().tolist(),
            "intercept": float(meta_mr.intercept_.ravel()[0]),
        },
        os.path.join(outdir, "fusion_model_MR_PLUS_CLINICAL.joblib")
    )

    # ============ NCCT + Clinical ============
    try:
        ncct_val_series, ncct_key = _get_ncct_prob_series(val)
        ncct_test_series, _ = _get_ncct_prob_series(test)

        X_val_ncct = np.vstack([ncct_val_series.values, val["clin_prob"].values]).T
        X_test_ncct = np.vstack([ncct_test_series.values, test["clin_prob"].values]).T

        scaler_ncct = StandardScaler().fit(X_val_ncct)
        meta_ncct = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42).fit(
            scaler_ncct.transform(X_val_ncct), y_val)
        p_val_ncct = meta_ncct.predict_proba(scaler_ncct.transform(X_val_ncct))[:, 1]
        p_test_ncct = meta_ncct.predict_proba(scaler_ncct.transform(X_test_ncct))[:, 1]
        thr_ncct, _, _ = _choose_threshold(y_val, p_val_ncct, mode=mode,
                                           target_sens=target_sens, target_tpr=target_tpr, target_fpr=target_fpr)

        _metrics_and_plots(y_val, p_val_ncct, outdir, "val_fused_NCCTplusCLIN",
                           threshold_for_report=thr_ncct, target_sens=target_sens, mode_used=mode)
        _metrics_and_plots(y_test, p_test_ncct, outdir, "test_fused_NCCTplusCLIN",
                           threshold_for_report=thr_ncct, target_sens=target_sens, mode_used=mode)

        if not external_test.empty:
            try:
                ncct_external_series, _ = _get_ncct_prob_series(external_test)
                X_external_ncct = np.vstack([ncct_external_series.values, external_test["clin_prob"].values]).T
                p_external_ncct = meta_ncct.predict_proba(scaler_ncct.transform(X_external_ncct))[:, 1]
                _metrics_and_plots(y_external, p_external_ncct, outdir, "external_fused_NCCTplusCLIN",
                                   threshold_for_report=thr_ncct, target_sens=target_sens, mode_used=mode)
            except Exception:
                pass

        # Persist + per-case
        val.assign(ncct_prob=ncct_val_series.values).to_csv(
            os.path.join(outdir, "val_predictions_NCCT_plus_clinical.csv"), index=False)
        test.assign(ncct_prob=ncct_test_series.values).to_csv(
            os.path.join(outdir, "test_predictions_NCCT_plus_clinical.csv"), index=False)
        if not external_test.empty:
            try:
                external_test.assign(ncct_prob=ncct_external_series.values).to_csv(
                    os.path.join(outdir, "external_predictions_NCCT_plus_clinical.csv"), index=False)
            except Exception:
                pass

        res_val_ncct_d = {"auc": roc_auc_score(y_val, p_val_ncct)}
        res_test_ncct_d = {"auc": roc_auc_score(y_test, p_test_ncct)}
    except RuntimeError as e:
        with open(os.path.join(outdir, "ncct_plus_clinical.SKIPPED.txt"), "w") as f:
            f.write(str(e) + "\n")
        res_val_ncct_d = res_test_ncct_d = None

    # ============ NEW: Clinical + Individual imaging modalities (with SHAP) ============
    permod_cols_shared = _collect_shared_permodality_cols(val, test, external_test if not external_test.empty else val)
    if not permod_cols_shared:
        print("[warn] No shared per-modality *_prob_cal columns found; skipping CLIN+per-modality SHAP.")
    else:
        feat_names = permod_cols_shared + ["clin_prob"]
        display_names = _pretty_feature_names(feat_names)

        X_val_full = val[feat_names].values
        X_test_full = test[feat_names].values
        Z_scaler = StandardScaler().fit(X_val_full)
        Z_val = Z_scaler.transform(X_val_full)
        Z_test = Z_scaler.transform(X_test_full)

        lr_full = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42).fit(Z_val, y_val)
        p_val_full = lr_full.predict_proba(Z_val)[:, 1]
        p_test_full = lr_full.predict_proba(Z_test)[:, 1]
        thr_full, _, _ = _choose_threshold(y_val, p_val_full, mode=mode,
                                           target_sens=target_sens, target_tpr=target_tpr, target_fpr=target_fpr)

        _metrics_and_plots(y_val, p_val_full, outdir, "val_fused_CLIN_plus_MODALITIES",
                           threshold_for_report=thr_full, target_sens=target_sens, mode_used=mode)
        _metrics_and_plots(y_test, p_test_full, outdir, "test_fused_CLIN_plus_MODALITIES",
                           threshold_for_report=thr_full, target_sens=target_sens, mode_used=mode)

        # External (optional) for SHAP model
        if not external_test.empty:
            X_ext_full = external_test[feat_names].values
            Z_ext = Z_scaler.transform(X_ext_full)
            p_ext_full = lr_full.predict_proba(Z_ext)[:, 1]
            _metrics_and_plots(y_external, p_ext_full, outdir, "external_fused_CLIN_plus_MODALITIES",
                               threshold_for_report=thr_full, target_sens=target_sens, mode_used=mode)

        # Save per-case
        val.assign(stacked_prob=p_val_full).to_csv(
            os.path.join(outdir, "val_predictions_clinical_plus_modalities.csv"), index=False)
        test.assign(stacked_prob=p_test_full).to_csv(
            os.path.join(outdir, "test_predictions_clinical_plus_modalities.csv"), index=False)
        if not external_test.empty:
            external_test.assign(stacked_prob=p_ext_full).to_csv(
                os.path.join(outdir, "external_predictions_clinical_plus_modalities.csv"), index=False)

        # Persist model
        joblib.dump(
            {
                "scaler": Z_scaler,
                "meta_lr": lr_full,
                "feature_names": feat_names,
                "threshold_mode": mode,
                "chosen_threshold": float(thr_full),
                "target_sensitivity": target_sens,
                "target_tpr": target_tpr,
                "target_fpr": target_fpr,
                "coef": lr_full.coef_.ravel().tolist(),
                "intercept": float(lr_full.intercept_.ravel()[0]),
            },
            os.path.join(outdir, "fusion_model_CLIN_PLUS_MODALITIES.joblib")
        )

        # ---- SHAP (beeswarm + bar) for TEST and EXTERNAL ----
        try:
            import shap
            have_shap = True
        except Exception as e:
            print("[warn] SHAP not available:", e); have_shap = False

        def _save_plot(figpath):
            plt.tight_layout()
            plt.savefig(figpath, dpi=200, bbox_inches="tight")
            plt.close()

        mapping = {raw: disp for raw, disp in zip(feat_names, display_names)}
        with open(os.path.join(outdir, "shap_feature_name_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)

        if have_shap:
            bg = Z_val if Z_val.shape[0] <= 200 else Z_val[:200]
            try:
                explainer = shap.Explainer(lr_full, bg, feature_names=feat_names)
            except Exception:
                explainer = shap.LinearExplainer(lr_full, bg, feature_names=feat_names)

            # TEST
            try:
                sv_test = explainer(Z_test)
                try: sv_test.feature_names = display_names
                except Exception: pass
                shap.plots.beeswarm(sv_test, show=False, max_display=len(display_names))
                _save_plot(os.path.join(outdir, "shap_beeswarm_CLIN_plus_MODALITIES_TEST.png"))
                shap.plots.bar(sv_test, show=False, max_display=len(display_names))
                _save_plot(os.path.join(outdir, "shap_bar_CLIN_plus_MODALITIES_TEST.png"))
                np.save(os.path.join(outdir, "shap_values_CLIN_plus_MODALITIES_TEST.npy"), sv_test.values)
                np.save(os.path.join(outdir, "shap_base_CLIN_plus_MODALITIES_TEST.npy"), sv_test.base_values)
                pd.DataFrame(Z_test, columns=feat_names).to_csv(
                    os.path.join(outdir, "shap_features_raw_CLIN_plus_MODALITIES_TEST.csv"), index=False)
                pd.DataFrame(Z_test, columns=display_names).to_csv(
                    os.path.join(outdir, "shap_features_display_CLIN_plus_MODALITIES_TEST.csv"), index=False)
            except Exception as e:
                print("[warn] SHAP TEST failed:", e)

            # EXTERNAL
            if not external_test.empty:
                try:
                    sv_ext = explainer(Z_ext)
                    try: sv_ext.feature_names = display_names
                    except Exception: pass
                    shap.plots.beeswarm(sv_ext, show=False, max_display=len(display_names))
                    _save_plot(os.path.join(outdir, "shap_beeswarm_CLIN_plus_MODALITIES_EXTERNAL.png"))
                    shap.plots.bar(sv_ext, show=False, max_display=len(display_names))
                    _save_plot(os.path.join(outdir, "shap_bar_CLIN_plus_MODALITIES_EXTERNAL.png"))
                    np.save(os.path.join(outdir, "shap_values_CLIN_plus_MODALITIES_EXTERNAL.npy"), sv_ext.values)
                    np.save(os.path.join(outdir, "shap_base_CLIN_plus_MODALITIES_EXTERNAL.npy"), sv_ext.base_values)
                    pd.DataFrame(Z_ext, columns=feat_names).to_csv(
                        os.path.join(outdir, "shap_features_raw_CLIN_plus_MODALITIES_EXTERNAL.csv"), index=False)
                    pd.DataFrame(Z_ext, columns=display_names).to_csv(
                        os.path.join(outdir, "shap_features_display_CLIN_plus_MODALITIES_EXTERNAL.csv"), index=False)
                except Exception as e:
                    print("[warn] SHAP EXTERNAL failed:", e)

    # ============ Summary JSON (compact) ============
    summary_path = os.path.join(outdir, "fusion_summary_all.json")
    summary = {
        "threshold_mode": mode,
        "target_sensitivity": target_sens,
        "target_tpr": target_tpr,
        "target_fpr": target_fpr,
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_external_test": int(len(external_test)) if not external_test.empty else 0,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Late fusion variants complete ===")
    print("Saved artifacts to:", outdir)


if __name__ == "__main__":
    main()
