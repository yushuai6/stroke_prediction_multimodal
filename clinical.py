"""
Predicts 90-day modified Rankin Scale (mRS, 0–2 vs 3–6) outcomes using a Random Forest classifier
based solely on structured clinical features.

Includes:
• Validation-based threshold selection (Youden index)
• Evaluation on validation, test, and external splits
• ROC, confusion matrices, and per-case probability exports
• Permutation importance analysis
• SHAP summary and bar plots for interpretability
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils import check_random_state
from joblib import dump
import shap
import matplotlib.pyplot as plt

# ==============================================================
# Configuration
# ==============================================================
RANDOM_SEED = 42
rng = check_random_state(RANDOM_SEED)
RESULTS_DIR = "results_clinical"


# ==============================================================
# I/O helpers
# ==============================================================
def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def export_confusion_matrix(y_true, y_prob, threshold, prefix):
    thr = float(threshold)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    df = pd.DataFrame([{"threshold": thr, "TN": tn, "FP": fp, "FN": fn, "TP": tp}])
    label = "0.5" if abs(thr - 0.5) < 1e-12 else "selectedT"
    path = os.path.join(RESULTS_DIR, f"{prefix}_confusion_at_{label}.csv")
    df.to_csv(path, index=False)
    return path


def export_roc_points(y_true, y_prob, prefix):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    path = os.path.join(RESULTS_DIR, f"{prefix}_roc_points.csv")
    df.to_csv(path, index=False)
    return path


def detect_id_column(df: pd.DataFrame):
    """Try to infer ID column name heuristically."""
    candidates = ["patient", "case", "id", "subject", "mrn", "study_id"]
    for c in candidates:
        for variant in [c, c.upper(), c.capitalize()]:
            if variant in df.columns:
                return variant
    return None


def export_per_case_predictions(indices, ids, split, y_true, y_prob, thr, filename):
    preds = (np.asarray(y_prob) >= float(thr)).astype(int)
    df = pd.DataFrame({
        "case_index": indices,
        "split": split,
        "y_true": y_true.astype(int),
        "proba": y_prob.astype(float),
        "pred_at_selectedT": preds,
        "threshold_used": float(thr),
    })
    if ids is not None:
        df.insert(1, "case_id", ids.values)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    return path


# ==============================================================
# Data loading and preprocessing
# ==============================================================
def load_data(xlsx_path: str) -> pd.DataFrame:
    """Load the clinical Excel dataset. Adjust usecols if needed."""
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def make_label(y_col: pd.Series) -> pd.Series:
    """Binary label: mRS ≥3 → 1 (poor), else 0 (favorable)."""
    y_num = pd.to_numeric(y_col, errors="coerce")
    return (y_num >= 3).astype(int)


def infer_splits(col: pd.Series):
    """Infer split masks from a column containing split labels (train/val/test/external)."""
    vals = col.astype(str).str.strip().str.lower()
    mask_train = vals.str.contains("train|trn|0")
    mask_val = vals.str.contains("val|valid|1")
    mask_test = vals.str.contains("test|tst|2")
    mask_external = vals.str.contains("external|ext|3|4")
    return mask_train, mask_val, mask_test, mask_external


def coerce_mixed_types(X: pd.DataFrame) -> pd.DataFrame:
    """Convert mostly-numeric object columns to numeric."""
    Xc = X.copy()
    for c in Xc.columns:
        s = Xc[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sn = pd.to_numeric(s, errors="coerce")
            if sn.notna().sum() / max(s.notna().sum(), 1) >= 0.7:
                Xc[c] = sn
    return Xc


def _make_onehot():
    """Compatibility helper for scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build numeric/categorical preprocessing pipeline."""
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_onehot()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop", sparse_threshold=0.0)


# ==============================================================
# Modeling
# ==============================================================
def evaluate_on_val_get_threshold(pipe, X_train, y_train, X_val, y_val):
    """Train RF on training data, find optimal validation threshold."""
    pipe.fit(X_train, y_train)
    prob_val = pipe.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, prob_val)
    fpr, tpr, thr = roc_curve(y_val, prob_val)
    thr_opt = float(thr[np.nanargmax(tpr - fpr)])
    export_roc_points(y_val, prob_val, "val_rf")
    export_confusion_matrix(y_val, prob_val, 0.5, "val_rf")
    export_confusion_matrix(y_val, prob_val, thr_opt, "val_rf")
    return thr_opt, prob_val, auc_val


def evaluate_on_split(pipe, X_trainval, y_trainval, X_eval, y_eval, thr, prefix):
    """Evaluate model performance on test/external split."""
    pipe.fit(X_trainval, y_trainval)
    prob = pipe.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_eval, prob)
    export_roc_points(y_eval, prob, prefix)
    export_confusion_matrix(y_eval, prob, 0.5, prefix)
    export_confusion_matrix(y_eval, prob, thr, prefix)
    y_pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return {"auc": auc, "sens": sens, "spec": spec, "prob": prob}


def compute_permutation_importance(pipe, X_eval, y_eval, n_repeats=50):
    """Compute permutation importance for model features."""
    result = permutation_importance(
        pipe, X_eval, y_eval, scoring="roc_auc", n_repeats=n_repeats,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    df_imp = pd.DataFrame({
        "feature": X_eval.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df_imp.insert(0, "rank", np.arange(1, len(df_imp) + 1))
    return df_imp


# ==============================================================
# Main
# ==============================================================
def main(xlsx_path: str):
    ensure_results_dir()
    print("📘 Loading clinical data from:", xlsx_path)
    df_all = load_data(xlsx_path)

    # Identify columns
    label_col = df_all.columns[-2]
    split_col = df_all.columns[-1]
    y = make_label(df_all[label_col])
    X = df_all.drop(columns=[label_col, split_col])

    id_col = detect_id_column(df_all)
    ids = df_all[id_col] if id_col else None

    X = coerce_mixed_types(X)
    pre = build_preprocessor(X)
    m_train, m_val, m_test, m_ext = infer_splits(df_all[split_col])

    # Split datasets
    X_train, y_train = X[m_train], y[m_train]
    X_val, y_val = X[m_val], y[m_val]
    X_test, y_test = X[m_test], y[m_test]
    X_ext, y_ext = X[m_ext], y[m_ext]

    rf_pipe = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
        ))
    ])

    print("\n=== Random Forest Training & Evaluation ===")
    thr_opt, prob_val, auc_val = evaluate_on_val_get_threshold(rf_pipe, X_train, y_train, X_val, y_val)
    rf_trval = Pipeline(steps=rf_pipe.steps)

    res_test = evaluate_on_split(rf_trval, pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), X_test, y_test, thr_opt, "test_rf")
    res_ext = evaluate_on_split(rf_trval, pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), X_ext, y_ext, thr_opt, "external_rf")

    print(f"Validation AUC={auc_val:.3f} | Test AUC={res_test['auc']:.3f} | External AUC={res_ext['auc']:.3f}")

    # Save per-case predictions
    export_per_case_predictions(X_val.index, ids.loc[X_val.index] if ids is not None else None,
                                "val", y_val, prob_val, thr_opt, "val_rf_predictions.csv")
    export_per_case_predictions(X_test.index, ids.loc[X_test.index] if ids is not None else None,
                                "test", y_test, res_test["prob"], thr_opt, "test_rf_predictions.csv")
    export_per_case_predictions(X_ext.index, ids.loc[X_ext.index] if ids is not None else None,
                                "external", y_ext, res_ext["prob"], thr_opt, "external_rf_predictions.csv")

    # Permutation importance
    print("\n=== Permutation Importance (ROC-AUC) ===")
    df_perm = compute_permutation_importance(rf_trval, X_test, y_test)
    df_perm.to_csv(os.path.join(RESULTS_DIR, "rf_permutation_importance.csv"), index=False)
    print(df_perm.head(10).to_string(index=False))

    # SHAP analysis
    print("\n=== SHAP Analysis ===")
    X_test_t = rf_trval.named_steps["pre"].transform(X_test)
    fnames = rf_trval.named_steps["pre"].get_feature_names_out()
    df_t = pd.DataFrame(X_test_t, columns=fnames, index=X_test.index)
    explainer = shap.Explainer(rf_trval.named_steps["clf"], df_t)
    shap_vals = explainer(df_t)
    shap_values = shap_vals.values[..., 1] if shap_vals.values.ndim == 3 else shap_vals.values

    plt.figure()
    shap.summary_plot(shap_values, df_t, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_summary.png"), dpi=300)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, df_t, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_bar.png"), dpi=300)
    plt.close()
    print("✅ SHAP plots saved to results_clinical/")

    # Save model
    dump(rf_trval, os.path.join(RESULTS_DIR, "rf_model.joblib"))
    print("✅ Model pipeline saved.")


if __name__ == "__main__":
    default_path = "clinical_records.xlsx"
    cli_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    path = cli_path if os.path.exists(cli_path) else default_path
    main(path)
