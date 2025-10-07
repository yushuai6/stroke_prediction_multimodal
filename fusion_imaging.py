# Late-fusion for binary 90d mRS using config.py, dataset.py, and resnet3D.py
# Supports:
#  - Per-modality temperature calibration
#  - Logistic-regression stacker baseline
#  - Weighted fusion search to meet an operating target
#  - Threshold selection modes: "knee", "youden", "target_sensitivity",
#    "target_point", "constrained_sens"
#  - Exports confusion matrices, ROC/PR points for each evaluation
#  - MR-only fusion (DWI, ADC, FLAIR) parallel to all-modality fusion
#  - External-set evaluation (split == "external")
#  - SHAP plots for LR stacker + per-modality contribution plots for weighted fusion

import os
import json
import numpy as np
import pandas as pd
import joblib

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, accuracy_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Project imports (run from project root)
import config as cfg
import dataset as ds_mod
import resnet3D as net_mod


# ========= Utilities =========
def safe_load(path):
    """torch.load with weights_only=True when available (silences security warning)."""
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _sanitize_probs(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with mean of finite values (or 0.5 if none)."""
    arr = np.asarray(arr, dtype=float)
    if not np.isfinite(arr).any():
        return np.full_like(arr, 0.5, dtype=float)
    m = np.nanmean(np.where(np.isfinite(arr), arr, np.nan))
    arr = np.where(np.isfinite(arr), arr, m)
    return arr


def _safe_auc(y_true, y_prob) -> float:
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return float('nan')


def _safe_ap(y_true, y_prob) -> float:
    try:
        return average_precision_score(y_true, y_prob)
    except Exception:
        return float('nan')


def _roc_safe(y_true, y_prob):
    """
    Safe ROC: returns arrays even if only one class present.
    """
    try:
        return roc_curve(y_true, y_prob)
    except Exception:
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        # two points: (FPR=1-spec, TPR=sens) and ideal (0,1) just for plotting logic
        return np.array([1 - spec, 0.0]), np.array([sens, 1.0]), np.array([0.5, 1.0])


def write_thresh_metrics(y_true, y_prob, threshold):
    y_pred = (np.asarray(y_prob) >= float(threshold)).astype(int)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return acc, sens, spec


def export_confusion_matrix(y_true, y_prob, threshold, outpath):
    """Export a 2x2 confusion matrix (TN, FP, FN, TP) at a given threshold."""
    y_pred = (np.asarray(y_prob) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pd.DataFrame(
        [{'threshold': float(threshold), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}]
    ).to_csv(outpath, index=False)


def pick_threshold_for_sensitivity(y_true, y_prob, target_sens=0.70):
    """Pick the smallest threshold whose TPR >= target_sens; fallback to max TPR."""
    fpr, tpr, thresholds = _roc_safe(y_true, y_prob)
    idxs = np.where(tpr >= float(target_sens))[0]
    if len(idxs) > 0:
        idx = int(idxs[0])
    else:
        idx = int(np.argmax(tpr))
    thr = float(thresholds[idx])
    return thr, float(tpr[idx]), float(1.0 - fpr[idx])


def pick_threshold_knee(y_true, y_prob):
    """Pick the ROC knee: point closest to (FPR=0, TPR=1)."""
    fpr, tpr, thresholds = _roc_safe(y_true, y_prob)
    dist = np.sqrt((fpr - 0.0) ** 2 + (1.0 - tpr) ** 2)
    idx = int(np.argmin(dist))
    return float(thresholds[idx]), float(tpr[idx]), float(1.0 - fpr[idx])


def pick_threshold_youden(y_true, y_prob):
    """Pick threshold maximizing Youden’s J = TPR - FPR."""
    fpr, tpr, thresholds = _roc_safe(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(tpr[idx]), float(1.0 - fpr[idx])


def pick_threshold_target_point(y_true, y_prob, target_tpr=0.8, target_fpr=0.25):
    """Pick threshold whose (FPR,TPR) is closest to the target point."""
    fpr, tpr, thresholds = _roc_safe(y_true, y_prob)
    dist = np.sqrt((fpr - float(target_fpr))**2 + (tpr - float(target_tpr))**2)
    idx = int(np.argmin(dist))
    return float(thresholds[idx]), float(tpr[idx]), float(1.0 - fpr[idx])


def pick_threshold_constrained_sens(y_true, y_prob, max_fpr=0.25):
    """Maximize TPR subject to FPR <= max_fpr (on validation)."""
    fpr, tpr, thresholds = _roc_safe(y_true, y_prob)
    mask = fpr <= float(max_fpr) + 1e-12
    if np.any(mask):
        idxs = np.where(mask)[0]
        order = np.lexsort((fpr[idxs], -tpr[idxs]))  # sort by -TPR desc, then FPR asc
        best = idxs[order][0]
        return float(thresholds[best]), float(tpr[best]), float(1.0 - fpr[best])
    # fallback: closest FPR to max_fpr (tie-breaker: highest TPR)
    dif = np.abs(fpr - float(max_fpr))
    best = int(np.lexsort((dif, -tpr))[0])
    return float(thresholds[best]), float(tpr[best]), float(1.0 - fpr[best])


def choose_threshold(y_true, y_prob, mode="knee",
                     target_sens=0.75, target_tpr=0.8, target_fpr=0.25):
    mode = (mode or "knee").lower()
    if mode == "knee":
        return pick_threshold_knee(y_true, y_prob)
    if mode == "youden":
        return pick_threshold_youden(y_true, y_prob)
    if mode == "target_point":
        return pick_threshold_target_point(y_true, y_prob, target_tpr=target_tpr, target_fpr=target_fpr)
    if mode == "target_sensitivity":
        return pick_threshold_for_sensitivity(y_true, y_prob, target_sens)
    if mode == "constrained_sens":
        return pick_threshold_constrained_sens(y_true, y_prob, max_fpr=target_fpr)
    return pick_threshold_knee(y_true, y_prob)


def metrics_and_plots(y_true, y_prob, outdir, prefix,
                      threshold_for_report=None, target_sens=None, mode_used=None):
    os.makedirs(outdir, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = _sanitize_probs(np.asarray(y_prob, dtype=float))

    # Prob-based metrics
    auc = _safe_auc(y_true, y_prob)
    ap = _safe_ap(y_true, y_prob)
    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = float('nan')

    # Thresholded metrics @0.5
    acc05, sens05, spec05 = write_thresh_metrics(y_true, y_prob, 0.5)

    # Curves (guard when only one class present)
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    except Exception:
        fpr, tpr, thresholds = np.array([0, 1]), np.array([0, 1]), np.array([np.inf, -np.inf])
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
    except Exception:
        prec, rec = np.array([1.0]), np.array([0.0])
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    except Exception:
        frac_pos, mean_pred = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # === Export ROC & PR points ===
    pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}).to_csv(
        os.path.join(outdir, f'{prefix}_roc_points.csv'), index=False
    )
    pd.DataFrame({'recall': rec, 'precision': prec}).to_csv(
        os.path.join(outdir, f'{prefix}_pr_points.csv'), index=False
    )

    # Write metrics (+ prevalence & N)
    with open(os.path.join(outdir, f'{prefix}_metrics.txt'), 'w') as f:
        prev = float(np.mean(y_true))
        f.write(f"N: {len(y_true)}\n")
        f.write(f"Prevalence: {prev:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"PR-AUC: {ap:.4f}\n")
        f.write(f"Brier: {brier:.4f}\n")
        f.write(f"Accuracy@0.5: {acc05:.4f}\n")
        f.write(f"Sensitivity@0.5: {sens05:.4f}\n")
        f.write(f"Specificity@0.5: {spec05:.4f}\n")
        if threshold_for_report is not None:
            accT, sensT, specT = write_thresh_metrics(y_true, y_prob, threshold_for_report)
            tag = f"T={threshold_for_report:.4f}"
            f.write("---\n")
            if mode_used is not None:
                f.write(f"ThresholdMode: {mode_used}\n")
            if target_sens is not None:
                f.write(f"TargetSensitivity: {target_sens:.2f}\n")
            f.write(f"Accuracy@{tag}: {accT:.4f}\n")
            f.write(f"Sensitivity@{tag}: {sensT:.4f}\n")
            f.write(f"Specificity@{tag}: {specT:.4f}\n")

    # === Export confusion matrices ===
    export_confusion_matrix(y_true, y_prob, 0.5,
                            os.path.join(outdir, f'{prefix}_confusion_at_0.5.csv'))
    if threshold_for_report is not None:
        export_confusion_matrix(y_true, y_prob, threshold_for_report,
                                os.path.join(outdir, f'{prefix}_confusion_at_selectedT.csv'))

    # ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC - {prefix}')
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_roc.png'), dpi=200); plt.close()

    # PR
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f'AP={ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR - {prefix}')
    plt.legend(loc='lower left'); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_pr.png'), dpi=200); plt.close()

    # Calibration
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], '--')
    plt.plot(mean_pred, frac_pos, marker='o')
    plt.xlabel('Mean predicted prob'); plt.ylabel('Fraction of positives')
    plt.title(f'Calibration - {prefix}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'{prefix}_calibration.png'), dpi=200); plt.close()

    return {
        'auc': auc,
        'pr_auc': ap,
        'brier': brier,
        'acc@0.5': acc05,
        'sens@0.5': sens05,
        'spec@0.5': spec05,
    }


# ========= Calibration (per modality) =========
class _TempScale(nn.Module):
    """Temperature scaling for a single modality (fits T on val to minimize NLL)."""
    def __init__(self, init_T=1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(float(np.log(init_T)), dtype=torch.float32))
    def forward(self, logits):
        T = torch.exp(self.log_T)
        return logits / T
    def temperature(self):
        return float(torch.exp(self.log_T).detach().cpu().item())


def fit_temperature_scaler(val_logits, val_labels, max_iter=200):
    """Fit temperature on VAL logits/labels; return scaler on CPU to avoid device mismatches."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logits_t = torch.tensor(val_logits, dtype=torch.float32, device=device).view(-1, 1)
    labels_t = torch.tensor(val_labels, dtype=torch.float32, device=device).view(-1, 1)

    model = _TempScale().to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = optim.LBFGS(list(model.parameters()), lr=1.0, max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = bce(model(logits_t), labels_t)
        loss.backward()
        return loss

    prev = np.inf
    for _ in range(max_iter):
        loss = opt.step(closure)
        cur = float(loss.detach().cpu().item())
        if abs(prev - cur) < 1e-9:
            break

    return model.cpu()


@torch.no_grad()
def infer_logits(csv_path, image_dir, split, ckpt_path, input_cha,
                 target_shape, batch_size, num_workers, device):
    """Runs a single-modality model and returns aligned (case_ids, logits, labels)."""
    ds = ds_mod.StrokeImageDataset(
        csv_path=csv_path, image_dir=image_dir, split=split, target_shape=target_shape
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = net_mod.i3_res50forStrokeOutcome(input_cha=input_cha, num_classes=1).to(device)
    model.eval()

    ckpt = safe_load(ckpt_path)
    state_dict = None
    for key in ['state_dict', 'model_state_dict', 'net', 'model']:
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            break
    if state_dict is None:
        state_dict = ckpt if isinstance(ckpt, dict) else None
        if state_dict is None:
            raise RuntimeError("Unsupported checkpoint format.")
    new_sd = {(k[len('module.'): ] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    res = model.load_state_dict(new_sd, strict=False)
    missing = getattr(res, 'missing_keys', [])
    unexpected = getattr(res, 'unexpected_keys', [])
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {sorted(missing)[:5]}{' ...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}{' ...' if len(unexpected)>5 else ''}")

    all_logits, all_labels = [], []
    for (images, labels) in dl:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.view(-1).to(torch.long).cpu())

    logits = torch.cat(all_logits).numpy() if len(all_logits) else np.zeros((0,), dtype=float)
    labels = torch.cat(all_labels).numpy().astype(int) if len(all_labels) else np.zeros((0,), dtype=int)
    case_ids = ds.df['patient'].astype(str).tolist()
    return case_ids, logits, labels


# ========= Weighted Fusion Search =========
def _fused_prob_from_weights(logits_matrix, weights):
    # logits_matrix: (N, M) calibrated logits per modality
    # weights: (M,)
    fused_logit = logits_matrix @ weights
    return sigmoid(fused_logit)


def _sample_weights(M, bounds, normalize, rng):
    low, high = bounds
    w = rng.uniform(low, high, size=(M,))
    if normalize:
        w = w / (np.sum(w) + 1e-12)
    return w


def search_weights_for_target(
    X_val_logits, y_val, mode, target_sens, target_tpr, target_fpr,
    trials=2000, bounds=(0.0, 2.5), normalize=True, seed=42, initial_weights=None
):
    rng = np.random.default_rng(seed)
    M = X_val_logits.shape[1]

    best = {'weights': None, 'threshold': 0.5, 'val_sens': -1.0, 'val_spec': -1.0}

    # include provided initial weights first (if any & correct dim)
    if isinstance(initial_weights, (list, tuple, np.ndarray)) and len(initial_weights) == M:
        w0 = np.array(initial_weights, dtype=float)
        if normalize:
            w0 = w0 / (np.sum(w0) + 1e-12)
        p = _fused_prob_from_weights(X_val_logits, w0)
        if mode == "knee":
            thr, sens, spec = pick_threshold_knee(y_val, p)
        elif mode == "youden":
            thr, sens, spec = pick_threshold_youden(y_val, p)
        elif mode == "target_point":
            thr, sens, spec = pick_threshold_target_point(y_val, p, target_tpr, target_fpr)
        elif mode == "constrained_sens":
            thr, sens, spec = pick_threshold_constrained_sens(y_val, p, max_fpr=target_fpr)
        else:
            thr, sens, spec = pick_threshold_for_sensitivity(y_val, p, target_sens)
        best.update({'weights': w0, 'threshold': thr, 'val_sens': sens, 'val_spec': spec})

    # random search
    for _ in range(trials):
        w = _sample_weights(M, bounds, normalize, rng)
        p = _fused_prob_from_weights(X_val_logits, w)
        if mode == "knee":
            thr, sens, spec = pick_threshold_knee(y_val, p)
        elif mode == "youden":
            thr, sens, spec = pick_threshold_youden(y_val, p)
        elif mode == "target_point":
            thr, sens, spec = pick_threshold_target_point(y_val, p, target_tpr, target_fpr)
        elif mode == "constrained_sens":
            thr, sens, spec = pick_threshold_constrained_sens(y_val, p, max_fpr=target_fpr)
        else:
            thr, sens, spec = pick_threshold_for_sensitivity(y_val, p, target_sens)

        # objective preference: higher spec, then higher sens
        if (spec > best['val_spec'] + 1e-6 or
            (abs(spec - best['val_spec']) < 1e-6 and sens > best['val_sens'] + 1e-6)):
            best.update({'weights': w, 'threshold': thr, 'val_sens': sens, 'val_spec': spec})

    return best


def _set_global_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _has_external_split(csv_path, split_col, external_name):
    try:
        df = pd.read_csv(csv_path)
        if split_col not in df.columns:
            return False
        return df[split_col].astype(str).str.lower().eq(str(external_name).lower()).any()
    except Exception as e:
        print(f"[warn] Unable to scan CSV for external split: {e}")
        return False


def main():
    # ---- Config knobs ----
    device = cfg.device if hasattr(cfg, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu')
    # Defensive fallback if a stale config is imported
    if not hasattr(cfg, "fusion_modalities") and hasattr(cfg, "fusion_image_dirs"):
        cfg.fusion_modalities = list(cfg.fusion_image_dirs.keys())
        print("[warn] config.fusion_modalities missing; using keys from fusion_image_dirs:", cfg.fusion_modalities)

    modalities = cfg.fusion_modalities
    img_dirs = cfg.fusion_image_dirs
    ckpts = cfg.fusion_ckpts

    # threshold strategy
    mode = str(getattr(cfg, 'fusion_threshold_mode', 'constrained_sens')).lower()
    target_sens = float(getattr(cfg, 'fusion_target_sensitivity', 0.75))
    target_tpr  = float(getattr(cfg, 'fusion_target_tpr', 0.80))
    target_fpr  = float(getattr(cfg, 'fusion_target_fpr', 0.35))  # FPR cap

    # weighted fusion search params
    enable_weight_search = bool(getattr(cfg, 'fusion_weight_search', True))
    trials = int(getattr(cfg, 'fusion_weight_trials', 2000))
    bounds = tuple(getattr(cfg, 'fusion_weight_bounds', (0.0, 2.5)))
    normalize_weights = bool(getattr(cfg, 'fusion_weight_normalize', True))
    seed = int(getattr(cfg, 'fusion_seed', 42))
    init_weights_cfg = getattr(cfg, 'modality_weights', None)  # dict or None

    # Splits
    split_col = getattr(cfg, 'split_column', 'split')
    val_split = getattr(cfg, 'val_split', 'val')
    test_split = getattr(cfg, 'test_split', 'test')
    external_split = getattr(cfg, 'external_split', 'external')

    # Determinism
    _set_global_seeds(seed)

    # ---- sanity checks ----
    for m in modalities:
        if m not in img_dirs: raise ValueError(f"Missing image_dir for {m}")
        if m not in ckpts:    raise ValueError(f"Missing ckpt for {m}")
        if not os.path.exists(img_dirs[m]): print(f"[warn] image_dir not found: {img_dirs[m]}")
        if not os.path.exists(ckpts[m]):    print(f"[warn] ckpt not found: {ckpts[m]}")

    tz, ty, tx = cfg.target_shape  # (Z, Y, X)
    target_shape = (tz, ty, tx)

    # workers fallback
    num_workers = int(getattr(cfg, 'num_workers', 0))

    # external split present?
    has_external = _has_external_split(cfg.label_csv, split_col, external_split)
    if has_external:
        print(f"[info] External split '{external_split}' detected, external evaluation enabled.")
    else:
        print(f"[info] External split '{external_split}' not found, external evaluation skipped.")

    # ---- Inference per modality (val, test, external?) ----
    val_case_ids_all = None
    test_case_ids_all = None
    ext_case_ids_all  = None

    val_labels_all = None
    test_labels_all = None
    ext_labels_all  = None

    val_logits_by_mod = {}
    test_logits_by_mod = {}
    ext_logits_by_mod  = {}

    for m in modalities:
        print(f"[{m}] Inference on {val_split}...")
        v_ids, v_logits, v_labels = infer_logits(
            cfg.label_csv, img_dirs[m], val_split,
            ckpts[m], cfg.input_channels, target_shape, cfg.batch_size, num_workers, device
        )
        print(f"[{m}] Inference on {test_split}...")
        t_ids, t_logits, t_labels = infer_logits(
            cfg.label_csv, img_dirs[m], test_split,
            ckpts[m], cfg.input_channels, target_shape, cfg.batch_size, num_workers, device
        )
        if has_external:
            print(f"[{m}] Inference on {external_split}...")
            e_ids, e_logits, e_labels = infer_logits(
                cfg.label_csv, img_dirs[m], external_split,
                ckpts[m], cfg.input_channels, target_shape, cfg.batch_size, num_workers, device
            )
        else:
            e_ids, e_logits, e_labels = [], np.zeros((0,), dtype=float), np.zeros((0,), dtype=int)

        if val_case_ids_all is None:
            val_case_ids_all = v_ids; val_labels_all = v_labels
        else:
            assert v_ids == val_case_ids_all, f"Val case_id order mismatch for {m}"
            assert np.array_equal(v_labels, val_labels_all), f"Val labels mismatch for {m}"

        if test_case_ids_all is None:
            test_case_ids_all = t_ids; test_labels_all = t_labels
        else:
            assert t_ids == test_case_ids_all, f"Test case_id order mismatch for {m}"
            assert np.array_equal(t_labels, test_labels_all), f"Test labels mismatch for {m}"

        if has_external:
            if ext_case_ids_all is None:
                ext_case_ids_all = e_ids; ext_labels_all = e_labels
            else:
                assert e_ids == ext_case_ids_all, f"External case_id order mismatch for {m}"
                assert np.array_equal(e_labels, ext_labels_all), f"External labels mismatch for {m}"

        val_logits_by_mod[m] = v_logits
        test_logits_by_mod[m] = t_logits
        if has_external:
            ext_logits_by_mod[m] = e_logits

    # ---- Temperature calibration (per modality) on val ----
    temps = {}
    cal_val = []
    cal_test = []
    cal_ext  = [] if has_external else None
    for m in modalities:
        scaler = fit_temperature_scaler(val_logits_by_mod[m], val_labels_all)
        temps[m] = scaler.temperature()

        dev = scaler.log_T.device  # CPU
        v_logits_t = torch.from_numpy(val_logits_by_mod[m]).float().to(dev).view(-1, 1)
        t_logits_t = torch.from_numpy(test_logits_by_mod[m]).float().to(dev).view(-1, 1)

        cal_v = scaler(v_logits_t).squeeze(1).detach().cpu().numpy()
        cal_t = scaler(t_logits_t).squeeze(1).detach().cpu().numpy()

        cal_val.append(cal_v)
        cal_test.append(cal_t)

        if has_external:
            e_logits = ext_logits_by_mod[m]
            e_logits_t = torch.from_numpy(e_logits).float().to(dev).view(-1, 1)
            cal_e = scaler(e_logits_t).squeeze(1).detach().cpu().numpy()
            cal_ext.append(cal_e)

    # Stack calibrated logits (N, M)
    X_val_logits = np.stack(cal_val, axis=1)
    X_test_logits = np.stack(cal_test, axis=1)
    X_ext_logits  = np.stack(cal_ext, axis=1) if has_external and len(cal_ext) else None

    # ---- Baseline LR stacker (optionally standardized) ----
    X_val_lr = X_val_logits.copy()
    X_test_lr = X_test_logits.copy()
    X_ext_lr  = X_ext_logits.copy() if X_ext_logits is not None else None
    scaler_std = None
    if getattr(cfg, 'fusion_standardize', True):
        scaler_std = StandardScaler().fit(X_val_lr)
        X_val_lr = scaler_std.transform(X_val_lr)
        X_test_lr = scaler_std.transform(X_test_lr)
        if X_ext_lr is not None and X_ext_lr.shape[0] > 0:
            X_ext_lr = scaler_std.transform(X_ext_lr)

    penalty = getattr(cfg, 'fusion_penalty', 'l2')
    C = getattr(cfg, 'fusion_C', 1.0)
    lr = LogisticRegression(max_iter=1000,
                            penalty=('none' if penalty == 'none' else 'l2'),
                            C=C, solver='lbfgs',
                            random_state=seed)
    lr.fit(X_val_lr, val_labels_all)

    outdir = cfg.fusion_outdir
    os.makedirs(outdir, exist_ok=True)

    # ---- LR evaluation + threshold (VAL) using chosen mode ----
    val_prob_lr = lr.predict_proba(X_val_lr)[:, 1]
    test_prob_lr = lr.predict_proba(X_test_lr)[:, 1]
    ext_prob_lr  = (lr.predict_proba(X_ext_lr)[:, 1] if X_ext_lr is not None and X_ext_lr.shape[0] > 0 else None)

    lr_thr, lr_val_sens, lr_val_spec = choose_threshold(
        val_labels_all, val_prob_lr, mode, target_sens, target_tpr, target_fpr
    )
    with open(os.path.join(outdir, 'chosen_threshold_LR.txt'), 'w') as f:
        f.write(f"ThresholdMode={mode}\n")
        f.write(f"target_sensitivity={target_sens:.2f}\n")
        f.write(f"target_tpr={target_tpr:.2f}\n")
        f.write(f"target_fpr={target_fpr:.2f}\n")
        f.write(f"chosen_threshold={lr_thr:.6f}\n")
        f.write(f"val_sensitivity_at_T={lr_val_sens:.4f}\n")
        f.write(f"val_specificity_at_T={lr_val_spec:.4f}\n")

    metrics_and_plots(val_labels_all,  val_prob_lr, outdir, 'val_fused_LR',
                      threshold_for_report=lr_thr, target_sens=target_sens, mode_used=mode)
    metrics_and_plots(test_labels_all, test_prob_lr, outdir, 'test_fused_LR',
                      threshold_for_report=lr_thr, target_sens=target_sens, mode_used=mode)
    if ext_prob_lr is not None:
        metrics_and_plots(ext_labels_all,  ext_prob_lr, outdir, 'external_fused_LR',
                          threshold_for_report=lr_thr, target_sens=target_sens, mode_used=mode)

    # ---- Weighted fusion search (on calibrated logits) ----
    weights_result = None
    if enable_weight_search:
        # initial weights in the order of modalities (if provided as dict)
        init_w = None
        if isinstance(init_weights_cfg, dict):
            init_w = [float(init_weights_cfg.get(m, 1.0)) for m in modalities]

        weights_result = search_weights_for_target(
            X_val_logits, val_labels_all, mode=mode, target_sens=target_sens,
            target_tpr=target_tpr, target_fpr=target_fpr,
            trials=trials, bounds=bounds, normalize=normalize_weights, seed=seed,
            initial_weights=init_w
        )

        w_best = weights_result['weights']
        thr_best = weights_result['threshold']

        # Save weights
        weights_named = {m: float(w_best[i]) for i, m in enumerate(modalities)}
        with open(os.path.join(outdir, 'weighted_fusion.json'), 'w') as f:
            json.dump({
                'modalities': modalities,
                'weights': weights_named,
                'normalize_weights': normalize_weights,
                'bounds': bounds,
                'trials': trials,
                'ThresholdMode': mode,
                'target_sensitivity': target_sens,
                'target_tpr': target_tpr,
                'target_fpr': target_fpr,
                'val_sensitivity_at_T': weights_result['val_sens'],
                'val_specificity_at_T': weights_result['val_spec'],
                'chosen_threshold': thr_best
            }, f, indent=2)

        # Apply weights to VAL/TEST(/EXTERNAL) → fused probabilities
        val_prob_w = _fused_prob_from_weights(X_val_logits, w_best)
        test_prob_w = _fused_prob_from_weights(X_test_logits, w_best)
        ext_prob_w  = (_fused_prob_from_weights(X_ext_logits, w_best)
                       if has_external and X_ext_logits is not None and X_ext_logits.shape[0] > 0 else None)

        # Metrics and predictions for weighted fusion
        metrics_and_plots(val_labels_all,  val_prob_w, outdir, 'val_fused_WEIGHTED',
                          threshold_for_report=thr_best, target_sens=target_sens, mode_used=mode)
        metrics_and_plots(test_labels_all, test_prob_w, outdir, 'test_fused_WEIGHTED',
                          threshold_for_report=thr_best, target_sens=target_sens, mode_used=mode)
        if ext_prob_w is not None:
            metrics_and_plots(ext_labels_all, ext_prob_w, outdir, 'external_fused_WEIGHTED',
                              threshold_for_report=thr_best, target_sens=target_sens, mode_used=mode)

        # Save per-case predictions for weighted fusion
        pd.DataFrame({
            'case_id': val_case_ids_all, 'label': val_labels_all, 'fused_prob_weighted': val_prob_w
        }).to_csv(os.path.join(outdir, 'val_predictions_weighted.csv'), index=False)
        pd.DataFrame({
            'case_id': test_case_ids_all, 'label': test_labels_all, 'fused_prob_weighted': test_prob_w
        }).to_csv(os.path.join(outdir, 'test_predictions_weighted.csv'), index=False)
        if ext_prob_w is not None:
            pd.DataFrame({
                'case_id': ext_case_ids_all, 'label': ext_labels_all, 'fused_prob_weighted': ext_prob_w
            }).to_csv(os.path.join(outdir, 'external_predictions_weighted.csv'), index=False)

    # ---- MR-only fusion (DWI, ADC, FLAIR) ----
    target_mr_names = getattr(cfg, 'mr_modalities', ['DWI', 'ADC', 'FLAIR'])

    def _resolve_modalities(requested, available):
        # exact (case-insensitive) first, then substring (case-insensitive)
        avail_lower = {m.lower(): m for m in available}
        resolved = []
        for r in requested:
            r_low = r.lower()
            if r_low in avail_lower:
                resolved.append(avail_lower[r_low]); continue
            cand = [m for m in available if r_low in m.lower()]
            if len(cand) > 0:
                resolved.append(cand[0])
        seen, out = set(), []
        for m in resolved:
            if m not in seen:
                out.append(m); seen.add(m)
        return out

    mr_modalities = _resolve_modalities(target_mr_names, modalities)
    if len(mr_modalities) >= 2:
        print(f"[MR fusion] Using modalities: {mr_modalities}")

        # Build per-modality calibrated logits dicts
        cal_val_by_mod  = {m: cal_val[i]  for i, m in enumerate(modalities)}
        cal_test_by_mod = {m: cal_test[i] for i, m in enumerate(modalities)}
        cal_ext_by_mod  = ({m: cal_ext[i] for i, m in enumerate(modalities)} if has_external else {})

        # Stack MR-only calibrated logits
        X_val_mr  = np.stack([cal_val_by_mod[m]  for m in mr_modalities], axis=1)
        X_test_mr = np.stack([cal_test_by_mod[m] for m in mr_modalities], axis=1)
        X_ext_mr  = (np.stack([cal_ext_by_mod[m] for m in mr_modalities], axis=1)
                     if has_external and len(cal_ext_by_mod) else None)

        # Optional standardization
        X_val_mr_lr  = X_val_mr.copy()
        X_test_mr_lr = X_test_mr.copy()
        X_ext_mr_lr  = X_ext_mr.copy() if X_ext_mr is not None else None
        mr_scaler_std = None
        if getattr(cfg, 'fusion_standardize', True):
            mr_scaler_std = StandardScaler().fit(X_val_mr_lr)
            X_val_mr_lr  = mr_scaler_std.transform(X_val_mr_lr)
            X_test_mr_lr = mr_scaler_std.transform(X_test_mr_lr)
            if X_ext_mr_lr is not None and X_ext_mr_lr.shape[0] > 0:
                X_ext_mr_lr = mr_scaler_std.transform(X_ext_mr_lr)

        # Train LR stacker on VAL (MR-only)
        penalty = getattr(cfg, 'fusion_penalty', 'l2')
        C = getattr(cfg, 'fusion_C', 1.0)
        lr_mr = LogisticRegression(max_iter=1000,
                                   penalty=('none' if penalty == 'none' else 'l2'),
                                   C=C, solver='lbfgs',
                                   random_state=seed)
        lr_mr.fit(X_val_mr_lr, val_labels_all)

        # Evaluate & choose threshold on VAL
        val_prob_lr_mr  = lr_mr.predict_proba(X_val_mr_lr)[:, 1]
        test_prob_lr_mr = lr_mr.predict_proba(X_test_mr_lr)[:, 1]
        ext_prob_lr_mr  = (lr_mr.predict_proba(X_ext_mr_lr)[:, 1]
                           if X_ext_mr_lr is not None and X_ext_mr_lr.shape[0] > 0 else None)

        mr_lr_thr, mr_lr_val_sens, mr_lr_val_spec = choose_threshold(
            val_labels_all, val_prob_lr_mr, mode, target_sens, target_tpr, target_fpr
        )
        with open(os.path.join(outdir, 'chosen_threshold_LR_MR.txt'), 'w') as f:
            f.write(f"ThresholdMode={mode}\n")
            f.write(f"target_sensitivity={target_sens:.2f}\n")
            f.write(f"target_tpr={target_tpr:.2f}\n")
            f.write(f"target_fpr={target_fpr:.2f}\n")
            f.write(f"chosen_threshold={mr_lr_thr:.6f}\n")
            f.write(f"val_sensitivity_at_T={mr_lr_val_sens:.4f}\n")
            f.write(f"val_specificity_at_T={mr_lr_val_spec:.4f}\n")

        metrics_and_plots(val_labels_all,  val_prob_lr_mr,  outdir, 'val_fused_LR_MR',
                          threshold_for_report=mr_lr_thr, target_sens=target_sens, mode_used=mode)
        metrics_and_plots(test_labels_all, test_prob_lr_mr, outdir, 'test_fused_LR_MR',
                          threshold_for_report=mr_lr_thr, target_sens=target_sens, mode_used=mode)
        if ext_prob_lr_mr is not None:
            metrics_and_plots(ext_labels_all, ext_prob_lr_mr, outdir, 'external_fused_LR_MR',
                              threshold_for_report=mr_lr_thr, target_sens=target_sens, mode_used=mode)

        # Weighted fusion search (MR-only)
        mr_weights_result = None
        if enable_weight_search:
            init_w_mr = None
            if isinstance(init_weights_cfg, dict):
                init_w_mr = [float(init_weights_cfg.get(m, 1.0)) for m in mr_modalities]

            mr_weights_result = search_weights_for_target(
                X_val_mr, val_labels_all, mode=mode, target_sens=target_sens,
                target_tpr=target_tpr, target_fpr=target_fpr,
                trials=trials, bounds=bounds, normalize=normalize_weights, seed=seed,
                initial_weights=init_w_mr
            )

            w_best_mr   = mr_weights_result['weights']
            thr_best_mr = mr_weights_result['threshold']

            # Save MR weights
            weights_named_mr = {m: float(w_best_mr[i]) for i, m in enumerate(mr_modalities)}
            with open(os.path.join(outdir, 'weighted_fusion_MR.json'), 'w') as f:
                json.dump({
                    'modalities': mr_modalities,
                    'weights': weights_named_mr,
                    'normalize_weights': normalize_weights,
                    'bounds': bounds,
                    'trials': trials,
                    'ThresholdMode': mode,
                    'target_sensitivity': target_sens,
                    'target_tpr': target_tpr,
                    'target_fpr': target_fpr,
                    'val_sensitivity_at_T': mr_weights_result['val_sens'],
                    'val_specificity_at_T': mr_weights_result['val_spec'],
                    'chosen_threshold': thr_best_mr
                }, f, indent=2)

            # Apply weights → fused probabilities (MR-only)
            val_prob_w_mr  = _fused_prob_from_weights(X_val_mr,  w_best_mr)
            test_prob_w_mr = _fused_prob_from_weights(X_test_mr, w_best_mr)
            ext_prob_w_mr  = (_fused_prob_from_weights(X_ext_mr, w_best_mr)
                               if has_external and X_ext_mr is not None and X_ext_mr.shape[0] > 0 else None)

            metrics_and_plots(val_labels_all,  val_prob_w_mr,  outdir, 'val_fused_WEIGHTED_MR',
                              threshold_for_report=thr_best_mr, target_sens=target_sens, mode_used=mode)
            metrics_and_plots(test_labels_all, test_prob_w_mr, outdir, 'test_fused_WEIGHTED_MR',
                              threshold_for_report=thr_best_mr, target_sens=target_sens, mode_used=mode)
            if ext_prob_w_mr is not None:
                metrics_and_plots(ext_labels_all, ext_prob_w_mr, outdir, 'external_fused_WEIGHTED_MR',
                                  threshold_for_report=thr_best_mr, target_sens=target_sens, mode_used=mode)

            # Save per-case predictions (MR weighted)
            pd.DataFrame({
                'case_id': val_case_ids_all, 'label': val_labels_all, 'fused_prob_weighted_MR': val_prob_w_mr
            }).to_csv(os.path.join(outdir, 'val_predictions_weighted_MR.csv'), index=False)
            pd.DataFrame({
                'case_id': test_case_ids_all, 'label': test_labels_all, 'fused_prob_weighted_MR': test_prob_w_mr
            }).to_csv(os.path.join(outdir, 'test_predictions_weighted_MR.csv'), index=False)
            if ext_prob_w_mr is not None:
                pd.DataFrame({
                    'case_id': ext_case_ids_all, 'label': ext_labels_all, 'fused_prob_weighted_MR': ext_prob_w_mr
                }).to_csv(os.path.join(outdir, 'external_predictions_weighted_MR.csv'), index=False)

        # Save MR per-case predictions (per-modality calibrated + LR fused)
        mr_val_df = pd.DataFrame({'case_id': val_case_ids_all, 'label': val_labels_all})
        mr_test_df = pd.DataFrame({'case_id': test_case_ids_all, 'label': test_labels_all})
        for m in mr_modalities:
            mr_val_df[f'{m}_prob_cal']  = sigmoid(cal_val_by_mod[m])
            mr_test_df[f'{m}_prob_cal'] = sigmoid(cal_test_by_mod[m])
        mr_val_df['fused_prob_LR_MR']  = val_prob_lr_mr
        mr_test_df['fused_prob_LR_MR'] = test_prob_lr_mr
        mr_val_df.to_csv(os.path.join(outdir, 'val_predictions_MR.csv'), index=False)
        mr_test_df.to_csv(os.path.join(outdir, 'test_predictions_MR.csv'), index=False)

        if has_external:
            mr_ext_df = pd.DataFrame({'case_id': ext_case_ids_all, 'label': ext_labels_all})
            for m in mr_modalities:
                mr_ext_df[f'{m}_prob_cal'] = sigmoid(cal_ext_by_mod[m])
            if ext_prob_lr_mr is not None:
                mr_ext_df['fused_prob_LR_MR'] = ext_prob_lr_mr
            mr_ext_df.to_csv(os.path.join(outdir, 'external_predictions_MR.csv'), index=False)

        # Persist MR LR (and weighted) models
        joblib.dump(
            {'lr': lr_mr, 'scaler': mr_scaler_std, 'temps': {m: temps[m] for m in mr_modalities},
             'modalities': mr_modalities, 'target_shape': tuple(cfg.target_shape),
             'input_cha': cfg.input_channels, 'chosen_threshold_LR': mr_lr_thr,
             'threshold_mode': mode, 'target_sensitivity': target_sens,
             'target_tpr': target_tpr, 'target_fpr': target_fpr},
            os.path.join(outdir, 'fusion_model_LR_MR.joblib')
        )
        if 'mr_weights_result' in locals() and mr_weights_result is not None:
            joblib.dump(
                {'weights': mr_weights_result['weights'], 'normalize': normalize_weights,
                 'chosen_threshold': mr_weights_result['threshold'], 'threshold_mode': mode,
                 'target_sensitivity': target_sens, 'target_tpr': target_tpr, 'target_fpr': target_fpr,
                 'modalities': mr_modalities, 'temps': {m: temps[m] for m in mr_modalities}},
                os.path.join(outdir, 'fusion_model_WEIGHTED_MR.joblib')
            )
    else:
        print(f"[MR fusion] Skipped: unable to resolve enough MR modalities from {target_mr_names}. Found={mr_modalities}")

    # ---- Always save temperatures + per-modality calibrated probs + LR probs ----
    with open(os.path.join(outdir, 'temperatures.txt'), 'w') as f:
        for m in modalities:
            f.write(f'{m}\tT={temps[m]:.4f}\n')

    val_df = pd.DataFrame({'case_id': val_case_ids_all, 'label': val_labels_all})
    test_df = pd.DataFrame({'case_id': test_case_ids_all, 'label': test_labels_all})
    for i, m in enumerate(modalities):
        val_df[f'{m}_prob_cal'] = sigmoid(cal_val[i])
        test_df[f'{m}_prob_cal'] = sigmoid(cal_test[i])
    val_df['fused_prob_LR'] = val_prob_lr
    test_df['fused_prob_LR'] = test_prob_lr
    val_df.to_csv(os.path.join(outdir, 'val_predictions.csv'), index=False)
    test_df.to_csv(os.path.join(outdir, 'test_predictions.csv'), index=False)

    if has_external:
        ext_df = pd.DataFrame({'case_id': ext_case_ids_all, 'label': ext_labels_all})
        for i, m in enumerate(modalities):
            ext_probs_m = sigmoid(cal_ext[i]) if cal_ext and len(cal_ext) == len(modalities) else np.array([])
            ext_df[f'{m}_prob_cal'] = ext_probs_m
        if ext_prob_lr is not None:
            ext_df['fused_prob_LR'] = ext_prob_lr
        ext_df.to_csv(os.path.join(outdir, 'external_predictions.csv'), index=False)

    # ---- Persist models (all-modality) ----
    joblib.dump(
        {'lr': lr, 'scaler': scaler_std, 'temps': temps, 'modalities': modalities,
         'target_shape': tuple(cfg.target_shape), 'input_cha': cfg.input_channels,
         'chosen_threshold_LR': lr_thr, 'threshold_mode': mode,
         'target_sensitivity': target_sens, 'target_tpr': target_tpr, 'target_fpr': target_fpr},
        os.path.join(outdir, 'fusion_model_LR.joblib')
    )

    if weights_result is not None:
        joblib.dump(
            {'weights': weights_result['weights'], 'normalize': normalize_weights,
             'chosen_threshold': weights_result['threshold'], 'threshold_mode': mode,
             'target_sensitivity': target_sens, 'target_tpr': target_tpr, 'target_fpr': target_fpr,
             'modalities': modalities, 'temps': temps},
            os.path.join(outdir, 'fusion_model_WEIGHTED.joblib')
        )

    # ======== SHAP & Contribution Plots ========
    try:
        import shap
        _have_shap = True
    except Exception as e:
        print("[warn] SHAP not available:", e)
        _have_shap = False

    def _ensure_dir(p):
        os.makedirs(p, exist_ok=True)
        return p

    def _save_matplotlib(figpath):
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(figpath, dpi=200, bbox_inches="tight")
        plt.close()

    def make_lr_shap_plots(
        lr_model, X_val_lr, X_eval_lr, feature_names, outdir, tag="LR"
    ):
        """
        Creates SHAP summary plots for the LR stacker.
        X_*_lr should match what the LR saw (standardized if you used a StandardScaler).
        """
        if not _have_shap:
            return
        _ensure_dir(outdir)

        # modest background for stability
        bg = X_val_lr if X_val_lr.shape[0] <= 200 else X_val_lr[:200]

        try:
            explainer = shap.Explainer(lr_model, bg, feature_names=feature_names)
        except Exception:
            explainer = shap.LinearExplainer(lr_model, bg, feature_names=feature_names)

        shap_vals = explainer(X_eval_lr)

        # Beeswarm & bar
        try:
            shap.plots.beeswarm(shap_vals, show=False, max_display=len(feature_names))
            _save_matplotlib(os.path.join(outdir, f"shap_beeswarm_{tag}.png"))
        except Exception as e:
            print("[warn] SHAP beeswarm failed:", e)
        try:
            shap.plots.bar(shap_vals, show=False, max_display=len(feature_names))
            _save_matplotlib(os.path.join(outdir, f"shap_bar_{tag}.png"))
        except Exception as e:
            print("[warn] SHAP bar failed:", e)

        # Save raw arrays
        try:
            np.save(os.path.join(outdir, f"shap_values_{tag}.npy"), shap_vals.values)
            np.save(os.path.join(outdir, f"shap_base_{tag}.npy"), shap_vals.base_values)
            pd.DataFrame(X_eval_lr, columns=feature_names).to_csv(
                os.path.join(outdir, f"shap_features_{tag}.csv"), index=False
            )
        except Exception as e:
            print("[warn] Failed to save SHAP arrays:", e)

    def weighted_fusion_contributions(
        X_ref, X_eval, weights, feature_names, outdir, tag="WEIGHTED"
    ):
        """
        For a linear logit model f(x) = sigmoid(w·x), contributions on log-odds axis:
        ϕ_i ≈ w_i * (x_i - E[x_i]). We compute contributions per sample and save:
          - mean(|contribution|) bar chart
          - CSV with per-sample contributions
        """
        _ensure_dir(outdir)
        w = np.asarray(weights, dtype=float).reshape(-1)
        X_ref = np.asarray(X_ref, dtype=float)
        X_eval = np.asarray(X_eval, dtype=float)

        if X_ref.size == 0 or X_eval.size == 0:
            print(f"[warn] Skipping contributions for {tag}: empty arrays.")
            return

        mu = X_ref.mean(axis=0)  # background expectation
        contrib = (X_eval - mu) * w  # log-odds contributions per feature

        # Plot mean absolute contribution per feature
        mean_abs = np.mean(np.abs(contrib), axis=0)
        order = np.argsort(mean_abs)[::-1]
        plt.figure(figsize=(6, 4))
        plt.bar(range(len(feature_names)), mean_abs[order])
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in order], rotation=45, ha="right")
        plt.ylabel("Mean |contribution| (log-odds)")
        plt.title(f"Per-modality contributions — {tag}")
        _save_matplotlib(os.path.join(outdir, f"weighted_contrib_bar_{tag}.png"))

        # Save per-sample contributions
        pd.DataFrame(contrib, columns=feature_names).to_csv(
            os.path.join(outdir, f"weighted_contrib_per_sample_{tag}.csv"), index=False
        )

    # LR SHAP (TEST and EXTERNAL if available)
    if _have_shap:
        try:
            make_lr_shap_plots(
                lr_model=lr,
                X_val_lr=X_val_lr,
                X_eval_lr=X_test_lr,
                feature_names=modalities,
                outdir=outdir,
                tag="LR_all_modalities_TEST",
            )
        except Exception as e:
            print("[warn] SHAP for LR TEST failed:", e)
        if X_ext_lr is not None and X_ext_lr.shape[0] > 0:
            try:
                make_lr_shap_plots(
                    lr_model=lr,
                    X_val_lr=X_val_lr,
                    X_eval_lr=X_ext_lr,
                    feature_names=modalities,
                    outdir=outdir,
                    tag="LR_all_modalities_EXTERNAL",
                )
            except Exception as e:
                print("[warn] SHAP for LR EXTERNAL failed:", e)

    # Weighted fusion contributions (TEST and EXTERNAL)
    if 'weights_result' in locals() and weights_result is not None:
        try:
            weighted_fusion_contributions(
                X_ref=X_val_logits,
                X_eval=X_test_logits,
                weights=weights_result['weights'],
                feature_names=modalities,
                outdir=outdir,
                tag="all_modalities_TEST",
            )
        except Exception as e:
            print("[warn] Weighted-fusion TEST contribution plot failed:", e)

        if has_external and X_ext_logits is not None and X_ext_logits.shape[0] > 0:
            try:
                weighted_fusion_contributions(
                    X_ref=X_val_logits,
                    X_eval=X_ext_logits,
                    weights=weights_result['weights'],
                    feature_names=modalities,
                    outdir=outdir,
                    tag="all_modalities_EXTERNAL",
                )
            except Exception as e:
                print("[warn] Weighted-fusion EXTERNAL contribution plot failed:", e)

    # MR-only SHAP/Contrib
    if 'mr_modalities' in locals() and len(mr_modalities) >= 2:
        # LR on MR subset
        if _have_shap and 'lr_mr' in locals():
            try:
                make_lr_shap_plots(
                    lr_model=lr_mr,
                    X_val_lr=X_val_mr_lr,
                    X_eval_lr=X_test_mr_lr,
                    feature_names=mr_modalities,
                    outdir=outdir,
                    tag="LR_MR_only_TEST",
                )
            except Exception as e:
                print("[warn] SHAP for LR_MR TEST failed:", e)
            if X_ext_mr_lr is not None and X_ext_mr_lr.shape[0] > 0:
                try:
                    make_lr_shap_plots(
                        lr_model=lr_mr,
                        X_val_lr=X_val_mr_lr,
                        X_eval_lr=X_ext_mr_lr,
                        feature_names=mr_modalities,
                        outdir=outdir,
                        tag="LR_MR_only_EXTERNAL",
                    )
                except Exception as e:
                    print("[warn] SHAP for LR_MR EXTERNAL failed:", e)

        # Weighted MR subset
        if 'mr_weights_result' in locals() and mr_weights_result is not None:
            try:
                weighted_fusion_contributions(
                    X_ref=X_val_mr,
                    X_eval=X_test_mr,
                    weights=mr_weights_result['weights'],
                    feature_names=mr_modalities,
                    outdir=outdir,
                    tag="MR_only_TEST",
                )
            except Exception as e:
                print("[warn] MR weighted TEST contribution plot failed:", e)
            if has_external and X_ext_mr is not None and X_ext_mr.shape[0] > 0:
                try:
                    weighted_fusion_contributions(
                        X_ref=X_val_mr,
                        X_eval=X_ext_mr,
                        weights=mr_weights_result['weights'],
                        feature_names=mr_modalities,
                        outdir=outdir,
                        tag="MR_only_EXTERNAL",
                    )
                except Exception as e:
                    print("[warn] MR weighted EXTERNAL contribution plot failed:", e)

    print("Done. Outputs saved to", outdir)


if __name__ == '__main__':
    main()
