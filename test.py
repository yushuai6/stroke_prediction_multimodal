import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import nibabel as nib
import matplotlib.pyplot as plt

from resnet3D import I3Res50forStrokeOutcome
from dataset import StrokeImageDataset
from config import (
    label_csv, data_dir, target_shape, batch_size,
    device, checkpoint_dir, model_dir
)


# =========================
# Utility: infer modality
# =========================
def infer_modality_from_path(p: str) -> str:
    """
    Infer imaging modality from the folder name.
    e.g., '/path/to/DWI' → 'DWI'
    """
    name = os.path.basename(os.path.normpath(p))
    return name if name else "unknown"


# =========================
# Grad-CAM for 3D CNN
# =========================
class GradCAM3D:
    """
    Grad-CAM for 3D CNNs using a forward hook on the target layer.
    Produces normalized activation maps per sample.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        self.activations = out

        def _save_grad(grad):
            self.gradients = grad
        out.register_hook(_save_grad)

    def remove_hooks(self):
        if self.fwd_handle is not None:
            self.fwd_handle.remove()
            self.fwd_handle = None

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Grad-CAM maps for positive logits.
        Args:
            logits: tensor of shape (B, 1)
        Returns:
            CAM tensor of shape (B, D, H, W), normalized to [0,1]
        """
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError(f"Expected logits shape (B, 1), got {tuple(logits.shape)}")

        self.model.zero_grad(set_to_none=True)
        logits[:, 0].sum().backward(retain_graph=True)

        A = self.activations      # (B, C, D, H, W)
        dA = self.gradients       # (B, C, D, H, W)
        weights = dA.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * A).sum(dim=1))  # (B, D, H, W)

        # Normalize each sample
        B = cam.size(0)
        for i in range(B):
            cmin, cmax = cam[i].min(), cam[i].max()
            if torch.isfinite(cmin) and torch.isfinite(cmax) and (cmax - cmin) > 0:
                cam[i] = (cam[i] - cmin) / (cmax - cmin + 1e-8)
            else:
                cam[i] = torch.zeros_like(cam[i])
        return cam


# =========================
# Model loading helper
# =========================
def load_model(model_path: str) -> I3Res50forStrokeOutcome:
    """
    Load pretrained model weights for evaluation.
    """
    model = I3Res50forStrokeOutcome(input_cha=1, num_classes=1)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and any(k in state for k in ["state_dict", "model", "model_state_dict"]):
        state = state.get("model_state_dict", state.get("state_dict", state.get("model")))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


# =========================
# Evaluation + Grad-CAM
# =========================
def run_test_and_cam(model, dataloader, cams_outdir):
    """
    Perform inference and Grad-CAM visualization.
    Generates heatmaps for true positives and saves:
      - NIfTI (.nii.gz) CAM volumes
      - PNG overlays for the max-activation slice

    Returns:
        labels, preds, probs, ids
    """
    os.makedirs(cams_outdir, exist_ok=True)
    target_layer = model.layer3[-1].conv3
    cam_engine = GradCAM3D(model, target_layer=target_layer)

    all_probs, all_preds, all_labels, all_ids = [], [], [], []

    for batch in tqdm(dataloader, desc="Running Test + CAM"):
        images, labels, patients = batch  # dataset must have return_meta=True
        images, labels = images.to(device), labels.to(device)

        with torch.enable_grad():
            logits = model(images)
            probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).int()
        all_probs.extend(probs.detach().cpu().numpy().flatten())
        all_preds.extend(preds.detach().cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        all_ids.extend(list(patients))

        # Generate CAMs for true positives
        tp_mask = ((labels == 1) & (preds == 1)).squeeze(1)
        if tp_mask.any():
            cam_maps = cam_engine(logits).detach().cpu()
            cam_maps_up = F.interpolate(
                cam_maps.unsqueeze(1),
                size=images.shape[2:],  # (Z, Y, X)
                mode="trilinear",
                align_corners=False
            ).squeeze(1)

            imgs_cpu = images.detach().cpu().squeeze(1)
            for i in range(cam_maps_up.shape[0]):
                if not tp_mask[i]:
                    continue
                patient_id = patients[i]
                cam_zyx = cam_maps_up[i].numpy().astype(np.float32)
                img_zyx = imgs_cpu[i].numpy().astype(np.float32)

                # Save CAM as NIfTI
                nii_cam = nib.Nifti1Image(cam_zyx, affine=np.eye(4))
                nib.save(nii_cam, os.path.join(cams_outdir, f"{patient_id}_cam.nii.gz"))

                # Save overlay PNG at max activation slice
                z_idx = int(np.argmax(cam_zyx.max(axis=(1, 2))))
                base = (img_zyx[z_idx] - img_zyx[z_idx].min()) / (img_zyx[z_idx].ptp() + 1e-8)
                heat = cam_zyx[z_idx]

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(base, cmap="gray", interpolation="nearest")
                ax.imshow(heat, alpha=0.45, interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"{patient_id} | TP slice z={z_idx}")
                plt.tight_layout()
                plt.savefig(os.path.join(cams_outdir, f"{patient_id}_cam_z{z_idx}.png"),
                            dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

    cam_engine.remove_hooks()
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_ids


# =========================
# Main script
# =========================
def main():
    # Load dataset
    test_dataset = StrokeImageDataset(
        csv_path=label_csv,
        image_dir=data_dir,
        split="external",
        target_shape=target_shape,
        return_meta=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load checkpoint
    candidate_paths = [os.path.join(checkpoint_dir, model_dir)]
    model_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(f"No checkpoint found in: {candidate_paths}")

    model = load_model(model_path)
    print(f"✅ Model loaded from {model_path}")

    # Prepare output directories
    cams_outdir = os.path.join(checkpoint_dir, "cams_tp")
    modality = infer_modality_from_path(data_dir)
    results_dir = os.path.join(checkpoint_dir, f"results_{modality}")
    os.makedirs(results_dir, exist_ok=True)

    labels, preds, probs, ids = run_test_and_cam(model, test_loader, cams_outdir)

    # =========================
    # Compute metrics
    # =========================
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    try:
        auc_val = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    except Exception as e:
        print(f"⚠️ AUC computation failed: {e}")
        auc_val = float("nan")

    print("\n--- Test Results ---")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"AUC:         {auc_val:.4f}" if np.isfinite(auc_val) else "AUC: nan")
    print(f"✅ CAMs saved to {cams_outdir}")

    # =========================
    # Export results
    # =========================
    df_results = pd.DataFrame({
        "patient": ids,
        "label": labels.astype(int),
        "prediction": preds.astype(int),
        "probability": probs.astype(float)
    })

    # (1) Probabilities
    prob_path = os.path.join(results_dir, "test_probabilities.xlsx")
    try:
        df_results[["patient", "label", "probability"]].to_excel(prob_path, index=False)
        print(f"✅ Probabilities saved to {prob_path}")
    except Exception as e:
        csv_path = prob_path.replace(".xlsx", ".csv")
        df_results[["patient", "label", "probability"]].to_csv(csv_path, index=False)
        print(f"⚠️ Excel unavailable ({e}); saved CSV instead: {csv_path}")

    # (2) Full predictions
    df_results.to_csv(os.path.join(results_dir, "test_predictions_full.csv"), index=False)

    # (3) Confusion matrix
    cm_df = pd.DataFrame(
        confusion_matrix(labels, preds, labels=[0, 1]),
        index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"]
    )
    cm_df.to_csv(os.path.join(results_dir, "confusion_matrix.csv"))

    # (4) Scalar metrics
    pd.DataFrame([{
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "auc": auc_val
    }]).to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)

    # (5) ROC curve
    try:
        if np.isfinite(auc_val):
            fpr, tpr, thr = roc_curve(labels, probs)
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
                os.path.join(results_dir, "roc_curve.csv"), index=False
            )
            print("✅ ROC curve exported.")
        else:
            print("⚠️ Skipped ROC export (AUC undefined).")
    except Exception as e:
        print(f"⚠️ ROC export failed: {e}")


if __name__ == "__main__":
    main()
