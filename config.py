import os
import torch

# =========================
# Core training config
# =========================
# Primary (single-modality) training defaults
data_dir = r"data/image/MR_DWI"
label_csv = r"data/labels/label_dict.csv"
checkpoint_dir = r"checkpoints"
model_dir = r"checkpoints/best_models/DWI_best.pth"

num_epochs = 500
batch_size = 6
learning_rate = 1e-4
input_channels = 1
num_classes = 1  # Binary classification
split_column = "split"

# Shape of preprocessed volumes (Z, Y, X)
target_shape = (30, 256, 256)

early_stop_patience = 100
lr_scheduler_patience = 5
lr_scheduler_factor = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Fusion config (multi-modal)
# =========================
base_data_dir = r"data"
base_model_dir = r"models"

label_csv = fr"{base_model_dir}/label_dict.csv"

# Modalities used in multimodal fusion
fusion_modalities = ["DWI", "FLAIR", "ADC", "NonContrast"]

fusion_image_dirs = {
    "DWI":         fr"{base_data_dir}/image/MR_DWI",
    "FLAIR":       fr"{base_data_dir}/image/MR_FLAIR",
    "ADC":         fr"{base_data_dir}/image/MR_ADC",
    "NonContrast": fr"{base_data_dir}/image/NCCT",
}

fusion_ckpts = {
    "DWI":         fr"{base_model_dir}/checkpoints/best_models/DWI_best.pth",
    "FLAIR":       fr"{base_model_dir}/checkpoints/best_models/FLAIR_best.pth",
    "ADC":         fr"{base_model_dir}/checkpoints/best_models/ADC_best.pth",
    "NonContrast": fr"{base_model_dir}/checkpoints/best_models/NCCT_best.pth",
}

val_split = "val"
test_split = "test"

num_workers = 4  # For DataLoader

fusion_outdir = fr"{base_model_dir}/results_fusion"

# =========================
# Logistic Regression (LR) stacker settings
# =========================
fusion_standardize = True
fusion_penalty = "l2"
fusion_C = 1.0

# =========================
# Threshold selection (for fusion_final.py)
# =========================
fusion_threshold_mode = "youden"
fusion_target_sensitivity = 0.80
fusion_target_tpr = 0.80
fusion_target_fpr = 0.35

# =========================
# Weighted fusion search (logit-space)
# =========================
fusion_weight_search = True
fusion_weight_trials = 2000
fusion_weight_bounds = (0.0, 2.5)
fusion_weight_normalize = True
fusion_seed = 42

# =========================
# Clinical data (for fusion_final.py)
# =========================
clinical_xlsx = fr"{base_model_dir}/clinical_records.xlsx"

clinical_id_candidates = ["patient", "Identity", "case_id"]
clinical_label_col = "90dmRS"
mrs_bad_threshold = 3  # mRS >= 3 → poor outcome (1)

# =========================
# Imaging subset discovery
# =========================
mr_fused_col = None
mr_modalities = ["DWI", "ADC", "FLAIR"]

ncct_col = None
ncct_token = "NonContrast"
