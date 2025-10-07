import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from dataset import StrokeImageDataset
from resnet3D import i3_res50forStrokeOutcome
import config


# =========================
# Metric computation
# =========================
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor):
    """
    Compute sensitivity, specificity, and accuracy from predictions.
    """
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return sensitivity, specificity, accuracy


# =========================
# Evaluation loop
# =========================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            all_preds.append(preds)
            all_labels.append(labels.int())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    sens, spec, acc = compute_metrics(all_preds, all_labels)
    avg_loss = val_loss / len(dataloader)
    return avg_loss, sens, spec, acc


# =========================
# Model checkpoint utilities
# =========================
def save_model(model, metrics, epoch, save_dir, prefix="best"):
    """
    Save model weights and validation metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{prefix}_model_epoch{epoch + 1}.pth")
    metrics_path = os.path.join(save_dir, f"{prefix}_model_epoch{epoch + 1}_metrics.json")

    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


# =========================
# Main training loop
# =========================
def train():
    """
    Train a 3D ResNet model for binary stroke outcome prediction (favorable vs unfavorable).
    """
    model = i3_res50forStrokeOutcome(
        input_cha=config.input_channels,
        num_classes=config.num_classes
    ).to(config.device)

    # Datasets
    train_set = StrokeImageDataset(
        config.label_csv, config.data_dir, split="train", target_shape=config.target_shape
    )
    val_set = StrokeImageDataset(
        config.label_csv, config.data_dir, split="val", target_shape=config.target_shape
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience
    )

    best_val_loss = float("inf")
    patience_counter = 0

    train_loss_list, val_loss_list = [], []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            all_preds.append(preds)
            all_labels.append(labels.int())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        train_sens, train_spec, train_acc = compute_metrics(all_preds, all_labels)
        train_loss = total_loss / len(train_loader)
        train_loss_list.append(train_loss)

        val_loss, val_sens, val_spec, val_acc = evaluate(model, val_loader, criterion, config.device)
        val_loss_list.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[Epoch {epoch + 1}] LR: {current_lr:.2e}")
        print(f"  Train → Loss: {train_loss:.4f}, Sens: {train_sens:.3f}, Spec: {train_spec:.3f}, Acc: {train_acc:.3f}")
        print(f"  Val   → Loss: {val_loss:.4f}, Sens: {val_sens:.3f}, Spec: {val_spec:.3f}, Acc: {val_acc:.3f}")

        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "epoch": epoch + 1,
                "val_loss": float(val_loss),
                "val_sensitivity": float(val_sens),
                "val_specificity": float(val_spec),
                "val_accuracy": float(val_acc),
                "learning_rate": current_lr
            }
            save_model(model, best_metrics, epoch, config.checkpoint_dir, prefix="best")
            print("  ✅ Best model (val loss) saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ❌ No improvement. Patience counter: {patience_counter}/{config.early_stop_patience}")

        # Early stopping
        if patience_counter >= config.early_stop_patience:
            print("  ⏹ Early stopping triggered.")
            break

    # Plot and save loss curve
    plt.figure()
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label="Train Loss")
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    plot_path = os.path.join(config.checkpoint_dir, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"📉 Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    train()
