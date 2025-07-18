import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shutil

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from src.models.LSTMModel import LSTMClassifier
from utils.data_loader import ExcelDatasetTimeSeries

# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# === Load full dataset for stratified split ===
full_ds = ExcelDatasetTimeSeries(
    root_folder=config["data"]["root_folder"],
    columns=config["data"]["columns"],
    derivative_columns=config["data"]["derivative_columns"],
    smoothing_window=config["data"].get("smoothing_window"),
    iqr_factor=config["data"].get("iqr_factor"),
    log_columns=config["data"].get("log_columns"),
    stats_path=None,
    is_train=False,
    indices=None,
)

all_indices = list(range(len(full_ds)))
all_labels = [full_ds.labels[i] for i in all_indices]

# === Stratified K-Fold setup ===
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
fold_accuracies = []
fold_aurocs = []

# === For ROC plot ===
fold_fprs = []
fold_tprs = []

# === Fold loop ===
for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices, all_labels), 1):
    print(f"\n=== Fold {fold} ===")
    stats_file = f"stats_fold{fold}.json"

    train_ds = ExcelDatasetTimeSeries(
        root_folder=config["data"]["root_folder"],
        columns=config["data"]["columns"],
        derivative_columns=config["data"]["derivative_columns"],
        smoothing_window=config["data"].get("smoothing_window"),
        iqr_factor=config["data"].get("iqr_factor"),
        log_columns=config["data"].get("log_columns"),
        stats_path=stats_file,
        is_train=True,
        indices=train_idx,
    )
    val_ds = ExcelDatasetTimeSeries(
        root_folder=config["data"]["root_folder"],
        columns=config["data"]["columns"],
        derivative_columns=config["data"]["derivative_columns"],
        smoothing_window=config["data"].get("smoothing_window"),
        iqr_factor=config["data"].get("iqr_factor"),
        log_columns=config["data"].get("log_columns"),
        stats_path=stats_file,
        is_train=False,
        indices=val_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["train"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = LSTMClassifier(
        input_size=len(config["data"]["columns"])
        + len(config["data"]["derivative_columns"]),
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=len(train_ds.label_map),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train"]["learning_rate"], weight_decay=1e-5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_auc_for_patience = 0
    best_val_acc = 0
    best_auroc = float("-inf")
    patience = config["train"].get("early_stopping_patience", 20)
    min_delta = config["train"].get("early_stopping_min_delta", 0.001)

    best_metric = float("-inf")
    best_model_path = (
        config["train"]["model_save_path"].format(fold).replace(".pth", f"_fold{fold}_best.pth")
    )
    best_stats_path = stats_file.replace(".json", f"_fold{fold}_best.json")
    best_epoch = 0
    # === Training loop ===
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        total_loss = 0
        for seqs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # === Validation loop ===
        model.eval()
        val_loss = 0
        correct = 0
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for seqs, labels in val_loader:
                outputs = model(seqs)
                probs = F.softmax(outputs, dim=1)[:, 1]
                all_probs.append(probs.item())
                all_targets.append(labels.item())

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_ds)

        try:
            auroc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auroc = float("nan")

        print(
            f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%} | AUROC: {auroc:.4f}"
        )

        # Save best model based on combined metric
        current_metric = val_acc + (auroc if not np.isnan(auroc) else 0)
        if current_metric > best_metric:
            best_metric = current_metric
            best_val_acc = val_acc
            best_auroc = auroc
            torch.save(model.state_dict(), best_model_path)
            shutil.copy(stats_file, best_stats_path)
            best_epoch = epoch

        if auroc - best_auc_for_patience > min_delta:
            best_auc_for_patience = auroc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} due to no AUROC improvement "
                f"greater than {min_delta} for {patience} epochs."
            )
            break

        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        # elif val_acc >= 0.80:
        #     patience_counter += 1
        #     if patience_counter >= 12:
        #         print(
        #             f"Early stopping at epoch {epoch + 1} due to no improvement in val loss."
        #         )
        #         break

    fold_accuracies.append(best_val_acc)
    fold_aurocs.append(best_auroc)

    # === Collect fold-level ROC data
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    fold_fprs.append(fpr)
    fold_tprs.append(tpr)


    # === Save model
    save_path = config["train"]["model_save_path"].format(fold)
    shutil.copy(best_model_path, save_path)
    print(f"Best model from epoch {best_epoch} saved to {save_path}")

# === Summary Metrics ===
mean_acc = np.mean(fold_accuracies)
mean_auroc = np.nanmean(fold_aurocs)

# === Final ROC plot
plt.figure(figsize=(10, 7))
for i, (fpr, tpr, auc) in enumerate(zip(fold_fprs, fold_tprs, fold_aurocs), 1):
    plt.plot(fpr, tpr, label=f"Fold {i} (AUROC = {auc:.2f})", alpha=0.7)

plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.plot([], [], " ", label=f"Mean AUROC = {mean_auroc:.2f}")
plt.title("ROC Curves by Fold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_all_folds.png")
print("\nAll-fold ROC curve saved to roc_curve_all_folds.png")

print(f"\nCross-validation complete.")
print(f"Mean Val Accuracy: {mean_acc:.2%}")
print(f"Mean AUROC (fold-wise): {mean_auroc:.4f}")
