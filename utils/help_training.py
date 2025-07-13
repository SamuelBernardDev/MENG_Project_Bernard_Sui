import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from utils.data_loader import ExcelDatasetTimeSeries
from src.models.LSTMModel import LSTMClassifier
from src.models.ConvSeqModel import ConvSequenceClassifier


def build_model(model_type: str, input_size: int, num_classes: int, config: dict):
    """Create sequence model based on configuration."""
    if model_type == "lstm":
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            num_classes=num_classes,
        )
    elif model_type == "conv":
        return ConvSequenceClassifier(
            input_size=input_size,
            num_classes=num_classes,
            conv_channels=config["model"].get("conv_channels", 32),
            conv_dilations=config["model"].get("conv_dilations", [1, 2, 4]),
            rnn_hidden=config["model"].get("hidden_size", 64),
            rnn_layers=config["model"].get("num_layers", 1),
            sequence_module=config["model"].get("sequence_module", "lstm"),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for seqs, labels in loader:
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, threshold=0.5, return_probs=False):
    """Evaluate model on given loader."""
    model.eval()
    val_loss = 0.0
    correct = 0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for seqs, labels in loader:
            outputs = model(seqs)
            probs = F.softmax(outputs, dim=1)[:, 1]
            all_probs.append(probs.item())
            all_targets.append(labels.item())
            val_loss += criterion(outputs, labels).item()
            preds = (probs >= threshold).long()
            correct += (preds == labels).sum().item()
    avg_val_loss = val_loss / len(loader)
    val_acc = correct / len(loader.dataset)
    try:
        auroc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auroc = float("nan")
    if return_probs:
        return avg_val_loss, val_acc, auroc, (all_probs, all_targets)
    return avg_val_loss, val_acc, auroc


def cross_validate(model_type: str, config: dict, indices, labels):
    """Run 4-fold cross validation and return best threshold."""
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    fold_aurocs = []
    fold_thresholds = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices, labels), 1):
        stats_file = f"stats_fold{fold}.json"
        # compute stats on training subset
        ExcelDatasetTimeSeries(
            root_folder=config["data"]["root_folder"],
            columns=config["data"]["columns"],
            derivative_columns=config["data"].get("derivative_columns", []),
            min_required_length=config["data"].get("min_required_length", 50),
            stats_path=stats_file,
            is_train=True,
            indices=train_idx,
        )

        train_ds = ExcelDatasetTimeSeries(
            root_folder=config["data"]["root_folder"],
            columns=config["data"]["columns"],
            derivative_columns=config["data"].get("derivative_columns", []),
            min_required_length=config["data"].get("min_required_length", 50),
            stats_path=stats_file,
            is_train=False,
            indices=train_idx,
        )
        val_ds = ExcelDatasetTimeSeries(
            root_folder=config["data"]["root_folder"],
            columns=config["data"]["columns"],
            derivative_columns=config["data"].get("derivative_columns", []),
            min_required_length=config["data"].get("min_required_length", 50),
            stats_path=stats_file,
            is_train=False,
            indices=val_idx,
        )
        train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        input_size = len(config["data"]["columns"]) + len(config["data"].get("derivative_columns", []))
        num_classes = len(train_ds.label_map)
        model = build_model(model_type, input_size, num_classes, config)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=1e-5)

        best_auc = 0.0
        best_threshold = 0.5
        patience = config["train"].get("early_stopping_patience", 20)
        min_delta = config["train"].get("early_stopping_min_delta", 0.001)
        patience_counter = 0

        for epoch in range(1, config["train"]["epochs"] + 1):
            train_epoch(model, train_loader, criterion, optimizer)
            _, val_acc, auroc, (probs, targets) = evaluate(
                model, val_loader, criterion, return_probs=True
            )
            from sklearn.metrics import roc_curve

            fpr, tpr, thresholds = roc_curve(targets, probs)
            youden = tpr - fpr
            optimal_threshold = thresholds[int(np.argmax(youden))]

            print(
                f"Fold {fold} Epoch {epoch:02d} | Val Acc: {val_acc:.2%} | AUROC: {auroc:.4f}"
            )

            if auroc - best_auc > min_delta:
                best_auc = auroc
                best_threshold = optimal_threshold
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping on fold {fold} at epoch {epoch}")
                break
        fold_aurocs.append(best_auc)
        fold_thresholds.append(best_threshold)
        os.remove(stats_file)
    mean_auroc = np.nanmean(fold_aurocs)
    mean_threshold = float(np.nanmean(fold_thresholds))
    print(f"\nCross-validation mean AUROC: {mean_auroc:.4f}")
    print(f"Mean optimal threshold: {mean_threshold:.4f}\n")
    return mean_threshold


def train_full(model_type: str, config: dict, threshold: float = 0.5):
    """Train selected model on the entire dataset and save weights."""
    ExcelDatasetTimeSeries(
        root_folder=config["data"]["root_folder"],
        columns=config["data"]["columns"],
        derivative_columns=config["data"].get("derivative_columns", []),
        min_required_length=config["data"].get("min_required_length", 50),
        stats_path=config["data"]["stats_path"],
        is_train=True,
    )

    dataset = ExcelDatasetTimeSeries(
        root_folder=config["data"]["root_folder"],
        columns=config["data"]["columns"],
        derivative_columns=config["data"].get("derivative_columns", []),
        min_required_length=config["data"].get("min_required_length", 50),
        stats_path=config["data"]["stats_path"],
        is_train=False,
    )

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split = int(len(indices) * config["train"].get("val_ratio", 0.2))
    val_idx = indices[:split]
    train_idx = indices[split:]

    train_ds = ExcelDatasetTimeSeries(
        root_folder=config["data"]["root_folder"],
        columns=config["data"]["columns"],
        derivative_columns=config["data"].get("derivative_columns", []),
        min_required_length=config["data"].get("min_required_length", 50),
        stats_path=config["data"]["stats_path"],
        is_train=False,
        indices=train_idx,
    )
    val_ds = ExcelDatasetTimeSeries(
        root_folder=config["data"]["root_folder"],
        columns=config["data"]["columns"],
        derivative_columns=config["data"].get("derivative_columns", []),
        min_required_length=config["data"].get("min_required_length", 50),
        stats_path=config["data"]["stats_path"],
        is_train=False,
        indices=val_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    input_size = len(config["data"]["columns"]) + len(config["data"].get("derivative_columns", []))
    num_classes = len(train_ds.label_map)
    model = build_model(model_type, input_size, num_classes, config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=1e-5)

    best_auc = 0.0
    patience = config["train"].get("early_stopping_patience", 20)
    min_delta = config["train"].get("early_stopping_min_delta", 0.001)
    patience_counter = 0

    for epoch in range(1, config["train"]["epochs"] + 1):
        train_epoch(model, train_loader, criterion, optimizer)
        _, val_acc, auroc = evaluate(model, val_loader, criterion, threshold)

        print(
            f"Epoch {epoch:02d} | Val Acc: {val_acc:.2%} | AUROC: {auroc:.4f}"
        )

        if auroc - best_auc > min_delta:
            best_auc = auroc
            patience_counter = 0
            torch.save(model.state_dict(), config["train"]["model_save_path"])
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch} due to no AUROC improvement greater than {min_delta}"
            )
            break

    torch.save(model.state_dict(), config["train"]["model_save_path"])
    print(f"Model saved to {config['train']['model_save_path']}")
