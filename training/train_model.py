import torch
from torch.utils.data import DataLoader, random_split
from src.models.LSTMModel import LSTMClassifier
from utils.data_loader import ExcelDatasetTimeSeries
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import yaml
import wandb

# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# === Initialize Weights & Biases ===
# wandb.init(project="lstm-breath-analysis", config=config)


# initialize dataset to get total count
full_ds = ExcelDatasetTimeSeries(
    root_folder=config["data"]["root_folder"],
    columns=config["data"]["columns"],
    derivative_columns=config["data"]["derivative_columns"],
    stats_path=None,
    is_train=False,
    indices=None,
)
all_indices = list(range(len(full_ds)))

kf = KFold(n_splits=4, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices), 1):
    print(f"\n=== Fold {fold} ===")
    stats_file = f"stats_fold{fold}.json"

    # create train/val datasets
    train_ds = ExcelDatasetTimeSeries(
        root_folder=config["data"]["root_folder"],
        columns=config["data"]["columns"],
        derivative_columns=config["data"]["derivative_columns"],
        stats_path=stats_file,
        is_train=True,
        indices=train_idx,
    )
    val_ds = ExcelDatasetTimeSeries(
        root_folder=config["data"]["root_folder"],
        columns=config["data"]["columns"],
        derivative_columns=config["data"]["derivative_columns"],
        stats_path=stats_file,
        is_train=False,
        indices=val_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["train"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # model, loss, optimizer
    model = LSTMClassifier(
        input_size=len(config["data"]["columns"])
        + len(config["data"]["derivative_columns"]),
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=len(train_ds.label_map),
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["learning_rate"], weight_decay=1e-5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    # training epochs
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

        # validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_ds)

        print(
            f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        elif val_acc >= 0.80:
            patience_counter += 1
            if patience_counter >= 10:
                print(
                    f"Early stopping at epoch {epoch + 1} due to no improvement in val loss."
                )
                break

    fold_accuracies.append(val_acc)
    # save model
    save_path = config["train"]["model_save_path"].format(fold)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

mean_acc = sum(fold_accuracies) / len(fold_accuracies)
print(f"\nCross-validation complete. Mean Val Acc: {mean_acc:.2%}")
