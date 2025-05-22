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
wandb.init(project="lstm-breath-analysis", config=config)

# === Initialize dataset ===
dataset = ExcelDatasetTimeSeries(
    root_folder=config["data"]["root_folder"],
    columns=config["data"]["columns"],
    stats_path=config["data"]["stats_path"],
    use_stats=config["data"]["use_stats"],
    min_required_length=config["data"]["min_required_length"],
    derivative_columns=config["data"].get("derivative_columns", [])
)

kf = KFold(n_splits=4, shuffle=True, random_state=42)
fold_val_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\nFold {fold + 1}")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model = LSTMClassifier(
        input_size=len(config["data"]["columns"] + config["data"].get("derivative_columns", [])),
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=len(dataset.label_map),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    best_val_loss = float("inf")
    patience = config["train"].get("early_stopping_patience", 10)
    patience_counter = 0

    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_set)

        print(
            f"Fold [{fold + 1}] Epoch [{epoch + 1}/{config['train']['epochs']}], "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}"
        )

        wandb.log({
            f"fold_{fold + 1}/epoch": epoch + 1,
            f"fold_{fold + 1}/train_loss": avg_train_loss,
            f"fold_{fold + 1}/val_loss": avg_val_loss,
            f"fold_{fold + 1}/val_accuracy": val_acc,
        })

        # === Early stopping check ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        elif val_acc>=0.80:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in val loss.")
                break

    fold_val_accuracies.append(val_acc)

    fold_model_path = config["train"]["model_save_path"]#.replace(".pth", f"_fold{fold + 1}.pth")
    torch.save(model.state_dict(), fold_model_path)
    print(f"Fold {fold + 1} model saved to {fold_model_path}")

# === Final summary ===
mean_acc = sum(fold_val_accuracies) / len(fold_val_accuracies)
print(f"\n Cross-validation complete. Mean Validation Accuracy: {mean_acc:.2%}")
wandb.log({"crossval_mean_val_accuracy": mean_acc})
