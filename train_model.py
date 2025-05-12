import torch
from torch.utils.data import DataLoader, random_split
from src.models.model import LSTMClassifier
from utils.data_loader import ExcelDataset
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
dataset = ExcelDataset(
    root_folder=config["data"]["root_folder"],
    columns=config["data"]["columns"],
    stats_path=config["data"]["stats_path"],
    use_stats=config["data"]["use_stats"],
)

# === Split into training and validation sets ===
val_ratio = config["train"].get("val_ratio", 0.25)
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_set, batch_size=config["train"]["batch_size"], shuffle=True
)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# === Initialize model ===
model = LSTMClassifier(
    input_size=len(config["data"]["columns"]),
    hidden_size=config["model"]["hidden_size"],
    num_layers=config["model"]["num_layers"],
    num_classes=len(dataset.label_map),
)

# === Loss and optimizer ===
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])


kf = KFold(n_splits=4, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n🔁 Fold {fold + 1}")

    # Create DataLoaders for this fold
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set, batch_size=config["train"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # Initialize a new model for each fold
    model = LSTMClassifier(
        input_size=len(config["data"]["columns"]),
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        num_classes=len(dataset.label_map),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["learning_rate"]
    )

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

        # Validation phase
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
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {val_acc:.2%}"
        )

        wandb.log(
            {
                f"fold_{fold + 1}/epoch": epoch + 1,
                f"fold_{fold + 1}/train_loss": avg_train_loss,
                f"fold_{fold + 1}/val_loss": avg_val_loss,
                f"fold_{fold + 1}/val_accuracy": val_acc,
            }
        )

    # Save model per fold (optional)
    fold_model_path = config["train"]["model_save_path"].replace(
        ".pth", f"_fold{fold + 1}.pth"
    )
    torch.save(model.state_dict(), fold_model_path)
    print(f"✅ Fold {fold + 1} model saved to {fold_model_path}")
