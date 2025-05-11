import torch
from torch.utils.data import DataLoader
from src.models.model import LSTMClassifier
from utils.data_loader import ExcelDataset  # make sure this is the updated version
import yaml

# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# === Initialize dataset ===
dataset = ExcelDataset(
    root_folder=config["data"]["root_folder"],
    columns=config["data"]["columns"],
    stats_path=config["data"]["stats_path"],
)

# === Create DataLoader ===
train_loader = DataLoader(
    dataset, batch_size=config["train"]["batch_size"], shuffle=True
)

# === Initialize model ===
model = LSTMClassifier(
    input_size=len(config["data"]["columns"]),
    hidden_size=config["model"]["hidden_size"],
    num_layers=config["model"]["num_layers"],
    num_classes=len(dataset.label_map),  # Dynamically get number of classes
)

# === Optimizer and loss ===
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

# === Training loop ===
for epoch in range(config["train"]["epochs"]):
    model.train()
    total_loss = 0
    for sequences, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences)  # shape: [batch_size, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{config['train']['epochs']}], Loss: {avg_loss:.4f}")

# === Save model ===
torch.save(model.state_dict(), config["train"]["model_save_path"])
print("✅ Model saved.")
