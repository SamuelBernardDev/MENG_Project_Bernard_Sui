import os
import glob
import torch
import json
import yaml
import pandas as pd
from src.models.LSTMModel import LSTMClassifier
from utils.loading_data import load_excel
from utils.preprocess import normalize, interpolate_df

# === Load config and stats ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

stats_path = config["data"]["stats_path"]
with open(stats_path) as f:
    stats = json.load(f)

global_min = pd.Series(stats["min"])
global_max = pd.Series(stats["max"])
min_seq_len = stats["min_seq_len"]
columns = config["data"]["columns"]
time_format = "%I:%M:%S%p"  # or config["data"].get("time_format", ...)

# === Load trained model ===
model = LSTMClassifier(
    input_size=len(columns),
    hidden_size=config["model"]["hidden_size"],
    num_layers=config["model"]["num_layers"],
    num_classes=2,  # brush vs fasted
)
model.load_state_dict(torch.load(config["train"]["model_save_path"]))
model.eval()

# === Test files ===
test_root = "data/test"
test_files = glob.glob(os.path.join(test_root, "*/*.xls"))

# === Inference loop ===
print("🧪 Running inference on test files:\n")
for file_path in test_files:
    # --- Preprocess single file ---
    df = load_excel(file_path, columns, time_format)
    df_norm = normalize(df, global_min, global_max)
    df_interp = interpolate_df(df_norm, interval=1, max_length=min_seq_len)

    sequence = torch.tensor(df_interp.values.astype("float32")).unsqueeze(
        0
    )  # shape [1, T, F]

    # --- Predict ---
    with torch.no_grad():
        output = model(sequence)  # [1, num_classes]
        pred = torch.argmax(output, dim=1).item()

    label_str = "Brush" if pred == 0 else "Fasted"
    print(f"{file_path}: 🧠 Predicted class = {label_str}")
