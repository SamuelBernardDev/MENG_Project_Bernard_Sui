import os
import glob
import torch
import yaml
import json
import pandas as pd
from src.models.LSTMModel import LSTMClassifier
from utils.loading_data import load_excel
from utils.preprocess import (
    normalize,
    interpolate_df,
    rolling_mean,
    iqr_filter,
    log_transform,
)

# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# === Load normalization stats ===
with open(config["data"]["stats_path"]) as f:
    stats = yaml.safe_load(f) if config["data"]["stats_path"].endswith(".yaml") else json.load(f)
global_min = pd.Series(stats["min"])
global_max = pd.Series(stats["max"])
min_seq_len = stats["min_seq_len"]

# === Load trained model ===
model = LSTMClassifier(
    input_size=len(config["data"]["columns"] + config["data"].get("derivative_columns", [])),
    hidden_size=config["model"]["hidden_size"],
    num_layers=config["model"]["num_layers"],
    num_classes=2  # Assuming binary classification
)
model.load_state_dict(torch.load(config["train"]["model_save_path"]))
model.eval()

# === Inference folder ===
test_folder = "data/test"
test_files = glob.glob(os.path.join(test_folder, "*/*.xls"))

# Map predicted indices to folder names for display
class_names = sorted(
    [d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))]
)

# === Predict on each file ===
for fpath in test_files:
    try:
        df = load_excel(fpath, config["data"]["columns"], time_format="%I:%M:%S%p")

        for col in config["data"].get("derivative_columns", []):
            if col in df.columns:
                df[f"{col}_rate"] = df[col].diff().fillna(0)
        if config["data"].get("smoothing_window"):
            cols = config["data"]["columns"] + [f"{c}_rate" for c in config["data"].get("derivative_columns", [])]
            existing = [c for c in cols if c in df.columns]
            df[existing] = rolling_mean(df[existing], config["data"]["smoothing_window"])
        if config["data"].get("log_columns"):
            log_transform(df, config["data"]["log_columns"])
        if config["data"].get("iqr_factor") is not None:
            iqr_filter(df, config["data"]["iqr_factor"])

        df_norm = normalize(df, global_min, global_max)
        df_final = interpolate_df(df_norm, interval=1, max_length=min_seq_len)
        sequence = torch.tensor(df_final.values.astype("float32")).unsqueeze(0)  # shape: [1, T, F]

        with torch.no_grad():
            output = model(sequence)
            probs = torch.softmax(output, dim=1).squeeze()
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()

        label = class_names[pred] if pred < len(class_names) else str(pred)
        print(
            f"File: {os.path.basename(fpath)} | Prediction: {label} | Confidence: {confidence:.2%}"
        )

    except Exception as e:
        print(f"Failed to process {os.path.basename(fpath)}: {e}")
