import torch
import json
import pandas as pd
from utils.preprocess import preprocess_single_file
from models.model import LSTMClassifier
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load normalization stats
with open("src/data/normalization_stats.json") as f:
    stats = json.load(f)

global_min = pd.Series(stats["min"])
global_max = pd.Series(stats["max"])

# File to test
file_path = "src/data/raw/test_fast_01.xls"

# Preprocess the file
sequence = preprocess_single_file(
    file_path,
    columns=config["data"]["columns"],
    global_min=global_min,
    global_max=global_max,
    time_format="%I:%M:%S%p",
)  # shape [1, T, F]

# Load model
model = LSTMClassifier(
    input_size=len(config["data"]["columns"]),
    hidden_size=64,
    num_layers=2,
    num_classes=2,
)
model.load_state_dict(torch.load(config["train"]["model_save_path"]))
model.eval()

# Run inference
with torch.no_grad():
    output = model(sequence)
    pred_class = torch.argmax(output, dim=1).item()
    print(f"Predicted class: {pred_class} ({'brush' if pred_class == 0 else 'fast'})")
