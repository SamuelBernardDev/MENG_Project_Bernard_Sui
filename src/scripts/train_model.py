import torch
from torch.utils.data import DataLoader
from models.model import LSTMClassifier
from utils.data_loader import load_excel_files
from utils.preprocess import (
    compute_global_min_max,
    normalize_data,
    interpolate_and_clip,
)
from pathlib import Path
import yaml
import glob

# Load config
with open("config.yaml") as file:
    config = yaml.safe_load(file)

xls_files = glob.glob(config["data"]["xls_files"])
data_frames = load_excel_files(xls_files)
global_min, global_max = compute_global_min_max(
    data_frames, config["data"]["feature_columns"]
)
normalized, common_times = normalize_data(
    data_frames, config["data"]["feature_columns"], global_min, global_max
)
normalized, _ = interpolate_and_clip(normalized, config["data"]["feature_columns"])

# Prepare PyTorch Dataset (custom implementation)
# ... Define your PyTorch dataset class here

# Model setup
model = LSTMClassifier(
    input_size=len(config["data"]["feature_columns"]),
    hidden_sizes=[64, 32],
    num_classes=len(config["data"]["label_map"]),
)

# Training loop (basic structure)
# ... PyTorch training loop here
