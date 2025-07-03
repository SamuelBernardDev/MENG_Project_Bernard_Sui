import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from utils.data_loader import ExcelDatasetTimeSeries

# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# === Initialize dataset (preprocessing happens in __getitem__) ===
dataset = ExcelDatasetTimeSeries(
    root_folder=config["data"]["root_folder"],
    columns=config["data"]["columns"],
    stats_path=config["data"]["stats_path"],
    is_train=True,
    min_required_length=config["data"]["min_required_length"],
    derivative_columns=config["data"].get("derivative_columns", []),
    indices=None,  # Use full dataset unless a specific split is needed
)

# === Load all sequences into arrays ===
X_data = []
y_labels = []

root_folder = config["data"]["root_folder"]

all_paths, all_labels = [], {}
subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
subfolders.sort()

for idx, folder in enumerate(subfolders):
    print("Label:", idx, "Folder:", folder)
    all_labels[idx] = os.path.basename(folder)

for i in range(len(dataset)):
    try:
        sequence, label = dataset[i]
        X_data.append(sequence.numpy().flatten())  # flatten [T, F] → [T*F]
        y_labels.append(label.item())
    except Exception as e:
        print(f"Skipping index {i} due to error: {e}")

X_data = np.array(X_data)
y_labels = np.array(y_labels)

print(f"Loaded {len(X_data)} samples for PCA.")

# === PCA Reduction ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_data)

# === Plot PCA result ===
plt.figure(figsize=(10, 6))
for label in np.unique(y_labels):
    idx = y_labels == label
    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=f"Class {all_labels[label]}",
        alpha=0.6,
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Preprocessed Time-Series Sequences")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
