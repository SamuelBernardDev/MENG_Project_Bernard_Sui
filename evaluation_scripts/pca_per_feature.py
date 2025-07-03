import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.data_loader import ExcelDatasetTimeSeries

# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# === Initialize dataset ===
dataset = ExcelDatasetTimeSeries(
    root_folder=config["data"]["root_folder"],
    columns=config["data"]["columns"],
    stats_path=config["data"]["stats_path"],
    min_required_length=config["data"]["min_required_length"],
    derivative_columns=config["data"].get("derivative_columns", []),
    is_train=True,  # Set to False if using for validation/inference
    indices=None,  # Use full dataset; pass list of indices for train/val split
)

# === Collect all sequences ===
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
        sequence, label = dataset[i]  # shape: [T, F]
        X_data.append(sequence.numpy())
        y_labels.append(label.item())
    except Exception as e:
        print(f"Skipping index {i} due to error: {e}")

X_data = np.array(X_data)  # shape: [N, T, F]
y_labels = np.array(y_labels)

if X_data.ndim != 3:
    raise ValueError("Expected 3D array [samples, time, features]")

N, T, F = X_data.shape
print(f"Loaded {N} sequences with {T} time steps and {F} features each.")

# === Setup ===
feature_names = config["data"]["columns"] + [
    f"{col}_rate" for col in config["data"].get("derivative_columns", [])
]
separation_scores = {}

# === PCA and plotting per feature ===
for i in range(F):
    X_feat = X_data[:, :, i]  # shape: [N, T]
    pca = PCA(n_components=2)
    try:
        X_pca = pca.fit_transform(X_feat)

        # Class separation score (distance between class means)
        classes = np.unique(y_labels)
        centroids = [X_pca[y_labels == c].mean(axis=0) for c in classes]
        score = np.linalg.norm(centroids[0] - centroids[1])
        separation_scores[feature_names[i]] = score

        # Plot
        plt.figure(figsize=(8, 6))
        for label in classes:
            idx = y_labels == label
            plt.scatter(
                X_pca[idx, 0],
                X_pca[idx, 1],
                label=f"Class {all_labels[label]}",
                alpha=0.6,
            )
        plt.title(f"{feature_names[i]} (Score: {score:.2f})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Failed PCA for feature {feature_names[i]}: {e}")

# === Ranking ===
print("\nPCA Separation Scores:")
ranked = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
for i, (name, score) in enumerate(ranked, 1):
    print(f"{i}. {name}: {score:.2f}")

print("\nClass Label Mapping (folder name → label index):")
for name, idx in dataset.label_map.items():
    print(f"  Label {idx}: {name}")
