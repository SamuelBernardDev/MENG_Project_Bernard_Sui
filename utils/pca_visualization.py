import os
import glob
import yaml
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from utils.preprocess import compute_global_min_max, normalize, interpolate_df
from utils.loading_data import load_excel  # Load and format an Excel file


# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# === Load normalization stats ===
with open(config["data"]["stats_path"]) as f:
    stats = json.load(f)
global_min = pd.Series(stats["min"])
global_max = pd.Series(stats["max"])
min_seq_len = stats["min_seq_len"]

# === Load and preprocess all data ===
root_folder = config["data"]["root_folder"]
columns = config["data"]["columns"]
deriv_cols = config["data"].get("derivative_columns", [])
all_columns = columns + [f"{col}_rate" for col in deriv_cols]

data = []
labels = []

for label_folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, label_folder)
    if not os.path.isdir(folder_path):
        continue

    for fpath in glob.glob(os.path.join(folder_path, "*.xls")):
        try:
            df = load_excel(fpath, columns, time_format="%I:%M:%S%p")
            for col in deriv_cols:
                if col in df.columns:
                    df[f"{col}_rate"] = df[col].diff().fillna(0)

            df_norm = normalize(df, global_min, global_max)
            df_final = interpolate_df(df_norm, interval=1, max_length=min_seq_len)
            flat = df_final.values.flatten()
            data.append(flat)
            labels.append(label_folder)
        except Exception as e:
            print(f"Skipping {fpath} due to error: {e}")

# === Convert to arrays and apply PCA ===
X = np.array(data)
y = np.array(labels)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# === Plot PCA result ===
plt.figure(figsize=(10, 6))
for i, class_name in enumerate(le.classes_):
    plt.scatter(X_pca[y_encoded == i, 0], X_pca[y_encoded == i, 1], label=class_name, alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Preprocessed Breath Sensor Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()