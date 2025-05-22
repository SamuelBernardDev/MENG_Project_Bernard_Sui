import os
import yaml
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils.data_loader import ExcelDatasetTimeSeries
import wandb

# === Load config ===
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# === Initialize Weights & Biases ===
wandb.init(project="rf-breath-analysis", config=config)

# === Initialize dataset ===
dataset = ExcelDatasetTimeSeries(
    root_folder=config["data"]["root_folder"],
    columns=config["data"]["columns"],
    stats_path=config["data"]["stats_path"],
    use_stats=config["data"]["use_stats"],
    min_required_length=config["data"]["min_required_length"],
    derivative_columns=config["data"].get("derivative_columns", [])
)

# === Load all data into arrays ===
X = []
y = []

for i in range(len(dataset)):
    try:
        sequence, label = dataset[i]
        X.append(sequence.numpy().flatten())
        y.append(label.item())
    except Exception as e:
        print(f"Skipping index {i} due to error: {e}")

X = np.array(X)
y = np.array(y)

# === K-Fold Cross Validation ===
kf = KFold(n_splits=4, shuffle=True, random_state=42)
fold_val_accuracies = []
all_importances = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    fold_val_accuracies.append(acc)
    all_importances.append(model.feature_importances_)

    print(f"Fold {fold + 1} Validation Accuracy: {acc:.2%}")
    wandb.log({f"fold_{fold + 1}/val_accuracy": acc})

# === Final Summary ===
mean_acc = sum(fold_val_accuracies) / len(fold_val_accuracies)
print(f"\n✅ Cross-validation complete. Mean Validation Accuracy: {mean_acc:.2%}")
wandb.log({"crossval_mean_val_accuracy": mean_acc})

# === Feature Importance ===
avg_importances = np.mean(all_importances, axis=0)
T, F = dataset[0][0].shape  # get shape of one sample
feature_names = config["data"]["columns"] + [f"{col}_rate" for col in config["data"].get("derivative_columns", [])]
expanded_names = [f"{name}_t{t}" for t in range(T) for name in feature_names]

# === Rank and plot top 20 ===
indices = np.argsort(avg_importances)[::-1][:20]
top_features = [expanded_names[i] for i in indices]
top_importances = avg_importances[indices]

plt.figure(figsize=(12, 6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.title("Top 20 Most Important Features")
plt.xlabel("Average Importance")
plt.tight_layout()
plt.show()

# === Log to wandb
wandb.log({f"feature_importance/{name}": float(imp) for name, imp in zip(top_features, top_importances)})