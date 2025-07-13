import os
import json
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from utils.preprocess import (
    compute_global_min_max,
    normalize,
    interpolate_df,
    contains_outliers,
    iqr_filter,
    rolling_mean,
    log_transform,
)
from utils.loading_data import load_excel  # Load and format an Excel file


class ExcelDatasetTimeSeries(Dataset):
    def __init__(
        self,
        root_folder,
        columns,
        time_format="%I:%M:%S%p",
        stats_path=None,
        derivative_columns=None,
        min_required_length=50,
        is_train=True,
        indices=None,
        smoothing_window=None,
        iqr_factor=None,
        log_columns=None,
    ):
        """
        root_folder: base data directory
        columns: list of columns to load
        stats_path: path to JSON file for saving/loading stats
        derivative_columns: columns for which to compute diff-rate
        min_required_length: minimum sequence length (in seconds)
        is_train: if True, compute & save stats on selected files;
                  if False, load stats (no recompute)
        indices: list of file indices to include (for train/val split)
        """
        self.columns = columns
        self.derivative_columns = derivative_columns or []
        self.time_format = time_format
        self.stats_path = stats_path
        self.is_train = is_train
        self.min_required_length = min_required_length
        self.indices = indices
        self.smoothing_window = smoothing_window
        self.iqr_factor = iqr_factor
        self.log_columns = log_columns or []

    def _apply_transforms(self, df):
        for c in self.derivative_columns:
            if c in df.columns:
                df[f"{c}_rate"] = df[c].diff().fillna(0)
        if self.smoothing_window and self.smoothing_window > 1:
            cols = self.columns + [f"{c}_rate" for c in self.derivative_columns]
            existing = [c for c in cols if c in df.columns]
            df[existing] = rolling_mean(df[existing], self.smoothing_window)
        if self.log_columns:
            log_transform(df, self.log_columns)
        if self.iqr_factor is not None:
            iqr_filter(df, self.iqr_factor)
        return df

        # initial scan: list all files and base labels
        all_paths, all_labels = [], []
        subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
        subfolders.sort()
        self.label_map = {}
        for idx, folder in enumerate(subfolders):
            print("Label:", idx, "Folder:", folder)
            label = idx  # use folder index as label
            self.label_map[os.path.basename(folder).lower()] = label
            for fpath in glob.glob(os.path.join(folder, "*.xls")):
                df = load_excel(fpath, self.columns, self.time_format)
                if df.empty or df.index.max() < self.min_required_length:
                    continue
                all_paths.append(fpath)
                all_labels.append(label)
        # restrict to provided indices or take all
        if self.indices is not None:
            self.file_paths = [all_paths[i] for i in self.indices]
            self.labels = [all_labels[i] for i in self.indices]
        else:
            self.file_paths = all_paths
            self.labels = all_labels

        # if simply listing files (no stats requested for validation), exit early
        if not self.is_train and self.stats_path is None:
            return

        # prepare stats
        self.global_min = {}
        self.global_max = {}
        self.min_seq_len = float("inf")
        all_features = self.columns + [f"{c}_rate" for c in self.derivative_columns]

        if self.is_train:
            # compute stats on training files only
            for fpath in self.file_paths:
                try:
                    df = load_excel(fpath, self.columns, self.time_format)
                    df = self._apply_transforms(df)
                    seq_len = df.index.max()
                    if seq_len < self.min_required_length:
                        continue
                    for feat in all_features:
                        if feat in df.columns:
                            vals = df[feat]
                            self.global_min[feat] = min(
                                self.global_min.get(feat, vals.min()), vals.min()
                            )
                            self.global_max[feat] = max(
                                self.global_max.get(feat, vals.max()), vals.max()
                            )
                    self.min_seq_len = 40  # min(self.min_seq_len, seq_len)
                except Exception:
                    continue
            # finalize and save
            self.global_min = pd.Series(self.global_min)
            self.global_max = pd.Series(self.global_max)
            self.min_seq_len = int(self.min_seq_len)
            if self.stats_path:
                with open(self.stats_path, "w") as sf:
                    json.dump(
                        {
                            "min": self.global_min.to_dict(),
                            "max": self.global_max.to_dict(),
                            "min_seq_len": self.min_seq_len,
                        },
                        sf,
                    )
        else:
            # load stats for validation
            if self.stats_path and os.path.exists(self.stats_path):
                with open(self.stats_path) as sf:
                    stats = json.load(sf)
                self.global_min = pd.Series(stats["min"])
                self.global_max = pd.Series(stats["max"])
                self.min_seq_len = stats["min_seq_len"]
            else:
                raise FileNotFoundError(f"Stats file not found: {self.stats_path}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fpath = self.file_paths[idx]
        label = self.labels[idx]
        df = load_excel(fpath, self.columns, self.time_format)
        df = self._apply_transforms(df)

        # normalize and interpolate
        df_norm = normalize(df, self.global_min, self.global_max)
        df_final = interpolate_df(df_norm, interval=1, max_length=self.min_seq_len)

        seq = torch.tensor(df_final.values, dtype=torch.float32)
        lbl = torch.tensor(label, dtype=torch.long)
        return seq, lbl


# For Stephanie to modify as needed
# For example, if you want to add a grayscale transformation, you can do it here.
class ExcelDatasetCNN(Dataset):
    def __init__(
        self,
        root_folder,
        columns,
        time_format="%I:%M:%S%p",
        stats_path=None,
        use_stats=True,
    ):
        self.columns = columns
        self.time_format = time_format
        self.stats_path = stats_path
        self.use_stats = use_stats

        # Assign labels based on folder names
        subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
        subfolders.sort()
        self.label_map = {
            os.path.basename(folder).lower(): i for i, folder in enumerate(subfolders)
        }

        self.file_paths = []
        self.labels = []
        for folder, label in self.label_map.items():
            full_folder = os.path.join(root_folder, folder)
            files = glob.glob(os.path.join(full_folder, "*.xls"))
            self.file_paths.extend(files)
            self.labels.extend([label] * len(files))

        # === Normalization stats: use or compute ===
        if self.stats_path and use_stats and os.path.exists(self.stats_path):
            with open(self.stats_path) as f:
                stats = json.load(f)
            self.global_min = pd.Series(stats["min"])
            self.global_max = pd.Series(stats["max"])
            self.min_seq_len = stats["min_seq_len"]
        else:
            self.global_min, self.global_max = compute_global_min_max(
                self.file_paths, columns
            )
            self.min_seq_len = self._compute_min_sequence_length()
            if self.stats_path:
                with open(self.stats_path, "w") as f:
                    json.dump(
                        {
                            "min": self.global_min.to_dict(),
                            "max": self.global_max.to_dict(),
                            "min_seq_len": self.min_seq_len,
                        },
                        f,
                    )

    def _compute_min_sequence_length(self):
        min_len = float("inf")
        for file in self.file_paths:
            df = load_excel(file, self.columns, self.time_format)
            duration = df.index.max()
            if duration < min_len:
                min_len = duration
        return int(min_len)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file = self.file_paths[idx]
        label = self.labels[idx]

        df = load_excel(file, self.columns, self.time_format)
        df_norm = normalize(df, self.global_min, self.global_max)
        df_final = interpolate_df(df_norm, interval=1, max_length=self.min_seq_len)
        # grayscale transformation
        sequence = torch.tensor(df_final.values.astype(np.float32))
        label = torch.tensor(label)

        return sequence, label
