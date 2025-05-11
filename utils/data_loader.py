import os
import json
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from utils.preprocess import compute_global_min_max, normalize, interpolate_df
from utils.loading_data import load_excel  # Load and format an Excel file


class ExcelDataset(Dataset):
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

        sequence = torch.tensor(df_final.values.astype(np.float32))
        label = torch.tensor(label)

        return sequence, label
