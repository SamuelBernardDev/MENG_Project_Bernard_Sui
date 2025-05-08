import numpy as np
import pandas as pd


def compute_global_min_max(data_frames, columns):
    global_min, global_max = {}, {}
    for df in data_frames.values():
        for col in columns:
            col_min, col_max = df[col].min(), df[col].max()
            global_min[col] = min(global_min.get(col, col_min), col_min)
            global_max[col] = max(global_max.get(col, col_max), col_max)
    return global_min, global_max


def normalize_data(data_frames, columns, global_min, global_max):
    normalized = {}
    for fp, df in data_frames.items():
        df_norm = df.copy()
        for col in columns:
            mn, mx = global_min[col], global_max[col]
            df_norm[col] = (df[col] - mn) / (mx - mn) if mx != mn else 0.0
        normalized[fp] = df_norm
    return normalized


def interpolate_and_clip(normalized, columns):
    min_end = min(df.index.max() for df in normalized.values())
    common_times = np.unique(
        np.concatenate([df.index.values for df in normalized.values()])
    )
    common_times = common_times[common_times <= min_end]

    for fp, df in normalized.items():
        df = df[columns].reindex(common_times).interpolate()
        normalized[fp] = df

    return normalized, common_times
