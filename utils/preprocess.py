import numpy as np
import pandas as pd
from utils.loading_data import load_excel
import torch


def compute_global_min_max(file_paths, columns):
    dfs = [pd.read_excel(fp)[columns] for fp in file_paths]
    concat_df = pd.concat(dfs)
    return concat_df.min(), concat_df.max()


def normalize(df, global_min, global_max):
    return (df - global_min) / (global_max - global_min)


def interpolate_df(df, interval=1, max_length=None):
    """
    Interpolates a DataFrame on a uniform time grid.

    Args:
        df (pd.DataFrame): Input dataframe with time index (seconds).
        interval (int): Time step in seconds (default = 1s).
        max_length (int or None): If set, truncate to this number of seconds.

    Returns:
        pd.DataFrame: Interpolated and optionally truncated dataframe.
    """
    max_time = df.index.max()
    if max_length is not None:
        end_time = min(max_time, max_length)
    else:
        end_time = max_time

    common_times = np.arange(0, end_time, interval)
    return df.reindex(common_times).interpolate().fillna(0)


def grayscale(df):
    """
    Convert a DataFrame to grayscale by averaging the columns.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Grayscale dataframe.
    """
    return df.mean(axis=1, keepdims=True)


def preprocess_single_file(
    file_path, columns, global_min, global_max, time_format="%I:%M:%S%p"
):
    df = load_excel(file_path, columns, time_format)
    df_norm = normalize(df, global_min, global_max)
    df_interp = interpolate_df(df_norm)
    return torch.tensor(df_interp.values.astype("float32")).unsqueeze(
        0
    )  # shape [1, T, F]
