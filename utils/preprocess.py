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


def contains_outliers(df, factor=10.0):
    """
    Return True if any column has values outside the IQR bounds by a given factor.
    This detects extreme outliers only.
    """
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        if ((df[col] < lower_bound) | (df[col] > upper_bound)).any():
            return True
    return False


def iqr_filter(df, factor=1.5):
    """Replace values outside IQR bounds with NaN."""
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
    return df


def rolling_mean(df, window):
    """Apply a centered rolling mean."""
    return df.rolling(window, min_periods=1, center=True).mean()


def log_transform(df, columns):
    """Apply log1p transform to specified columns."""
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def preprocess_single_file(
    file_path, columns, global_min, global_max, time_format="%I:%M:%S%p"
):
    df = load_excel(file_path, columns, time_format)
    df_norm = normalize(df, global_min, global_max)
    df_interp = interpolate_df(df_norm)
    return torch.tensor(df_interp.values.astype("float32")).unsqueeze(
        0
    )  # shape [1, T, F]
