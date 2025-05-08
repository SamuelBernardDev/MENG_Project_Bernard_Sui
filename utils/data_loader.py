import pandas as pd
from pathlib import Path


def load_excel_files(file_paths, time_format="%I:%M:%S%p"):
    data_frames = {}
    for fp in file_paths:
        df = pd.read_excel(fp)
        df["Time"] = pd.to_datetime(df["Time"], format=time_format)
        df["seconds"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
        df.set_index("seconds", inplace=True)
        data_frames[fp] = df
    return data_frames
