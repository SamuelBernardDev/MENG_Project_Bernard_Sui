import pandas as pd


def load_excel(file, columns, time_format="%I:%M:%S%p"):
    df = pd.read_excel(file)
    df["Time"] = pd.to_datetime(df["Time"], format=time_format)
    df["seconds"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
    df.set_index("seconds", inplace=True)
    return df[columns]
