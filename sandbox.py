# Trial code for testing the functionality of the codebase
import pandas as pd
import numpy as np
import os
import glob

liste = [1, 2, 3, 4, 5]
df = pd.DataFrame(liste, columns=["a"])
print(df)
print(df["a"].values)
print("hello world")


all_paths, all_labels = [], []
subfolders = [f.path for f in os.scandir("data/raw") if f.is_dir()]
subfolders.sort()
for idx, folder in enumerate(subfolders):
    label = idx
    print(label)
    for fpath in glob.glob(os.path.join(folder, "*.xls")):
        all_paths.append(fpath)
        all_labels.append(label)
print(all_labels)
