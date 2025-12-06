import os
import pandas as pd

def load_raw_data(data_path="data"):
    files = [f for f in os.listdir(data_path) if f.endswith(".tsv.gz")]

    if len(files) == 0:
        raise FileNotFoundError("No IMDb TSV files found in data folder")

    dfs = []

    for f in files:
        file_path = os.path.join(data_path, f)
        print("Loading:", file_path)

        df = pd.read_csv(file_path, sep="\t", compression="gzip", low_memory=False)
        dfs.append(df)

    return dfs