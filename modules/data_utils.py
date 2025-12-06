import os
import pandas as pd

def load_raw_data(data_path: str):
    """
    Tải dữ liệu thô từ thư mục (csv hoặc tsv).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    files = [f for f in os.listdir(data_path) if f.endswith(('.csv', '.tsv'))]
    if not files:
        raise FileNotFoundError("No CSV/TSV files found in dataset folder")

    dfs = []
    for f in files:
        full_path = os.path.join(data_path, f)
        if f.endswith(".csv"):
            df = pd.read_csv(full_path)
        else:
            df = pd.read_csv(full_path, sep="\t")
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)