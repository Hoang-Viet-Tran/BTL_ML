import pandas as pd
import numpy as np

def preprocess_df(df):
    # giữ các cột cần thiết
    keep_cols = [
        "tconst",
        "primaryTitle",
        "startYear",
        "runtimeMinutes",
        "genres",
        "averageRating",
        "numVotes"
    ]

    df = df[keep_cols].copy()

    # convert kiểu dữ liệu
    df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
    df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
    df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")
    df["numVotes"] = pd.to_numeric(df["numVotes"], errors="coerce")

    # drop null
    df = df.dropna()

    # ✅ TẠO LABEL
    df["is_success"] = (
        (df["averageRating"] >= 7.0) &
        (df["numVotes"] >= 10000)
    ).astype(int)

    return df