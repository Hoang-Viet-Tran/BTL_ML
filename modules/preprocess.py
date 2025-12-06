import pandas as pd
import numpy as np

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu:
    - clone dataframe
    - xử lý missing values
    - chuyển các kiểu dữ liệu cơ bản
    """
    df = df.copy()

    # Drop duplicate
    df.drop_duplicates(inplace=True)

    # Fill NA
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(0)

    return df