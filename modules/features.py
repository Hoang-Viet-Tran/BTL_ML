import numpy as np
import os
import h5py
from sklearn.preprocessing import StandardScaler

def extract_features(df, label_column="is_success"):
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. "
            f"Did you forget to run preprocess_df()?"
        )

    feature_cols = [
        "startYear",
        "runtimeMinutes",
        "averageRating",
        "numVotes"
    ]

    X = df[feature_cols].values
    y = df[label_column].values

    return X, y

def save_features(X, y, out_dir="features"):
    """
    Lưu feature ra file .npy và .h5
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save NPY
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

    # Save H5
    with h5py.File(os.path.join(out_dir, "features.h5"), "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)