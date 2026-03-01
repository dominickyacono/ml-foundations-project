"""
data_preprocessing.py
---------------------
Loads the Kaggle Store Item Demand Forecasting CSV, normalises sales per
(store, item) time series, and creates sliding-window sequences ready for
PyTorch training.

Usage
-----
    from src.data_preprocessing import load_and_preprocess

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess(
        csv_path="data/raw/train.csv",
        window_size=30,
        val_frac=0.1,
        test_frac=0.1,
    )
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess(
    csv_path: str,
    window_size: int = 30,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
):
    """Load the Kaggle dataset and return train/val/test window arrays.

    Parameters
    ----------
    csv_path : str
        Path to ``train.csv`` from the Kaggle competition.
    window_size : int
        Number of past days used as model input (look-back window).
    val_frac : float
        Fraction of sequences reserved for validation.
    test_frac : float
        Fraction of sequences reserved for testing.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray, shape (N, window_size, 1)
    y_train, y_val, y_test : np.ndarray, shape (N,)
    scaler : sklearn MinMaxScaler fitted on training sales values
    """
    df = _load_csv(csv_path)
    sequences, scaler = _build_sequences(df, window_size)
    return _split(sequences, val_frac, test_frac, scaler)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str) -> pd.DataFrame:
    """Read the CSV and return a tidy DataFrame sorted by date."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df


def _build_sequences(
    df: pd.DataFrame,
    window_size: int,
):
    """Normalise sales and create sliding-window (X, y) pairs.

    A single MinMaxScaler is fitted on *all* sales values so that the same
    scale is used consistently across every (store, item) series and at
    inference time.

    Returns
    -------
    sequences : (X, y) tuple of np.ndarray
    scaler    : fitted MinMaxScaler
    """
    all_sales = df["sales"].values.astype(np.float32).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_sales)

    all_X, all_y = [], []

    for (store, item), group in df.groupby(["store", "item"]):
        sales = group["sales"].values.astype(np.float32).reshape(-1, 1)
        scaled = scaler.transform(sales).flatten()

        for i in range(window_size, len(scaled)):
            all_X.append(scaled[i - window_size : i])
            all_y.append(scaled[i])

    X = np.array(all_X, dtype=np.float32)[:, :, np.newaxis]  # (N, W, 1)
    y = np.array(all_y, dtype=np.float32)
    return (X, y), scaler


def _split(sequences, val_frac: float, test_frac: float, scaler):
    """Chronology-aware split: train | val | test."""
    X, y = sequences
    n = len(X)

    test_start = int(n * (1 - test_frac))
    val_start = int(test_start * (1 - val_frac))

    X_train, y_train = X[:val_start],            y[:val_start]
    X_val,   y_val   = X[val_start:test_start],  y[val_start:test_start]
    X_test,  y_test  = X[test_start:],           y[test_start:]

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
