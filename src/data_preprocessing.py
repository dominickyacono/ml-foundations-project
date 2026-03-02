"""
data_preprocessing.py
---------------------
Loads the Kaggle Store Item Demand Forecasting ``train.csv`` (which contains
the ground-truth ``sales`` column), normalises per-series sales with
Min-Max scaling, creates sliding-window sequences, and splits them
chronologically into train / validation / test sets.

The Kaggle ``test.csv`` does **not** contain a ``sales`` column (it is meant
for competition submissions), so all modelling uses ``train.csv`` only.

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
    csv_path: str = "data/raw/train.csv",
    window_size: int = 30,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
):
    """Load train.csv and return chronological train/val/test window arrays.

    Pipeline
    --------
    1. Build sliding-window sequences from **raw** (un-normalised) sales.
    2. Split each store-item group chronologically so that every group
       contributes to train, val, and test (true temporal split).
    3. Fit a MinMaxScaler on the **training split only** (no data leakage).
    4. Transform all splits with that scaler.

    Parameters
    ----------
    csv_path : str
        Path to ``train.csv`` from the Kaggle competition (must contain a
        ``sales`` column).
    window_size : int
        Number of past days used as model input (look-back window).
    val_frac : float
        Fraction of each group's sequences reserved for validation.
    test_frac : float
        Fraction of each group's sequences reserved for the test set
        (taken from the chronological end).

    Returns
    -------
    X_train, X_val, X_test : np.ndarray, shape (N, window_size, 1)
    y_train, y_val, y_test : np.ndarray, shape (N,)
    scaler : sklearn MinMaxScaler fitted on training sales values only
    """
    df = _load_csv(csv_path)

    # Step 1 + 2: build raw windows and split per group chronologically
    X_train, X_val, X_test, y_train, y_val, y_test = _build_and_split(
        df, window_size, val_frac, test_frac,
    )

    # Step 3: fit scaler on training data only (both inputs and targets
    # are sales values on the same scale)
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_train_sales = np.concatenate([
        X_train.reshape(-1, 1),
        y_train.reshape(-1, 1),
    ])
    scaler.fit(all_train_sales)

    # Step 4: normalise all splits
    X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape).astype(np.float32)
    X_val   = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape).astype(np.float32)
    X_test  = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape).astype(np.float32)
    y_train = scaler.transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
    y_val   = scaler.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
    y_test  = scaler.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str) -> pd.DataFrame:
    """Read the CSV and return a tidy DataFrame sorted by date."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df


def _build_and_split(
    df: pd.DataFrame,
    window_size: int,
    val_frac: float,
    test_frac: float,
):
    """Build sliding windows from raw sales per group and split each group
    chronologically before concatenating.

    This guarantees that the test set contains the **latest** time-steps
    from every store-item pair, rather than the last few groups entirely.
    """
    tr_X, tr_y = [], []
    va_X, va_y = [], []
    te_X, te_y = [], []

    for (_store, _item), group in df.groupby(["store", "item"]):
        sales = group["sales"].values.astype(np.float32)

        # Build windows for this group
        X_group, y_group = [], []
        for i in range(window_size, len(sales)):
            X_group.append(sales[i - window_size : i])
            y_group.append(sales[i])

        X_g = np.array(X_group, dtype=np.float32)
        y_g = np.array(y_group, dtype=np.float32)
        n = len(X_g)

        # Chronological split within this group
        test_start = int(n * (1 - test_frac))
        val_start  = int(test_start * (1 - val_frac))

        tr_X.append(X_g[:val_start])
        tr_y.append(y_g[:val_start])
        va_X.append(X_g[val_start:test_start])
        va_y.append(y_g[val_start:test_start])
        te_X.append(X_g[test_start:])
        te_y.append(y_g[test_start:])

    X_train = np.concatenate(tr_X)[:, :, np.newaxis]
    X_val   = np.concatenate(va_X)[:, :, np.newaxis]
    X_test  = np.concatenate(te_X)[:, :, np.newaxis]
    y_train = np.concatenate(tr_y)
    y_val   = np.concatenate(va_y)
    y_test  = np.concatenate(te_y)

    return X_train, X_val, X_test, y_train, y_val, y_test
