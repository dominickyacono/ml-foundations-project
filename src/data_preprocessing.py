"""Data loading and multivariate sequence preprocessing.

Supports paper-aligned feature engineering and multiple temporal split strategies.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


PAPER_FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_30",
    "roll_mean_7",
    "roll_mean_30",
    "day_of_week",
    "month",
]

BASELINE_FEATURE_COLS = [
    "sales",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "day_of_month",
    "is_weekend",
]


SPLIT_STRATEGIES = {
    "per_group": "Chronological split within each store-item group.",
    "global_temporal_80_20": "Global date-ordered split, closer to paper's temporal 80/20 framing.",
}


def load_and_preprocess(
    csv_path: str = "data/raw/train.csv",
    window_size: int = 30,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    feature_set: str = "paper",
    split_strategy: str = "global_temporal_80_20",
):
    """Load train.csv and return chronological train/val/test arrays.

    Parameters
    ----------
    feature_set : str
        "paper" for lag/rolling/time features or "baseline" for cyclical features.
    split_strategy : str
        "per_group" or "global_temporal_80_20".
    val_frac : float
        Fraction of pre-test data reserved for validation.
    test_frac : float
        Fraction of samples reserved for test set from temporal tail.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray, shape (N, window_size, 7)
    y_train, y_val, y_test : np.ndarray, shape (N,)
    y_scaler : sklearn MinMaxScaler fitted on training targets only
    """
    if feature_set not in {"paper", "baseline"}:
        raise ValueError("feature_set must be 'paper' or 'baseline'.")
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split_strategy must be one of {list(SPLIT_STRATEGIES.keys())}.")

    df = _load_csv(csv_path)

    X_train_raw, X_val_raw, X_test_raw, y_train_raw, y_val_raw, y_test_raw = _build_and_split(
        df=df,
        window_size=window_size,
        val_frac=val_frac,
        test_frac=test_frac,
        feature_set=feature_set,
        split_strategy=split_strategy,
    )

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit(y_train_raw.reshape(-1, 1))

    X_train = x_scaler.transform(X_train_raw.reshape(-1, X_train_raw.shape[-1])).reshape(X_train_raw.shape).astype(np.float32)
    X_val = x_scaler.transform(X_val_raw.reshape(-1, X_val_raw.shape[-1])).reshape(X_val_raw.shape).astype(np.float32)
    X_test = x_scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape).astype(np.float32)

    y_train = y_scaler.transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
    y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).flatten().astype(np.float32)
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test, y_scaler


def _load_csv(csv_path: str) -> pd.DataFrame:
    """Read CSV and add shared time columns."""
    df = pd.read_csv(csv_path, parse_dates=["date"])

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df


def _engineer_paper_features(group: pd.DataFrame) -> pd.DataFrame:
    """Create lag/rolling features using only past information."""
    g = group.copy()
    g["lag_1"] = g["sales"].shift(1)
    g["lag_7"] = g["sales"].shift(7)
    g["lag_30"] = g["sales"].shift(30)
    g["roll_mean_7"] = g["sales"].shift(1).rolling(window=7, min_periods=7).mean()
    g["roll_mean_30"] = g["sales"].shift(1).rolling(window=30, min_periods=30).mean()
    return g.dropna(subset=PAPER_FEATURE_COLS)


def _build_group_windows(group: pd.DataFrame, window_size: int, feature_cols: list[str]):
    """Build supervised windows for one group and return target dates."""
    if len(group) <= window_size:
        return None, None, None

    group_features = group[feature_cols].values.astype(np.float32)
    group_target = group["sales"].values.astype(np.float32)
    group_dates = group["date"].values

    X_group, y_group, d_group = [], [], []
    for i in range(window_size, len(group)):
        X_group.append(group_features[i - window_size : i])
        y_group.append(group_target[i])
        d_group.append(group_dates[i])

    if not X_group:
        return None, None, None

    return (
        np.array(X_group, dtype=np.float32),
        np.array(y_group, dtype=np.float32),
        np.array(d_group),
    )


def _split_by_indices(X: np.ndarray, y: np.ndarray, val_frac: float, test_frac: float):
    n = len(X)
    test_start = int(n * (1 - test_frac))
    val_start = int(test_start * (1 - val_frac))

    if val_start == 0 or test_start <= val_start or test_start >= n:
        raise ValueError("Invalid split sizes. Adjust val_frac/test_frac.")

    X_train = X[:val_start]
    y_train = y[:val_start]
    X_val = X[val_start:test_start]
    y_val = y[val_start:test_start]
    X_test = X[test_start:]
    y_test = y[test_start:]
    return X_train, X_val, X_test, y_train, y_val, y_test


def _build_and_split(
    df: pd.DataFrame,
    window_size: int,
    val_frac: float,
    test_frac: float,
    feature_set: str,
    split_strategy: str,
):
    """Build windows and split by selected strategy."""
    feature_cols = PAPER_FEATURE_COLS if feature_set == "paper" else BASELINE_FEATURE_COLS

    all_X, all_y, all_dates = [], [], []

    for (_store, _item), group in df.groupby(["store", "item"]):
        g = _engineer_paper_features(group) if feature_set == "paper" else group
        X_g, y_g, d_g = _build_group_windows(g, window_size, feature_cols)
        if X_g is None:
            continue

        if split_strategy == "per_group":
            X_tr, X_va, X_te, y_tr, y_va, y_te = _split_by_indices(X_g, y_g, val_frac, test_frac)
            all_X.append((X_tr, X_va, X_te))
            all_y.append((y_tr, y_va, y_te))
        else:
            all_X.append(X_g)
            all_y.append(y_g)
            all_dates.append(d_g)

    if not all_X:
        raise ValueError("No windows were generated from the dataset.")

    if split_strategy == "per_group":
        X_train = np.concatenate([x[0] for x in all_X])
        X_val = np.concatenate([x[1] for x in all_X])
        X_test = np.concatenate([x[2] for x in all_X])
        y_train = np.concatenate([y[0] for y in all_y])
        y_val = np.concatenate([y[1] for y in all_y])
        y_test = np.concatenate([y[2] for y in all_y])
        return X_train, X_val, X_test, y_train, y_val, y_test

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    d_all = np.concatenate(all_dates)

    order = np.argsort(d_all)
    X_all = X_all[order]
    y_all = y_all[order]

    return _split_by_indices(X_all, y_all, val_frac, test_frac)
