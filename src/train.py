"""
train.py
--------
Training script for the LSTM retail sales forecasting models.

Usage (command line)
--------------------
    python src/train.py --variant heavy   --epochs 20 --batch_size 256

Data
----
    Expects ``train.csv`` in ``data/raw/`` (must contain a ``sales`` column).

Saved artefacts
---------------
    models/lstm_<variant>.pt   – model state dict
    results/train_<variant>.csv – per-epoch train/val MSE loss
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_preprocessing import load_and_preprocess
from src.model import build_model


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    csv_path="data/raw/train.csv",
    variant="heavy",
    window_size=30,
    epochs=20,
    batch_size=256,
    lr=1e-3,
    val_frac=0.1,
    test_frac=0.1,
    models_dir="models",
    results_dir="results",
)


# ---------------------------------------------------------------------------
# Core training function (importable by notebooks)
# ---------------------------------------------------------------------------

def train(
    csv_path: str = DEFAULTS["csv_path"],
    variant: str = DEFAULTS["variant"],
    window_size: int = DEFAULTS["window_size"],
    epochs: int = DEFAULTS["epochs"],
    batch_size: int = DEFAULTS["batch_size"],
    lr: float = DEFAULTS["lr"],
    val_frac: float = DEFAULTS["val_frac"],
    test_frac: float = DEFAULTS["test_frac"],
    models_dir: str = DEFAULTS["models_dir"],
    results_dir: str = DEFAULTS["results_dir"],
):
    """Train an LSTM variant and save the model weights + loss history.

    Returns
    -------
    model   : trained LSTMForecaster
    history : pd.DataFrame with columns ['epoch', 'train_mse', 'val_mse']
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading and preprocessing data …")
    t_data = time.time()
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_and_preprocess(
        csv_path=csv_path,
        window_size=window_size,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    print(f"  Data loaded in {time.time() - t_data:.1f}s")
    print(f"  Train samples : {len(X_train):,}")
    print(f"  Val   samples : {len(X_val):,}")
    print(f"  Test  samples : {len(X_test):,}")
    print(f"  Window size   : {window_size}")

    def to_tensors(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    train_loader = DataLoader(to_tensors(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(to_tensors(X_val,   y_val),   batch_size=batch_size)
    print(f"  Train batches : {len(train_loader):,}  (batch_size={batch_size})")
    print(f"  Val   batches : {len(val_loader):,}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(variant=variant, input_size=1).to(device)
    print(f"Variant '{variant}' | Parameters: {model.count_parameters():,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Starting training: {epochs} epochs, lr={lr}")
    print(f"{'='*60}")
    total_start = time.time()
    history = []
    best_val_mse = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        n_batches = len(train_loader)
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader, 1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Print progress every 25% of batches
            if batch_idx % max(1, n_batches // 4) == 0 or batch_idx == n_batches:
                print(f"  Epoch {epoch:3d}/{epochs}  "
                      f"batch {batch_idx:>5d}/{n_batches}  "
                      f"batch_loss={loss.item():.6f}", end="\r")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())

        train_mse = float(np.mean(train_losses))
        val_mse   = float(np.mean(val_losses))
        epoch_time = time.time() - epoch_start
        elapsed    = time.time() - total_start
        eta        = elapsed / epoch * (epochs - epoch)

        improved = val_mse < best_val_mse
        if improved:
            best_val_mse = val_mse

        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  "
              f"({epoch_time:.1f}s) "
              f"{'↓ best' if improved else ''}")

    total_time = time.time() - total_start
    print(f"{'='*60}")
    print(f"  Training complete in {total_time:.1f}s  "
          f"({total_time/60:.1f} min)")
    print(f"  Best val MSE: {best_val_mse:.6f}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    model_path = os.path.join(models_dir, f"lstm_{variant}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved → {model_path}")

    history_df = pd.DataFrame(history)
    history_path = os.path.join(results_dir, f"train_{variant}.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Loss history saved → {history_path}")

    return model, history_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Train an LSTM forecasting model.")
    p.add_argument("--csv_path", default=DEFAULTS["csv_path"])
    p.add_argument("--variant",        default=DEFAULTS["variant"],
                   choices=["heavy"])
    p.add_argument("--window_size",    type=int,   default=DEFAULTS["window_size"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch_size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--val_frac",       type=float, default=DEFAULTS["val_frac"])
    p.add_argument("--test_frac",      type=float, default=DEFAULTS["test_frac"])
    p.add_argument("--models_dir",     default=DEFAULTS["models_dir"])
    p.add_argument("--results_dir",    default=DEFAULTS["results_dir"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(**vars(args))
