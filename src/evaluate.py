"""
evaluate.py
-----------
Evaluation utilities: accuracy metrics, efficiency metrics, and inference
latency measurement.

Usage (command line)
--------------------
    python src/evaluate.py --model_path models/lstm_heavy.pt --variant heavy

Usage (notebook / script)
--------------------------
    from src.evaluate import evaluate_model

    report = evaluate_model(
        model_path="models/lstm_heavy.pt",
        variant="heavy",
        csv_path="data/raw/train.csv",
    )
    print(report)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from src.data_preprocessing import load_and_preprocess
from src.model import build_model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE (%) – zero-sale days are excluded to avoid division by zero."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def model_file_size_kb(model_path: str) -> float:
    """Return the size of a saved model file in kilobytes."""
    return os.path.getsize(model_path) / 1024


def measure_inference_latency(model, X_sample: torch.Tensor, n_runs: int = 100) -> float:
    """Return average inference time in milliseconds over ``n_runs`` runs.

    Parameters
    ----------
    model    : LSTMForecaster (in eval mode, on CPU)
    X_sample : torch.Tensor, shape (1, seq_len, 1)
    n_runs   : int
    """
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(X_sample)

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(X_sample)
        elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000  # ms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    variant: str,
    csv_path: str = "data/raw/train.csv",
    window_size: int = 30,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    results_dir: str = "results",
) -> dict:
    """Load a saved model, run inference on the test set, and report metrics.

    Returns
    -------
    dict with keys:
        variant, n_parameters, file_size_kb, mse, mape_pct, latency_ms
    """
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    _, _, X_test, _, _, y_test, scaler = load_and_preprocess(
        csv_path=csv_path,
        window_size=window_size,
        val_frac=val_frac,
        test_frac=test_frac,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(variant=variant, input_size=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Predictions (batched for memory efficiency)
    # ------------------------------------------------------------------
    batch_size = 512
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            preds.append(model(batch).numpy())
    y_pred_scaled = np.concatenate(preds)

    # Inverse-transform to original sales scale
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    mse  = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    latency = measure_inference_latency(model, X_tensor[:1])
    n_params = model.count_parameters()
    file_kb  = model_file_size_kb(model_path)

    report = {
        "variant":       variant,
        "n_parameters":  n_params,
        "file_size_kb":  round(file_kb, 2),
        "mse":           round(mse, 4),
        "mape_pct":      round(mape, 4),
        "latency_ms":    round(latency, 4),
    }

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    report_path = os.path.join(results_dir, f"eval_{variant}.csv")
    pd.DataFrame([report]).to_csv(report_path, index=False)
    print(f"Evaluation report saved → {report_path}")

    return report


def compare_all_variants(
    variants=("heavy",),
    models_dir: str = "models",
    results_dir: str = "results",
    **kwargs,
) -> pd.DataFrame:
    """Evaluate every available saved model variant and return a summary table."""
    rows = []
    for v in variants:
        model_path = os.path.join(models_dir, f"lstm_{v}.pt")
        if not os.path.isfile(model_path):
            print(f"Skipping '{v}' – model file not found: {model_path}")
            continue
        report = evaluate_model(model_path=model_path, variant=v,
                                results_dir=results_dir, **kwargs)
        rows.append(report)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(results_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary table saved → {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved LSTM model.")
    p.add_argument("--model_path",  required=True)
    p.add_argument("--variant",     required=True,
                   choices=["heavy"])
    p.add_argument("--csv_path", default="data/raw/train.csv")
    p.add_argument("--window_size",    type=int,   default=30)
    p.add_argument("--val_frac",       type=float, default=0.1)
    p.add_argument("--test_frac",      type=float, default=0.1)
    p.add_argument("--results_dir",    default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = evaluate_model(**vars(args))
    print("\n=== Evaluation Report ===")
    for k, v in report.items():
        print(f"  {k:20s}: {v}")
