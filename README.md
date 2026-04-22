# Optimizing LSTM Neural Networks for Resource-Constrained Retail Sales Forecasting

## Overview

Retail businesses rely on accurate sales forecasts to maintain the right amount of inventory,
preventing both product shortages and costly waste. However, the most accurate AI models used
for these predictions are often computationally "heavy," requiring expensive hardware and high
power consumption that many small businesses or in-store devices cannot support.

This project replicates a recent study that addresses this barrier by optimizing an LSTM neural
network for retail sales forecasting. The goal is to demonstrate that it is possible to reduce
the model's size and memory requirements for use on standard, resource-constrained devices
**without significantly sacrificing its prediction accuracy**.

---

## Dataset

**Kaggle Store Item Demand Forecasting Challenge**
- 5 years of daily sales data (2013–2017)
- 10 stores × 50 items = 500 time series
- Download from: <https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data>
- Place the CSV files inside `data/raw/`

---

## Project Structure

```
ml-foundations-project/
├── data/
│   ├── raw/               # Raw CSVs from Kaggle (not tracked by git)
│   └── processed/         # Preprocessed arrays  (not tracked by git)
├── models/                # Saved model weights   (not tracked by git)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_baseline_models.ipynb    # Linear regression baseline
│   └── 03_lstm_experiments.ipynb   # LSTM training & comparison
├── results/               # Plots and metric CSVs (not tracked by git)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data loading, normalisation, windowing
│   ├── model.py               # LSTM model definitions
│   ├── train.py               # Training loop
│   └── evaluate.py            # Evaluation metrics & efficiency report
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download data

Download `train.csv` and `test.csv` from the Kaggle competition and place them at `data/raw/train.csv` and `data/raw/test.csv`.

### 3. Preprocess data

```python
from src.data_preprocessing import load_and_preprocess

X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess(
    csv_path="data/raw/train.csv",
    window_size=30,
    val_frac=0.1,
    test_frac=0.2,
    feature_set="paper",
    split_strategy="global_temporal_80_20",
)
```

### 4. Train models

Current replication variants are:

- `lstm128` (full-capacity reference)
- `lstm64`
- `lstm48`
- `lstm32`
- `lstm16`

All variants use the same single-layer architecture and differ only in hidden unit count.

```bash
python src/train.py --variant lstm128 --epochs 30 --batch_size 64 --loss_name mae --feature_set paper --split_strategy global_temporal_80_20
python src/train.py --variant lstm64
python src/train.py --variant lstm48
python src/train.py --variant lstm32
python src/train.py --variant lstm16
```

Trained weights are saved to **`models/lstm_<variant>.pt`**. For these variant names, that becomes files such as `models/lstm_lstm64.pt` and `models/lstm_lstm128.pt`.
These files are excluded from git — see [`models/README.md`](models/README.md) for the full naming table and a code snippet to reload a saved model.

### 5. Evaluate

```bash
python src/evaluate.py --model_path models/lstm_lstm64.pt --variant lstm64 --feature_set paper --split_strategy global_temporal_80_20
```

### 6. Notebooks

Open the notebooks in order for a guided walkthrough:

```bash
jupyter notebook notebooks/
```

---

## Methodology

| Stage | Description |
|-------|-------------|
| **LSTM compression sweep** | Single-layer LSTM family (`lstm128`, `lstm64`, `lstm48`, `lstm32`, `lstm16`) with shared architecture and compressed hidden sizes |
| **Linear regression** | Simple statistical baseline for efficiency comparison |

---

## Evaluation Metrics

| Category | Metric |
|----------|--------|
| Accuracy | MSE (training), MAPE (final) |
| Efficiency | Trainable parameter count, model file size |
| Latency | Average inference time per sample |

---

## References

- Bandara, K., Bergmeir, C., & Smyl, S. (2019). *Forecasting across time series databases using recurrent neural networks on groups of similar series.* Expert Systems with Applications, 140, 112896.
- *Optimizing LSTM Neural Networks for Resource-Constrained Retail Sales Forecasting* (paper being replicated).
- Kaggle Store Item Demand Forecasting Challenge: <https://www.kaggle.com/competitions/demand-forecasting-kernels-only>
