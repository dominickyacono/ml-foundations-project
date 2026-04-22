# Result artifacts

Training/evaluation artifacts are written here by `src/train.py`, `src/evaluate.py`, and `notebooks/03_lstm_experiments.ipynb`.
These files are ignored by git (see `.gitignore`).

Typical outputs include:

- `summary.csv`: consolidated comparison across variants (`mape_pct`, `rmse`, `mse`, parameters, size, latency, feature/split metadata)
- `eval_<variant>.csv`: per-variant evaluation report (for example `eval_lstm64.csv`)
- `train_<variant>.csv`: per-epoch training history (for example `train_lstm64.csv`)
- `lstm_loss_curves.png`: validation-loss plot across variants
- `accuracy_efficiency_tradeoff.png`: accuracy/efficiency scatter plots
