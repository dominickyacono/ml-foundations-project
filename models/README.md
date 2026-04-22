# Saved model weights

Trained model weights are written here by `src/train.py` and are **not tracked by git** (see `.gitignore`).

Each file is named after its variant:

| File | Variant | Hidden units | Layers | ~Parameters |
|------|---------|-------------|--------|-------------|
| `lstm_lstm128.pt` | lstm128 | 128 | 1 | ~72,225 |
| `lstm_lstm64.pt`  | lstm64  | 64  | 1 | ~19,745 |
| `lstm_lstm48.pt`  | lstm48  | 48  | 1 | ~11,745 |
| `lstm_lstm32.pt`  | lstm32  | 32  | 1 | ~5,793 |
| `lstm_lstm16.pt`  | lstm16  | 16  | 1 | ~1,889 |

Note: `heavy` is kept in code as a backward-compatible alias, but the active replication workflow uses the `lstm*` variants above.

## How to generate these files

Run `src/train.py` once per variant, for example:

```bash
python src/train.py --variant lstm128 --epochs 30 --batch_size 64 --loss_name mae --feature_set paper --split_strategy global_temporal_80_20
python src/train.py --variant lstm64
python src/train.py --variant lstm48
python src/train.py --variant lstm32
python src/train.py --variant lstm16
```

Or use the `train()` function from `notebooks/03_lstm_experiments.ipynb`.

## How to load a saved model

```python
from src.model import build_model
import torch

model = build_model("lstm64", input_size=7)
model.load_state_dict(torch.load("models/lstm_lstm64.pt", map_location="cpu"))
model.eval()
```
