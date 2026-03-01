# Saved model weights

Trained model weights are written here by `src/train.py` and are **not tracked by git** (see `.gitignore`).

Each file is named after its variant:

| File | Variant | Hidden units | Layers | ~Parameters |
|------|---------|-------------|--------|-------------|
| `lstm_heavy.pt`  | heavy  | 128 | 2 | ~199 k |
| `lstm_medium.pt` | medium | 64  | 2 | ~50 k  |
| `lstm_light.pt`  | light  | 32  | 1 | ~4.5 k |
| `lstm_tiny.pt`   | tiny   | 16  | 1 | ~1.2 k |

## How to generate these files

Run `src/train.py` once per variant, for example:

```bash
python src/train.py --variant heavy  --epochs 20
python src/train.py --variant medium --epochs 20
python src/train.py --variant light  --epochs 20
python src/train.py --variant tiny   --epochs 20
```

Or use the `train()` function from `notebooks/03_lstm_experiments.ipynb`, which trains all four variants in sequence.

## How to load a saved model

```python
from src.model import build_model
import torch

model = build_model("heavy")
model.load_state_dict(torch.load("models/lstm_heavy.pt", map_location="cpu"))
model.eval()
```
