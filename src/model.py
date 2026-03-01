"""
model.py
--------
LSTM model definitions for retail sales forecasting.

Two classes are provided:

* ``LSTMForecaster`` – configurable LSTM (used for all experiments).
* ``build_model``    – convenience factory that creates named model variants.

Example
-------
    from src.model import build_model

    heavy  = build_model("heavy")    # 128 units, 2 layers
    medium = build_model("medium")   #  64 units, 2 layers
    light  = build_model("light")    #  32 units, 1 layer
    tiny   = build_model("tiny")     #  16 units, 1 layer
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Single- or multi-layer LSTM followed by a fully-connected output layer.

    Parameters
    ----------
    input_size  : int   Number of features per time step (1 for univariate).
    hidden_size : int   Number of hidden units per LSTM layer.
    num_layers  : int   Number of stacked LSTM layers.
    dropout     : float Dropout probability applied between LSTM layers
                        (only active when ``num_layers > 1``).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor, shape (batch,)
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]   # take output at final time step
        return self.fc(last_hidden).squeeze(-1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Named configurations
# ---------------------------------------------------------------------------

_CONFIGS = {
    "heavy":  dict(hidden_size=128, num_layers=2, dropout=0.2),
    "medium": dict(hidden_size=64,  num_layers=2, dropout=0.2),
    "light":  dict(hidden_size=32,  num_layers=1, dropout=0.0),
    "tiny":   dict(hidden_size=16,  num_layers=1, dropout=0.0),
}


def build_model(variant: str = "heavy", input_size: int = 1) -> LSTMForecaster:
    """Build a named LSTM variant.

    Parameters
    ----------
    variant : str
        One of ``"heavy"``, ``"medium"``, ``"light"``, ``"tiny"``.
    input_size : int
        Number of input features per time step.

    Returns
    -------
    LSTMForecaster
    """
    if variant not in _CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(_CONFIGS.keys())}"
        )
    cfg = _CONFIGS[variant]
    return LSTMForecaster(input_size=input_size, **cfg)
