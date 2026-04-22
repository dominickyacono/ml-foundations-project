"""LSTM model definitions for retail sales forecasting.

This module provides the architecture family used in the replication study:
single-layer LSTMs with hidden-size compression from 128 to 16 units.
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Single-layer LSTM followed by dropout and a Dense-16 bottleneck.

    Parameters
    ----------
    input_size  : int   Number of features per time step (7 for this project).
    hidden_size : int   Number of hidden units in the single LSTM layer.
    dropout     : float Dropout probability applied after LSTM summary state.
    dense_size  : int   Width of the bottleneck dense layer.
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        dropout: float = 0.2,
        dense_size: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.fc2 = nn.Linear(dense_size, 1)

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
        last_hidden = out[:, -1, :]  # sequence summary at final time step
        z = self.dropout(last_hidden)
        z = torch.relu(self.fc1(z))
        return self.fc2(z).squeeze(-1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


_CONFIGS = {
    "lstm128": dict(hidden_size=128, dropout=0.2, dense_size=16),
    "lstm64": dict(hidden_size=64, dropout=0.2, dense_size=16),
    "lstm48": dict(hidden_size=48, dropout=0.2, dense_size=16),
    "lstm32": dict(hidden_size=32, dropout=0.2, dense_size=16),
    "lstm16": dict(hidden_size=16, dropout=0.2, dense_size=16),
    # Backward-compatible alias used by earlier notebooks/scripts.
    "heavy": dict(hidden_size=128, dropout=0.2, dense_size=16),
}


def build_model(variant: str = "lstm128", input_size: int = 7) -> LSTMForecaster:
    """Build a named LSTM variant.

    Parameters
    ----------
    variant : str
        One of "lstm128", "lstm64", "lstm48", "lstm32", "lstm16".
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


VARIANT_NAMES = tuple(k for k in _CONFIGS.keys() if k != "heavy")
