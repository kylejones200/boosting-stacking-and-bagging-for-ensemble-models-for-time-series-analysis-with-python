"""
Generated script to create Tufte-style visualizations
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import StackingRegressor, VotingRegressor
from torch.utils.data import DataLoader, TensorDataset


class _LSTMForecaster(nn.Module):
    """LSTM forecaster (auto-generated PyTorch replacement for Keras Sequential)."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        output_size: int = 1,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()


def _train_torch(
    model: nn.Module,
    X_train,
    y_train,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    validation_split: float = 0.2,
    patience: int = 15,
) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = (X_t[-n_val:], y_t[-n_val:])
    X_tr, y_tr = (X_t[:-n_val], y_t[:-n_val])
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = (float("inf"), 0)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = (val_loss, 0)
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def savefig_tufte(filename, **kwargs):
    """Wrapper to save figures in images directory with Tufte style"""
    if not str(filename).startswith("/") and (not str(filename).startswith("images/")):
        filename = images_dir / filename
    original_savefig(filename, **kwargs)
    logger.info(f"Saved: {filename}")


def main() -> None:
    plt.savefig = savefig_tufte

    data_path = Path("../../geospatial/datasets/energy_indicators.csv")

    voting_model = VotingRegressor(
        [("arima", arima_model), ("lstm", lstm_model), ("prophet", prophet_model)]
    )

    np.random.seed(42)

    stacking_model = StackingRegressor(
        estimators=[("arima", arima), ("lstm", lstm)],
        final_estimator=LinearRegression(),
    )

    logger.info("All images generated successfully!")


if __name__ == "__main__":
    main()
