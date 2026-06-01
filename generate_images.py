"""
Generated script to create Tufte-style visualizations
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

images_dir = Path(__file__).resolve().parent / "images"
images_dir.mkdir(exist_ok=True)
original_savefig = plt.savefig


class _LSTMForecaster(nn.Module):
    def __init__(self, n_features: int = 1, hidden: int = 32, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TorchRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for the LSTM forecaster."""

    def __init__(self, epochs: int = 10):
        self.epochs = epochs
        self.model = _LSTMForecaster()

    def fit(self, X, y):
        X3 = X.reshape(X.shape[0], X.shape[1], 1).astype(np.float32)
        y2 = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        X_t = torch.from_numpy(X3)
        y_t = torch.from_numpy(y2)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=16, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.model(xb), yb).backward()
                opt.step()
        return self

    def predict(self, X):
        self.model.eval()
        X3 = X.reshape(X.shape[0], X.shape[1], 1).astype(np.float32)
        with torch.no_grad():
            return self.model(torch.from_numpy(X3)).numpy().ravel()


def savefig_tufte(filename, **kwargs):
    if not str(filename).startswith("/") and not str(filename).startswith("images/"):
        filename = images_dir / filename
    original_savefig(filename, **kwargs)
    logger.info("Saved: %s", filename)


def main() -> None:
    plt.savefig = savefig_tufte
    np.random.seed(42)
    torch.manual_seed(42)

    n = 120
    lag = 8
    series = np.sin(np.arange(n) / 5) + np.random.normal(0, 0.05, n)
    X, y = [], []
    for i in range(lag, len(series)):
        X.append(series[i - lag : i])
        y.append(series[i])
    X_arr = np.array(X)
    y_arr = np.array(y)

    voting = VotingRegressor(
        [("ridge", Ridge()), ("linear", LinearRegression())]
    )
    voting.fit(X_arr, y_arr)
    preds = voting.predict(X_arr)

    lstm = TorchRegressor(epochs=8)
    lstm.fit(X_arr, y_arr)
    lstm_preds = lstm.predict(X_arr)

    stacking = StackingRegressor(
        estimators=[("ridge", Ridge()), ("linear", LinearRegression())],
        final_estimator=LinearRegression(),
    )
    stacking.fit(X_arr, y_arr)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_arr, label="actual", linewidth=1)
    ax.plot(preds, label="voting ensemble", linewidth=1)
    ax.plot(lstm_preds, label="lstm", linewidth=1, alpha=0.8)
    ax.legend()
    plt.savefig("ensemble_forecast.png")
    logger.info("All images generated successfully!")


if __name__ == "__main__":
    main()
