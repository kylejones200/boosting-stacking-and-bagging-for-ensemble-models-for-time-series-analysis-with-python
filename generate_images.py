#!/usr/bin/env python3
"""
Generated script to create Tufte-style visualizations
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

import signalplot

logger = logging.getLogger(__name__)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set random seeds

# Tufte-style configuration
signalplot.apply(font_family="serif")

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# Update all savefig calls to use images_dir
original_savefig = plt.savefig

class _LSTMForecaster(nn.Module):
    """LSTM forecaster (auto-generated PyTorch replacement for Keras Sequential)."""
    def __init__(self, n_features: int, hidden: int = 64, output_size: int = 1,
                 n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))

def _train_torch(model: nn.Module, X_train, y_train, *,
                 epochs: int = 50, batch_size: int = 32,
                 lr: float = 0.001, validation_split: float = 0.2,
                 patience: int = 15) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
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
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model

def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()

def savefig_tufte(filename, **kwargs):
    """Wrapper to save figures in images directory with Tufte style"""
    if not str(filename).startswith("/") and not str(filename).startswith("images/"):
        filename = images_dir / filename
    original_savefig(filename, **kwargs)
    logger.info(f"Saved: {filename}")

def main():
    plt.savefig = savefig_tufte

    # Code blocks from article

    # Code block 1

    data_path = Path("../../geospatial/datasets/energy_indicators.csv")

    # Code block 2
    # ARIMA
    # LSTM
    # Prophet
    # XGBoost
    # Each provides different strengths

    # Code block 3
    from sklearn.ensemble import VotingRegressor

    # Simple average of predictions
    voting_model = VotingRegressor(
        [("arima", arima_model), ("lstm", lstm_model), ("prophet", prophet_model)]
    )

    # Code block 4
    # Weighted average with optimized weights
    # Use validation set to find best weights

    # Code block 5
    from sklearn.ensemble import StackingRegressor

    np.random.seed(42)

    # Meta-learner on base model predictions
    stacking_model = StackingRegressor(
        estimators=[("arima", arima), ("lstm", lstm)], final_estimator=LinearRegression()
    )

    # Code block 6
    # Learn how to combine models
    # Feature importance from base models
    # Dynamic weighting based on performance

    # Code block 7
    # Ensemble pipeline
    # Model versioning
    # A/B testing
    # Performance monitoring

    logger.info("All images generated successfully!")

if __name__ == "__main__":
    main()
