import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from xgboost import XGBRegressor


def create_lag_features() -> None:
    np.random.seed(42)
    data = np.cumsum(np.random.randn(365))
    lag = 5
    X = np.array([data[i : i + lag] for i in range(len(data) - lag)])
    y = data[lag:]
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_train, X_test = (X[train_idx], X[test_idx])
    y_train, _y_test = (y[train_idx], y[test_idx])
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf.predict(X_test)


def plot_results() -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="True Data", color="blue")
    test_data_index = range(len(data) - len(y_test), len(data))
    plt.plot(test_data_index, y_pred, label="Predicted Data", color="red")
    plt.title("Bagging with Random Forest")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("bagging_random_forest.png")
    plt.show()
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    xgb.predict(X_test)


def plot_results_2() -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="True Data", color="blue")
    test_data_index = range(len(data) - len(y_test), len(data))
    plt.plot(test_data_index, y_pred_xgb, label="Predicted Data", color="red")
    plt.title("Boosting with XGBoost")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("boosting_xgboost.png")
    plt.show()
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),
        ("gbr", GradientBoostingRegressor(n_estimators=50, random_state=42)),
        ("svr", SVR(kernel="rbf")),
    ]
    meta_model = Ridge()
    stacking_model = StackingRegressor(
        estimators=base_models, final_estimator=meta_model, cv=5
    )
    stacking_model.fit(X_train, y_train)
    stacking_model.predict(X_test)


def plot_results_3() -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="True Data", color="blue")
    test_data_index = range(len(data) - len(y_test), len(data))
    plt.plot(test_data_index, y_pred_stack, label="Predicted Data", color="red")
    plt.title("Stacking Ensemble")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("stacking_ensemble.png")
    plt.show()


def main() -> None:
    create_lag_features()
    plot_results()
    plot_results_2()
    plot_results_3()


if __name__ == "__main__":
    main()
