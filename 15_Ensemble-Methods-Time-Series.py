from pathlib import Path

from sklearn.ensemble import StackingRegressor, VotingRegressor


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parents[1]
    BASE_DIR / "data" / "energy_indicators.csv"
    VotingRegressor(
        [("arima", arima_model), ("lstm", lstm_model), ("prophet", prophet_model)]
    )
    StackingRegressor(
        estimators=[("arima", arima), ("lstm", lstm)],
        final_estimator=LinearRegression(),
    )


if __name__ == "__main__":
    main()
