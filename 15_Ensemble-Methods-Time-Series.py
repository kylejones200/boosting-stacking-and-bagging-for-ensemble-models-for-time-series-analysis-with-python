from pathlib import Path

from sklearn.ensemble import StackingRegressor, VotingRegressor


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parents[1]

    data_path = BASE_DIR / "data" / "energy_indicators.csv"

    voting_model = VotingRegressor(
        [("arima", arima_model), ("lstm", lstm_model), ("prophet", prophet_model)]
    )

    stacking_model = StackingRegressor(
        estimators=[("arima", arima), ("lstm", lstm)],
        final_estimator=LinearRegression(),
    )


if __name__ == "__main__":
    main()
