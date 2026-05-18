import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import signalplot
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
np.random.seed(42)
signalplot.apply(font_family="serif")


@dataclass
class Config:
    csv_path: str = "data/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
    season: int = 12


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s


def rolling_origin_ensemble(y: pd.Series, cfg: Config):
    idx = np.arange(len(y))
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    ets_maes, sar_maes, ens_maes = [], [], []
    last = {}
    for tr, te in tscv.split(idx):
        end = tr[-1]
        y_tr = y.iloc[: end + 1]
        y_te = y.iloc[end + 1 : end + 1 + cfg.horizon]
        if len(y_te) == 0:
            continue
        # ETS seasonal additive
        ets = ExponentialSmoothing(
            y_tr, trend="add", seasonal="add", seasonal_periods=cfg.season
        ).fit(optimized=True)
        f_ets = ets.forecast(len(y_te)).to_numpy()
        # SARIMAX seasonal
        sar = sm.tsa.statespace.SARIMAX(
            y_tr,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, cfg.season),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        f_sar = sar.forecast(len(y_te)).to_numpy()
        # Simple average ensemble
        f_ens = (f_ets + f_sar) / 2.0
        ets_maes.append(mean_absolute_error(y_te.values, f_ets))
        sar_maes.append(mean_absolute_error(y_te.values, f_sar))
        ens_maes.append(mean_absolute_error(y_te.values, f_ens))
        last = {
            "true": y_te,
            "ETS": pd.Series(f_ets, index=y_te.index),
            "SARIMAX": pd.Series(f_sar, index=y_te.index),
            "Ensemble": pd.Series(f_ens, index=y_te.index),
        }
    return (
        float(np.mean(ets_maes)),
        float(np.mean(sar_maes)),
        float(np.mean(ens_maes)),
        last,
    )


def main(plot: bool = False):
    cfg = Config()
    y = load_series(cfg)
    ets_m, sar_m, ens_m, last = rolling_origin_ensemble(y, cfg)
    logger.info(f"ETS mean MAE: {ets_m}")
    logger.info(f"SARIMAX mean MAE: {sar_m}")
    logger.info(f"Ensemble mean MAE: {ens_m}")
    if plot:
        plt.figure(figsize=(9, 4))
        plt.plot(y.index, y.values, label="history", alpha=0.6)
        if last:
            for name in ["ETS", "SARIMAX", "Ensemble"]:
                plt.plot(last[name].index, last[name].values, label=f"{name} last fold")
        plt.legend()
        signalplot.save("eia_ensemble_last_fold.png")


if __name__ == "__main__":
    main()
