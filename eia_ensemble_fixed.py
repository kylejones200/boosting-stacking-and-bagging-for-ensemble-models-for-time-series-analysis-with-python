import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

np.random.seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "/Users/k.jones/Downloads/medium-export-e6bf40a8b01915d7380f6f547e0dd25ddd791328d4d9fa3a77513e82e662373c/posts/2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
    season: int = 12


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
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
        ets = ExponentialSmoothing(y_tr, trend='add', seasonal='add', seasonal_periods=cfg.season).fit(optimized=True)
        f_ets = ets.forecast(len(y_te)).to_numpy()
        # SARIMAX seasonal
        sar = sm.tsa.statespace.SARIMAX(y_tr, order=(1,1,1), seasonal_order=(1,1,1,cfg.season), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        f_sar = sar.forecast(len(y_te)).to_numpy()
        # Simple average ensemble
        f_ens = (f_ets + f_sar) / 2.0
        ets_maes.append(mean_absolute_error(y_te.values, f_ets))
        sar_maes.append(mean_absolute_error(y_te.values, f_sar))
        ens_maes.append(mean_absolute_error(y_te.values, f_ens))
        last = {
            'true': y_te,
            'ETS': pd.Series(f_ets, index=y_te.index),
            'SARIMAX': pd.Series(f_sar, index=y_te.index),
            'Ensemble': pd.Series(f_ens, index=y_te.index),
        }
    return float(np.mean(ets_maes)), float(np.mean(sar_maes)), float(np.mean(ens_maes)), last


def main():
    cfg = Config()
    y = load_series(cfg)
    ets_m, sar_m, ens_m, last = rolling_origin_ensemble(y, cfg)
    print(f"ETS mean MAE: {ets_m}")
    print(f"SARIMAX mean MAE: {sar_m}")
    print(f"Ensemble mean MAE: {ens_m}")

    plt.figure(figsize=(9,4))
    plt.plot(y.index, y.values, label='history', alpha=0.6)
    if last:
        for name in ['ETS','SARIMAX','Ensemble']:
            plt.plot(last[name].index, last[name].values, label=f"{name} last fold")
    plt.legend()
    save_fig('eia_ensemble_last_fold.png')

if __name__ == '__main__':
    main()
