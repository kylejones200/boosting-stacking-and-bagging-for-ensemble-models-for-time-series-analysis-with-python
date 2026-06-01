"""Ordered lag features for ensemble models — DuckDB LAG() matrix."""

import duckdb
import numpy as np
import polars as pl


def create_ordered_features(data: pl.Series, lag: int = 5) -> pl.DataFrame:
    lag_exprs = ", ".join(
        f"LAG(target, {i}) OVER (ORDER BY idx) AS lag_{i}" for i in range(1, lag + 1)
    )
    pl.DataFrame({"target": data, "idx": list(range(data.len()))})
    return (
        duckdb.sql(f"""
        SELECT target, {lag_exprs}
        FROM df
        ORDER BY idx
    """)
        .pl()
        .drop_nulls()
    )


def generate_series(n: int, seed: int = 42) -> pl.Series:
    rng = np.random.default_rng(seed)
    return pl.Series("y", np.sin(np.arange(n) / 10).tolist() + rng.normal(0, 0.2, n).tolist())
