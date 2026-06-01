#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import ensemble_mean  # noqa: E402

def main() -> None:
    nm, ns = 8, 2000
    preds = np.ascontiguousarray(np.sin(np.arange(nm * ns) * 0.001))
    t0 = time.perf_counter()
    for _ in range(200):
        ensemble_mean(preds, nm, ns)
    py_s = time.perf_counter() - t0
    try:
        import boosting_stacking_and_bagging_for_ensemble_models_for_time_series_analysis_with_python_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(preds, nm, ns, 2000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(ensemble_mean(preds, nm, ns), np.asarray(rs.ensemble_mean_py(preds, nm, ns)), rtol=1e-10)
    print("Correctness: OK")

if __name__ == "__main__":
    main()
