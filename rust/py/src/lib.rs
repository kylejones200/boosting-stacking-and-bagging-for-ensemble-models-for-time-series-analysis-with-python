use boosting_stacking_and_bagging_for_ensemble_models_for_time_series_analysis_with_python_core::ensemble_mean;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn ensemble_mean_py<'py>(
    py: Python<'py>,
    predictions: PyReadonlyArray1<f64>,
    n_models: usize,
    n_samples: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(ensemble_mean(predictions.as_slice()?, n_models, n_samples).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (predictions, n_models, n_samples, iterations=5000))]
fn bench_kernel_py(
    predictions: PyReadonlyArray1<f64>,
    n_models: usize,
    n_samples: usize,
    iterations: usize,
) -> PyResult<f64> {
    let buf = predictions.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = ensemble_mean(&buf, n_models, n_samples);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn boosting_stacking_and_bagging_for_ensemble_models_for_time_series_analysis_with_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ensemble_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
