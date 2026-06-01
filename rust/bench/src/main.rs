use boosting_stacking_and_bagging_for_ensemble_models_for_time_series_analysis_with_python_core::ensemble_mean;

fn main() {
    let n_models = 8usize;
    let n_samples = 2000usize;
    let preds: Vec<f64> = (0..n_models * n_samples).map(|i| (i as f64 * 0.001).sin()).collect();
    for _ in 0..5000 {
        let _ = ensemble_mean(&preds, n_models, n_samples);
    }
}
