//! Row-wise mean of stacked model predictions (n_models x n_samples, row-major).

pub fn ensemble_mean(predictions: &[f64], n_models: usize, n_samples: usize) -> Vec<f64> {
    assert_eq!(predictions.len(), n_models * n_samples);
    let mut out = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        let mut sum = 0.0;
        for m in 0..n_models {
            sum += predictions[m * n_samples + s];
        }
        out.push(sum / n_models as f64);
    }
    out
}
