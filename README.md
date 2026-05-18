# Ensemble Models for Ordered Time Series

This project demonstrates ensemble modeling approaches for ordered time series data.

## Business context

No single model dominates across all time series problems. ARIMA handles autocorrelation well but misses nonlinear patterns. Gradient boosting captures complex feature interactions but needs careful treatment of temporal ordering. Neural networks can model long-range dependencies but require large datasets and are hard to interpret.

Ensemble methods do not solve this by picking the best model. They combine multiple models so that each one compensates for the weaknesses of the others. The result is often more accurate and more robust than any individual model — not because the models are smarter, but because their errors are less correlated.

There are three core ensemble strategies: bagging, boosting, and stacking. Each has different strengths and different failure modes.

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Ensemble model functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- Model parameters (lag, train_size)
- Ensemble method (mean, median)
- Output settings

## Ensemble Methods

### Mean Ensemble
- Average predictions from multiple models
- Reduces variance
- Simple and effective

### Median Ensemble
- Median of predictions
- Robust to outliers
- Less sensitive to extreme predictions

## Caveats

- By default, generates synthetic time series data.
- Ensemble performance depends on model diversity.
- Ordered time series requires preserving sequence.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).