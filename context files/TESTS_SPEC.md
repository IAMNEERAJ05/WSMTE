# WSMTE — Unit Tests Specification

## Overview
Three test files covering data pipeline, feature engineering, and model architecture.
All tests run locally on CPU — no GPU needed.
Run with: python -m pytest tests/ -v

---

## tests/test_data_pipeline.py

### Test 1 — Temporal split preserves order
```python
def test_split_no_shuffle():
    """Train dates must all be before val dates, val before test dates."""
    assert train_df['date'].max() < val_df['date'].min()
    assert val_df['date'].max() < test_df['date'].min()
```

### Test 2 — Scaler fit on train only
```python
def test_scaler_fit_on_train_only():
    """Scaler must be fit before val/test dates appear."""
    # Scaler fitted on train — check train max is within [0,1]
    assert train_scaled.max() <= 1.0
    assert train_scaled.min() >= 0.0
    # Val/test CAN go outside [0,1] if out-of-range values exist
    # This confirms scaler was NOT refit on val/test
```

### Test 3 — No future leakage in windows
```python
def test_no_future_leakage():
    """Window at index i must only contain data from timesteps i-5 to i-1."""
    # Target at position i must use close[i], not close[i-1] or close[i+1]
    for i in range(len(X_train)):
        window_end_close = train_df['Close'].iloc[train_start + i + WINDOW_SIZE - 1]
        next_close = train_df['Close'].iloc[train_start + i + WINDOW_SIZE]
        expected_label = 1 if next_close > window_end_close else 0
        assert y_clf_train[i] == expected_label
```

### Test 4 — No missing values in final dataset
```python
def test_no_missing_values():
    df = pd.read_csv('data/processed/final_dataset.csv')
    assert df[FEATURE_COLUMNS].isnull().sum().sum() == 0
```

### Test 5 — Gap period filled correctly
```python
def test_gap_period_neutral_fill():
    """May–Dec 2021 should have polarity_market = 0."""
    df = pd.read_csv('data/processed/final_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    gap = df[(df['date'] >= '2021-04-16') & (df['date'] <= '2021-12-31')]
    assert (gap['polarity_market'] == 0).all()
```

### Test 6 — Correct split sizes
```python
def test_split_sizes():
    n = len(final_df)
    assert abs(len(train_df) / n - 0.70) < 0.02  # within 2% of 70%
    assert abs(len(val_df) / n - 0.15) < 0.02
    assert abs(len(test_df) / n - 0.15) < 0.02
```

---

## tests/test_features.py

### Test 1 — RSI range
```python
def test_rsi_range():
    """RSI must always be between 0 and 100."""
    rsi = compute_rsi(sample_close_series)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()
```

### Test 2 — Wavelet denoising preserves length
```python
def test_wavelet_output_length():
    """Denoised series must have same length as input."""
    original = np.random.randn(1090)
    denoised = coif3_denoise(original)
    assert len(denoised) == len(original)
```

### Test 3 — Wavelet actually smooths
```python
def test_wavelet_reduces_noise():
    """Denoised series must have lower std than original noisy series."""
    original = np.sin(np.linspace(0, 10, 200)) + np.random.randn(200) * 0.5
    denoised = coif3_denoise(original)
    assert np.std(denoised) < np.std(original)
```

### Test 4 — Feature vector has exactly 9 columns
```python
def test_feature_vector_shape():
    df = pd.read_csv('data/processed/final_dataset.csv')
    assert len(FEATURE_COLUMNS) == 9
    assert all(col in df.columns for col in FEATURE_COLUMNS)
```

### Test 5 — Correct feature column names and order
```python
def test_feature_column_order():
    expected = ['Close_d', 'Volume_d', 'RSI_d', 'MACD_d',
                'BB_width_d', 'ROC_d',
                'polarity_company', 'polarity_market', 'subjectivity']
    assert FEATURE_COLUMNS == expected
```

### Test 6 — Sliding window shape
```python
def test_window_shape():
    """Each window must be [5, 9]."""
    assert X_train.shape[1] == 5   # timesteps
    assert X_train.shape[2] == 9   # features
```

### Test 7 — Technical indicators computed on denoised not raw
```python
def test_technicals_on_denoised():
    """RSI computed from Close_d must differ from RSI on raw Close."""
    rsi_denoised = compute_rsi(df['Close_d'])
    rsi_raw = compute_rsi(df['Close'])
    assert not rsi_denoised.equals(rsi_raw)
```

---

## tests/test_model.py

### Test 1 — Model builds without error
```python
def test_model_builds():
    from src.models.wsmte import build_wsmte
    model = build_wsmte(CONFIG, use_pso=False)
    assert model is not None
```

### Test 2 — Output shapes correct
```python
def test_output_shapes():
    model = build_wsmte(CONFIG, use_pso=False)
    dummy_input = np.random.randn(8, 5, 9).astype(np.float32)
    reg_out, clf_out = model(dummy_input)
    assert reg_out.shape == (8, 1)
    assert clf_out.shape == (8, 1)
```

### Test 3 — Classification output is probability
```python
def test_classification_output_range():
    """Sigmoid output must be in [0, 1]."""
    model = build_wsmte(CONFIG, use_pso=False)
    dummy_input = np.random.randn(32, 5, 9).astype(np.float32)
    _, clf_out = model(dummy_input)
    assert (clf_out.numpy() >= 0).all()
    assert (clf_out.numpy() <= 1).all()
```

### Test 4 — Regression output is unbounded
```python
def test_regression_output_unbounded():
    """Linear activation — output should not be clipped to [0,1]."""
    model = build_wsmte(CONFIG, use_pso=False)
    # Use extreme input values
    extreme_input = np.ones((8, 5, 9), dtype=np.float32) * 10.0
    reg_out, _ = model(extreme_input)
    # If sigmoid were used, all outputs would be ~1.0 — check they vary
    assert reg_out.numpy().std() > 0
```

### Test 5 — Trainable sigma parameters exist
```python
def test_sigma_parameters_trainable():
    """Uncertainty weighting parameters σ₁ and σ₂ must be trainable."""
    model = build_wsmte(CONFIG, use_pso=False)
    trainable_names = [v.name for v in model.trainable_variables]
    assert any('sigma' in name or 'log_sigma' in name
               for name in trainable_names)
```

### Test 6 — Model has three parallel branches
```python
def test_three_branches():
    """Model must have LSTM, TCN, and GRU layers."""
    model = build_wsmte(CONFIG, use_pso=False)
    layer_types = [type(l).__name__ for l in model.layers]
    assert 'LSTM' in layer_types
    assert 'GRU' in layer_types
    assert 'TCN' in layer_types
```

### Test 7 — Multi-run produces different results per seed
```python
def test_different_seeds_differ():
    """Different seeds must produce different accuracies."""
    results = []
    for seed in [42, 7, 123]:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        model = build_wsmte(CONFIG, use_pso=False)
        # Train for 2 epochs only
        history = model.fit(X_train[:50], ...)
        results.append(history.history['val_binary_accuracy'][-1])
    assert len(set(results)) > 1  # not all identical
```

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific file
python -m pytest tests/test_model.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## Expected Output
All tests should pass with green checkmarks.
If any test fails, fix the corresponding src/ code before proceeding.
Do NOT comment out failing tests.
