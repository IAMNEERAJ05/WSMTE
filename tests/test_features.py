"""
tests/test_features.py
Unit tests for feature engineering: RSI, wavelet denoising, window shapes.

Run: python -m pytest tests/test_features.py -v
"""

import os
import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CONFIG
from src.data.feature_engineering import (
    compute_rsi,
    create_windows,
)
from src.data.preprocessor import coif3_denoise

FEATURE_COLUMNS = CONFIG['feature_columns']
PROCESSED       = CONFIG['processed_data_dir']
FINAL_CSV       = PROCESSED + 'final_dataset.csv'


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def sample_close():
    """Synthetic close price series for unit tests (no file required)."""
    np.random.seed(0)
    prices = 15000 + np.cumsum(np.random.randn(200) * 50)
    return pd.Series(prices)


@pytest.fixture(scope='module')
def final_df():
    if not os.path.exists(FINAL_CSV):
        pytest.skip("final_dataset.csv not found — run 02_feature_engineering.ipynb first")
    return pd.read_csv(FINAL_CSV)


@pytest.fixture(scope='module')
def X_train():
    path = PROCESSED + 'X_train.npy'
    if not os.path.exists(path):
        pytest.skip("X_train.npy not found — run 02_feature_engineering.ipynb first")
    return np.load(path)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_rsi_range(sample_close):
    """RSI must always be between 0 and 100."""
    rsi = compute_rsi(sample_close, period=CONFIG['rsi_period'])
    valid = rsi.dropna()
    assert len(valid) > 0, "RSI produced no valid values"
    assert (valid >= 0).all() and (valid <= 100).all(), \
        f"RSI out of [0,100]: min={valid.min():.2f}, max={valid.max():.2f}"


def test_wavelet_output_length():
    """Denoised series must have same length as input."""
    original = np.random.randn(1090)
    denoised = coif3_denoise(original, CONFIG)
    assert len(denoised) == len(original), \
        f"Length mismatch: input={len(original)}, denoised={len(denoised)}"


def test_wavelet_reduces_noise():
    """Denoised series must have lower std than the noisy original."""
    np.random.seed(42)
    signal  = np.sin(np.linspace(0, 10, 200))
    noisy   = signal + np.random.randn(200) * 0.5
    denoised = coif3_denoise(noisy, CONFIG)
    assert np.std(denoised) < np.std(noisy), \
        "Wavelet denoising did not reduce noise"


def test_feature_vector_shape(final_df):
    """final_dataset.csv must have exactly 11 feature columns."""
    assert len(FEATURE_COLUMNS) == 11
    for col in FEATURE_COLUMNS:
        assert col in final_df.columns, f"Missing feature column: {col}"


def test_feature_column_order():
    """Feature column names and ORDER must match the locked list."""
    expected = [
        'Close_d', 'Volume_d', 'RSI_d', 'MACD_d', 'BB_width_d', 'ROC_d',
        'polarity_company', 'polarity_company_max',
        'polarity_market', 'polarity_market_max',
        'subjectivity',
    ]
    assert FEATURE_COLUMNS == expected, \
        f"Column order mismatch.\n  Expected: {expected}\n  Got:      {FEATURE_COLUMNS}"


def test_window_shape(X_train):
    """Each window must have shape [window_size, n_features] = [5, 11]."""
    assert X_train.shape[1] == CONFIG['window_size'],  \
        f"Window timesteps: expected {CONFIG['window_size']}, got {X_train.shape[1]}"
    assert X_train.shape[2] == CONFIG['n_features'], \
        f"Window features: expected {CONFIG['n_features']}, got {X_train.shape[2]}"


def test_technicals_on_denoised(final_df):
    """
    RSI computed on Close_d must differ from RSI computed on raw Close.
    This confirms indicators use the denoised column.
    """
    assert 'Close_d' in final_df.columns and 'Close' in final_df.columns, \
        "Need both Close and Close_d columns"
    rsi_denoised = compute_rsi(final_df['Close_d'], period=CONFIG['rsi_period'])
    rsi_raw      = compute_rsi(final_df['Close'],   period=CONFIG['rsi_period'])
    assert not rsi_denoised.equals(rsi_raw), \
        "RSI on Close_d equals RSI on Close — technicals may be using raw prices"


def test_create_windows_shapes():
    """create_windows must return correct array shapes and dtypes."""
    n, n_feat, w = 50, 9, 5
    scaled_data = np.random.rand(n, n_feat).astype(np.float32)
    raw_close   = np.cumsum(np.random.randn(n)) + 15000
    X, y_clf, y_reg = create_windows(scaled_data, raw_close, window_size=w)
    assert X.shape    == (n - w, w, n_feat), f"X shape wrong: {X.shape}"
    assert y_clf.shape == (n - w,),          f"y_clf shape wrong: {y_clf.shape}"
    assert y_reg.shape == (n - w,),          f"y_reg shape wrong: {y_reg.shape}"
    assert X.dtype    == np.float32
    assert y_clf.dtype == np.int32
    assert y_reg.dtype == np.float32
    assert set(y_clf).issubset({0, 1}),      "y_clf contains values other than 0 and 1"
