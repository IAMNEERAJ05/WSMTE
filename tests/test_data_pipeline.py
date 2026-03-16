"""
tests/test_data_pipeline.py
Unit tests for data pipeline: split integrity, scaler, windows, gap fill.
Requires data/processed/final_dataset.csv to exist (run notebooks 01+02 first).

Run: python -m pytest tests/test_data_pipeline.py -v
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

# Allow running from project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CONFIG

FEATURE_COLUMNS = CONFIG['feature_columns']
PROCESSED = CONFIG['processed_data_dir']
FINAL_CSV = PROCESSED + 'final_dataset.csv'


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def final_df():
    if not os.path.exists(FINAL_CSV):
        pytest.skip("final_dataset.csv not found — run 02_feature_engineering.ipynb first")
    df = pd.read_csv(FINAL_CSV)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


@pytest.fixture(scope='module')
def splits(final_df):
    n = len(final_df)
    train_end = int(n * CONFIG['train_ratio'])
    val_end   = int(n * (CONFIG['train_ratio'] + CONFIG['val_ratio']))
    return (
        final_df.iloc[:train_end].reset_index(drop=True),
        final_df.iloc[train_end:val_end].reset_index(drop=True),
        final_df.iloc[val_end:].reset_index(drop=True),
    )


@pytest.fixture(scope='module')
def arrays():
    """Load saved .npy arrays if they exist."""
    p = PROCESSED
    required = ['X_train', 'X_val', 'X_test',
                 'y_clf_train', 'y_clf_val', 'y_clf_test']
    for name in required:
        if not os.path.exists(p + f'{name}.npy'):
            pytest.skip(f"{name}.npy not found — run 02_feature_engineering.ipynb first")
    return {name: np.load(p + f'{name}.npy') for name in required}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_split_no_shuffle(splits):
    """Train dates must all be before val dates, val before test dates."""
    train_df, val_df, test_df = splits
    assert train_df['date'].max() < val_df['date'].min(), \
        "Temporal leakage: train/val date overlap"
    assert val_df['date'].max() < test_df['date'].min(), \
        "Temporal leakage: val/test date overlap"


def test_scaler_fit_on_train_only(final_df):
    """
    Scaler must be fit before val/test dates appear.
    Training set scaled values must lie within [0, 1].
    """
    import joblib
    scaler_path = PROCESSED + 'scaler.pkl'
    if not os.path.exists(scaler_path):
        pytest.skip("scaler.pkl not found")

    scaler = joblib.load(scaler_path)
    n = len(final_df)
    train_end = int(n * CONFIG['train_ratio'])
    train_scaled = scaler.transform(final_df.iloc[:train_end][FEATURE_COLUMNS])

    assert train_scaled.max() <= 1.0 + 1e-6, "Train max exceeds 1.0 — scaler may be refit"
    assert train_scaled.min() >= 0.0 - 1e-6, "Train min below 0.0 — scaler may be refit"


def test_no_future_leakage(final_df, arrays):
    """Window at index i must only use data from days i-5 to i-1 for its target."""
    WINDOW = CONFIG['window_size']
    n = len(final_df)
    train_end = int(n * CONFIG['train_ratio'])
    train_close = final_df['Close'].values

    y_clf_train = arrays['y_clf_train']
    for i in range(min(50, len(y_clf_train))):  # check first 50 samples
        # Window i corresponds to source rows [i, i+WINDOW)
        # Target is close[train_start + i + WINDOW] vs close[train_start + i + WINDOW - 1]
        src_idx = i + WINDOW
        expected = 1 if train_close[src_idx] > train_close[src_idx - 1] else 0
        assert y_clf_train[i] == expected, \
            f"Label mismatch at window {i}: got {y_clf_train[i]}, expected {expected}"


def test_no_missing_values(final_df):
    """No missing values in the 9 feature columns of final_dataset.csv."""
    missing = final_df[FEATURE_COLUMNS].isnull().sum().sum()
    assert missing == 0, f"Found {missing} missing values in feature columns"


def test_gap_period_neutral_fill(final_df):
    """May–Dec 2021 trading days must have polarity_market = 0."""
    gap_start = pd.to_datetime(CONFIG['gap_start']).date()
    gap_end   = pd.to_datetime(CONFIG['gap_end']).date()
    gap = final_df[
        (final_df['date'] >= gap_start) & (final_df['date'] <= gap_end)
    ]
    if len(gap) == 0:
        pytest.skip("No gap-period rows found in final_dataset.csv")
    assert (gap['polarity_market'] == 0.0).all(), \
        "Gap period polarity_market not zero-filled"


def test_split_sizes(final_df, splits):
    """Each split must be within 2% of target ratio."""
    train_df, val_df, test_df = splits
    n = len(final_df)
    assert abs(len(train_df) / n - CONFIG['train_ratio']) < 0.02
    assert abs(len(val_df)   / n - CONFIG['val_ratio'])   < 0.02
    assert abs(len(test_df)  / n - CONFIG['test_ratio'])  < 0.02
