"""
src/data/feature_engineering.py
Technical indicators and sliding-window construction.
ALL indicators are computed on DENOISED prices (_d columns).
"""

import numpy as np
import pandas as pd


def compute_rsi(series, period=14):
    """RSI(period) on a pandas Series. Returns Series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, ema_fast=12, ema_slow=26):
    """MACD histogram = EMA(fast) - EMA(slow). Returns Series."""
    fast = series.ewm(span=ema_fast, adjust=False).mean()
    slow = series.ewm(span=ema_slow, adjust=False).mean()
    return fast - slow


def compute_bb_width(series, period=20, n_std=2):
    """Bollinger Band Width = (Upper - Lower) / Middle. Returns Series."""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (n_std * 2 * std) / sma


def compute_roc(series, period=5):
    """Rate of Change (%) over `period` days. Returns Series."""
    return series.pct_change(periods=period) * 100


def build_feature_matrix(df, config):
    """
    Apply all technical indicators to denoised columns in df.
    Assumes df already has Close_d, Volume_d (and Open_d, High_d, Low_d).
    Adds: RSI_d, MACD_d, BB_width_d, ROC_d.
    Returns df with new columns (in-place).
    """
    df['RSI_d']      = compute_rsi(df['Close_d'], period=config['rsi_period'])
    df['MACD_d']     = compute_macd(df['Close_d'],
                                    ema_fast=config['ema_fast'],
                                    ema_slow=config['ema_slow'])
    df['BB_width_d'] = compute_bb_width(df['Close_d'],
                                        period=config['bb_period'],
                                        n_std=config['bb_std'])
    df['ROC_d']      = compute_roc(df['Close_d'], period=config['roc_period'])
    return df


def create_windows(scaled_data, raw_close, window_size=5):
    """
    Build sliding windows from scaled feature data.

    For each index i in [window_size, len(scaled_data)):
      X[i]     = scaled_data[i-window_size : i]       shape [window_size, n_features]
      y_clf[i] = 1 if raw_close[i] > raw_close[i-1] else 0
      y_reg[i] = scaled_data[i][0]                    (scaled Close_d, index 0)

    Args:
        scaled_data: np.ndarray shape (n, n_features) — already scaled
        raw_close:   np.ndarray shape (n,) — raw (unscaled) Close prices for label
        window_size: int (default 5)

    Returns:
        X      np.ndarray float32 shape (n - window_size, window_size, n_features)
        y_clf  np.ndarray int32   shape (n - window_size,)  {0, 1}
        y_reg  np.ndarray float32 shape (n - window_size,)  scaled Close_d
    """
    X, y_clf, y_reg = [], [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size : i])
        y_clf.append(1 if raw_close[i] > raw_close[i - 1] else 0)
        y_reg.append(float(scaled_data[i][0]))
    return (
        np.array(X, dtype=np.float32),
        np.array(y_clf, dtype=np.int32),
        np.array(y_reg, dtype=np.float32),
    )


def generate_targets(raw_close):
    """
    Generate classification labels for the full array (no windowing).
    Returns int32 array of same length: 1 if close[t] > close[t-1] else 0.
    First element is 0 (no previous day).
    """
    close = np.asarray(raw_close)
    labels = np.zeros(len(close), dtype=np.int32)
    labels[1:] = (close[1:] > close[:-1]).astype(np.int32)
    return labels
