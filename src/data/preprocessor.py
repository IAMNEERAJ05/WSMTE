"""
src/data/preprocessor.py
Wavelet denoising, scaling, and missing-value handling.
"""

import numpy as np
import pywt
import joblib
from sklearn.preprocessing import MinMaxScaler


def coif3_denoise(series, config):
    """
    Apply Coif3 wavelet denoising (level=1, soft threshold).
    Universal threshold via median absolute deviation (MAD).
    Returns denoised array of the same length as input.
    """
    values = np.asarray(series, dtype=float)
    coeffs = pywt.wavedec(values, config['wavelet'], level=config['wavelet_level'])
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(values)))
    coeffs[1:] = [
        pywt.threshold(c, threshold, mode=config['wavelet_mode'])
        for c in coeffs[1:]
    ]
    denoised = pywt.waverec(coeffs, config['wavelet'])
    return denoised[:len(values)]


def apply_denoising(df, config):
    """
    Apply coif3_denoise to Open, High, Low, Close, Volume independently.
    Adds columns Open_d, High_d, Low_d, Close_d, Volume_d to df.
    Returns modified df (in-place).
    """
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[f'{col}_d'] = coif3_denoise(df[col].values, config)
    return df


def apply_scaler(train_arr, val_arr, test_arr, save_path=None):
    """
    Fit MinMaxScaler on train_arr, transform val_arr and test_arr.
    Each column (feature) is scaled independently.

    Args:
        train_arr: np.ndarray shape (n_train, n_features)
        val_arr:   np.ndarray shape (n_val, n_features)
        test_arr:  np.ndarray shape (n_test, n_features)
        save_path: optional path to save fitted scaler (.pkl)

    Returns:
        (train_scaled, val_scaled, test_scaled, scaler)
    """
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_arr)
    val_scaled   = scaler.transform(val_arr)
    test_scaled  = scaler.transform(test_arr)

    if save_path is not None:
        joblib.dump(scaler, save_path)

    return train_scaled, val_scaled, test_scaled, scaler


def handle_missing_values(df, config):
    """
    Fill missing sentiment values with defaults from config.
      polarity_company → config['missing_polarity']    (0.0)
      polarity_market  → config['missing_polarity']    (0.0)
      subjectivity     → config['missing_subjectivity'] (0.5)
    Returns modified df.
    """
    df['polarity_company'] = df['polarity_company'].fillna(config['missing_polarity'])
    df['polarity_market']  = df['polarity_market'].fillna(config['missing_polarity'])
    df['subjectivity']     = df['subjectivity'].fillna(config['missing_subjectivity'])
    return df
