"""
src/evaluation/metrics.py
Classification metrics, regression metrics, and Sharpe ratio.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute all classification metrics for binary direction prediction.

    Args:
        y_true:       np.ndarray int {0, 1}
        y_pred_proba: np.ndarray float in [0, 1] — P(up)
        threshold:    float (default 0.5) for converting proba to label

    Returns:
        dict with keys: accuracy, balanced_accuracy, auc, precision, recall, f1
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        'accuracy':          accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'auc':               roc_auc_score(y_true, y_pred_proba),
        'precision':         precision_score(y_true, y_pred, zero_division=0),
        'recall':            recall_score(y_true, y_pred, zero_division=0),
        'f1':                f1_score(y_true, y_pred, zero_division=0),
    }


def compute_regression_metrics(y_true, y_pred):
    """
    Compute regression metrics.

    Args:
        y_true: np.ndarray float — actual normalized Close_d
        y_pred: np.ndarray float — predicted normalized Close_d

    Returns:
        dict with keys: rmse, mae, r2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        'rmse': float(np.sqrt(np.mean(residuals ** 2))),
        'mae':  float(np.mean(np.abs(residuals))),
        'r2':   float(r2),
    }


def compute_sharpe_ratio(daily_returns, risk_free_rate=0.06):
    """
    Compute annualised Sharpe ratio for a trading strategy.

    Sharpe = (mean_daily_return - daily_risk_free) / std_daily_return * sqrt(252)

    Args:
        daily_returns:  np.ndarray of daily strategy returns (fraction, e.g. 0.01 = 1%)
        risk_free_rate: annual risk-free rate (default 6% as per DECISIONS.md)

    Returns:
        float — annualised Sharpe ratio
    """
    daily_returns = np.asarray(daily_returns, dtype=float)
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    std = daily_returns.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))
