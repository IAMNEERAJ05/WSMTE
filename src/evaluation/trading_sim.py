"""
src/evaluation/trading_sim.py
Long-only trading simulation following Kotekar et al. Algorithm 1.

Strategy:
  - Predicted UP  (y_pred=1): go LONG  → daily return = actual market return
  - Predicted DOWN (y_pred=0): stay FLAT → daily return = 0
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.metrics import compute_sharpe_ratio


def run_trading_simulation(y_pred_proba, actual_returns, risk_free_rate=0.06,
                           threshold=0.5, save_path=None):
    """
    Simulate a long-only strategy based on next-day direction predictions.

    Args:
        y_pred_proba:   np.ndarray float [n_test] — P(up) from model
        actual_returns: np.ndarray float [n_test] — actual daily returns
                        (e.g. (Close_t - Close_{t-1}) / Close_{t-1})
        risk_free_rate: float — annual risk-free rate (default 0.06 = 6%)
        threshold:      float — decision threshold (default 0.5)
        save_path:      optional path to save the simulation figure

    Returns:
        dict with keys:
          strategy_total_return   — cumulative strategy return
          buyhold_total_return    — cumulative buy-and-hold return
          strategy_sharpe         — annualised Sharpe ratio (strategy)
          buyhold_sharpe          — annualised Sharpe ratio (buy-and-hold)
          n_long_days             — number of days strategy goes long
          n_flat_days             — number of days strategy stays flat
          strategy_cum_returns    — np.ndarray cumulative strategy returns
          buyhold_cum_returns     — np.ndarray cumulative buy-and-hold returns
    """
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    actual = np.asarray(actual_returns, dtype=float)

    # Strategy returns: long on predicted up-days, 0 on predicted down-days
    strategy_returns = y_pred * actual

    # Cumulative returns
    strategy_cum = np.cumprod(1 + strategy_returns) - 1
    buyhold_cum  = np.cumprod(1 + actual) - 1

    # Sharpe ratios
    strategy_sharpe = compute_sharpe_ratio(strategy_returns, risk_free_rate)
    buyhold_sharpe  = compute_sharpe_ratio(actual,           risk_free_rate)

    results = {
        'strategy_total_return':  float(strategy_cum[-1]),
        'buyhold_total_return':   float(buyhold_cum[-1]),
        'strategy_sharpe':        strategy_sharpe,
        'buyhold_sharpe':         buyhold_sharpe,
        'n_long_days':            int(y_pred.sum()),
        'n_flat_days':            int((y_pred == 0).sum()),
        'strategy_cum_returns':   strategy_cum,
        'buyhold_cum_returns':    buyhold_cum,
    }

    _print_summary(results)

    if save_path:
        _plot_simulation(strategy_cum, buyhold_cum, save_path)

    return results


def _print_summary(r):
    print("\n── Trading Simulation Summary ──────────────────────")
    print(f"  Strategy total return : {r['strategy_total_return']*100:.2f}%")
    print(f"  Buy & Hold return     : {r['buyhold_total_return']*100:.2f}%")
    print(f"  Strategy Sharpe ratio : {r['strategy_sharpe']:.4f}")
    print(f"  Buy & Hold Sharpe     : {r['buyhold_sharpe']:.4f}")
    print(f"  Long days             : {r['n_long_days']}")
    print(f"  Flat days             : {r['n_flat_days']}")
    print("────────────────────────────────────────────────────")


def _plot_simulation(strategy_cum, buyhold_cum, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(strategy_cum * 100, label='WSMTE Strategy', linewidth=1.5)
    plt.plot(buyhold_cum  * 100, label='Buy & Hold',     linewidth=1.5, linestyle='--')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return (%)')
    plt.title('WSMTE Long-Only Strategy vs Buy & Hold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trading simulation plot saved → {save_path}")
