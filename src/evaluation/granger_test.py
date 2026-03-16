"""
src/evaluation/granger_test.py
Granger causality tests: does polarity Granger-cause Nifty50 returns?
Tests both company-level and market-level polarity at lags 1–5.
"""

import os
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def run_granger_tests(returns, polarity_company, polarity_market,
                      max_lag=5, significance=0.05, save_path=None):
    """
    Run Granger causality tests:
      H0: polarity does NOT Granger-cause returns.

    Tests performed (following Kotekar et al.):
      - polarity_company → returns
      - polarity_market  → returns

    Args:
        returns:           pd.Series or np.ndarray — daily log/pct returns
        polarity_company:  pd.Series or np.ndarray — daily company-level polarity
        polarity_market:   pd.Series or np.ndarray — daily market-level polarity
        max_lag:           int — max lag order (default 5)
        significance:      float — p-value threshold (default 0.05)
        save_path:         optional CSV path to save results table

    Returns:
        pd.DataFrame with columns:
          source, lag, ssr_ftest_pvalue, lrtest_pvalue, params_ftest_pvalue,
          significant (bool at given significance level)
    """
    results = []

    for source_name, polarity in [
        ('polarity_company', polarity_company),
        ('polarity_market',  polarity_market),
    ]:
        # Build bivariate time series: [returns, polarity]
        df = pd.DataFrame({
            'returns':  returns,
            'polarity': polarity,
        }).dropna().reset_index(drop=True)

        print(f"\nGranger causality: {source_name} → returns | "
              f"n={len(df)}, max_lag={max_lag}")

        try:
            test_results = grangercausalitytests(
                df[['returns', 'polarity']],
                maxlag=max_lag,
                verbose=False,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        for lag in range(1, max_lag + 1):
            r = test_results[lag][0]
            ssr_p    = r['ssr_ftest'][1]
            lr_p     = r['lrtest'][1]
            params_p = r['params_ftest'][1]
            sig      = ssr_p < significance

            results.append({
                'source':               source_name,
                'lag':                  lag,
                'ssr_ftest_pvalue':     round(ssr_p, 6),
                'lrtest_pvalue':        round(lr_p, 6),
                'params_ftest_pvalue':  round(params_p, 6),
                'significant':          sig,
            })

            marker = '✓' if sig else ' '
            print(f"  lag={lag}  ssr_F p={ssr_p:.4f}  {marker}")

    df_results = pd.DataFrame(results)

    if save_path and len(df_results) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_results.to_csv(save_path, index=False)
        print(f"\nGranger results saved → {save_path}")

    return df_results
