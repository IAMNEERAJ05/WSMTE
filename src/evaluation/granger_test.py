"""
src/evaluation/granger_test.py
Granger causality tests: does polarity Granger-cause Nifty50 log-returns?

Tests four sentiment sources (polarity_mean, polarity_max, polarity_std,
polarity_market) at lags 1–10, for both the full dataset and the COVID
sub-period (2020-01-01 – 2021-12-31).
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def run_granger_tests(df, kotekar_art, max_lag=10, significance=0.05,
                      save_path=None):
    """
    Run Granger causality tests for multiple sentiment sources and periods.

    H0: sentiment does NOT Granger-cause log-returns.

    Sources tested:
      - polarity_mean   : mean FinBERT polarity per day (article-level)
      - polarity_max    : signed polarity of the most-extreme article per day
      - polarity_std    : std of polarity per day (0 if only 1 article)
      - polarity_market : daily mean market-level polarity (from df)

    Periods:
      - full       : entire dataset
      - covid_only : 2020-01-01 – 2021-12-31

    Args:
        df:           DataFrame with columns [date, Close, polarity_market].
                      date may be str or datetime.date.
        kotekar_art:  DataFrame with columns [date, polarity_company]
                      at article level (one row per article).
                      date may be str or datetime.date.
        max_lag:      int — maximum lag order (default 10)
        significance: float — p-value threshold for 'significant' flag (default 0.05)
        save_path:    optional str — CSV path to save results table

    Returns:
        pd.DataFrame with columns:
          source, period, lag, ssr_ftest_pvalue, lrtest_pvalue, significant
    """
    # ── Prepare daily df ──────────────────────────────────────────────────
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Log returns (NaN for first row)
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # ── Aggregate kotekar to daily features ───────────────────────────────
    art = kotekar_art.copy()
    art['date'] = pd.to_datetime(art['date']).dt.date

    art_daily = art.groupby('date').agg(
        polarity_mean=('polarity_company', 'mean'),
        polarity_max=('polarity_company',
                      lambda x: x.loc[x.abs().idxmax()]),
        polarity_std=('polarity_company',
                      lambda x: float(x.std()) if len(x) > 1 else 0.0),
    ).reset_index()

    # Merge article-level features into df
    df = df.merge(art_daily, on='date', how='left')
    df['polarity_mean'] = df['polarity_mean'].fillna(0.0)
    df['polarity_max']  = df['polarity_max'].fillna(0.0)
    df['polarity_std']  = df['polarity_std'].fillna(0.0)

    # ── Define periods ────────────────────────────────────────────────────
    covid_start = pd.to_datetime('2020-01-01').date()
    covid_end   = pd.to_datetime('2021-12-31').date()

    periods = {
        'full':       df,
        'covid_only': df[
            (df['date'] >= covid_start) & (df['date'] <= covid_end)
        ].copy().reset_index(drop=True),
    }

    # ── Sources to test ───────────────────────────────────────────────────
    source_names = ['polarity_mean', 'polarity_max', 'polarity_std',
                    'polarity_market']

    # ── Run tests ─────────────────────────────────────────────────────────
    results = []

    for period_name, period_df in periods.items():
        for source_name in source_names:
            bivariate = pd.DataFrame({
                'returns':  period_df['returns'],
                'polarity': period_df[source_name],
            }).dropna().reset_index(drop=True)

            print(f"\nGranger: {source_name} → returns | "
                  f"period={period_name}, n={len(bivariate)}, max_lag={max_lag}")

            try:
                test_out = grangercausalitytests(
                    bivariate[['returns', 'polarity']],
                    maxlag=max_lag,
                    verbose=False,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            for lag in range(1, max_lag + 1):
                r = test_out[lag][0]
                ssr_p = r['ssr_ftest'][1]
                lr_p  = r['lrtest'][1]
                sig   = ssr_p < significance

                results.append({
                    'source':           source_name,
                    'period':           period_name,
                    'lag':              lag,
                    'ssr_ftest_pvalue': round(ssr_p, 6),
                    'lrtest_pvalue':    round(lr_p, 6),
                    'significant':      sig,
                })

                marker = '✓' if sig else ' '
                print(f"  lag={lag:2d}  ssr_F p={ssr_p:.4f}  {marker}")

    df_results = pd.DataFrame(results)

    if save_path and len(df_results) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_results.to_csv(save_path, index=False)
        print(f"\nGranger results saved → {save_path}")

    return df_results
