"""
src/sentiment/aggregator.py
Aggregate per-article sentiment to daily level and handle gap periods.
"""

import pandas as pd


def aggregate_company_daily(kotekar_sent):
    """
    Aggregate Kotekar per-article sentiment to per-trading-day means.

    Args:
        kotekar_sent: DataFrame with columns [date, polarity_company, subjectivity]

    Returns:
        DataFrame with columns [date, polarity_company, subjectivity]
    """
    daily = kotekar_sent.groupby('date').agg(
        polarity_company=('polarity_company', 'mean'),
        subjectivity=('subjectivity', 'mean'),
    ).reset_index()
    return daily


def aggregate_market_daily(kaggle1_pol, kaggle2_pol):
    """
    Combine Kaggle1 + Kaggle2 polarity and aggregate to daily means.
    No date overlap (Kaggle1 ends Apr 2021, Kaggle2 starts Jan 2022).

    Args:
        kaggle1_pol: DataFrame with columns [date, polarity_market]
        kaggle2_pol: DataFrame with columns [date, polarity_market]

    Returns:
        DataFrame with columns [date, polarity_market], sorted ascending
    """
    combined = pd.concat(
        [kaggle1_pol[['date', 'polarity_market']],
         kaggle2_pol[['date', 'polarity_market']]],
        ignore_index=True,
    ).sort_values('date').reset_index(drop=True)

    daily = combined.groupby('date').agg(
        polarity_market=('polarity_market', 'mean')
    ).reset_index()
    return daily


def fill_gap_period(df, config):
    """
    Ensure trading days in the gap period (May–Dec 2021) have polarity_market = 0.
    The left-join in merge_sources already fills NaN with 0 via fillna();
    this function is an explicit verification / post-merge safety net.

    Args:
        df:     merged DataFrame with 'date' and 'polarity_market' columns
        config: CONFIG dict (uses 'gap_start', 'gap_end', 'missing_polarity')

    Returns:
        df with gap period polarity_market forced to missing_polarity (0.0)
    """
    gap_start = pd.to_datetime(config['gap_start']).date()
    gap_end   = pd.to_datetime(config['gap_end']).date()
    mask = (df['date'] >= gap_start) & (df['date'] <= gap_end)
    df.loc[mask, 'polarity_market'] = config['missing_polarity']
    return df
