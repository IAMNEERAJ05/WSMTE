"""
src/data/loader.py
Load and merge all raw data sources for WSMTE.
"""

import pandas as pd


def load_price_data(config):
    """Load nifty50_ohlcv.csv and return DataFrame with a 'date' column (datetime.date)."""
    df = pd.read_csv(config['raw_data_dir'] + 'nifty50_ohlcv.csv')
    df['date'] = pd.to_datetime(df['Date']).dt.date
    df = df.sort_values('date').reset_index(drop=True)
    return df


def load_kotekar_sentiment(config):
    """
    Load kotekar_sentiment.csv (output of FinBERT + mDeBERTa inference).
    Columns: date, company, symbol, polarity_company, subjectivity
    """
    df = pd.read_csv(config['kotekar_sentiment_file'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def load_market_sentiment(config):
    """
    Load kaggle1_polarity.csv and kaggle2_polarity.csv, combine into one DataFrame.
    Columns: date, polarity_market
    Gap period May–Dec 2021 is absent here; filled in merge_sources().
    """
    k1 = pd.read_csv(config['kaggle1_polarity_file'])
    k1['date'] = pd.to_datetime(k1['date']).dt.date

    k2 = pd.read_csv(config['kaggle2_polarity_file'])
    k2['date'] = pd.to_datetime(k2['date']).dt.date

    combined = pd.concat(
        [k1[['date', 'polarity_market']], k2[['date', 'polarity_market']]],
        ignore_index=True
    ).sort_values('date').reset_index(drop=True)

    return combined


def merge_sources(price_df, kotekar_sent, market_combined, config):
    """
    Aggregate sentiment to daily level and left-join onto trading dates.

    Steps:
      1. Daily mean polarity_company + polarity_company_max + subjectivity from kotekar_sent
      2. Daily mean polarity_market from market_combined
      3. Left join both onto price_df (trading days are the anchor)
      4. Fill missing values:
           polarity_market      → 0.0  (covers gap May–Dec 2021 + any other missing)
           polarity_company     → 0.0
           polarity_company_max → 0.0
           subjectivity         → 0.5

    Returns merged DataFrame with all columns intact.
    """
    # Aggregate company sentiment — mean, max-absolute (signed), and mean subjectivity
    company_daily = kotekar_sent.groupby('date').agg(
        polarity_company=('polarity_company', 'mean'),
        polarity_company_max=('polarity_company',
                              lambda x: x.loc[x.abs().idxmax()]),
        subjectivity=('subjectivity', 'mean')
    ).reset_index()

    # Aggregate market sentiment
    market_daily = market_combined.groupby('date').agg(
        polarity_market=('polarity_market', 'mean')
    ).reset_index()

    df = price_df.merge(company_daily, on='date', how='left')
    df = df.merge(market_daily, on='date', how='left')

    df['polarity_market']      = df['polarity_market'].fillna(config['missing_polarity'])
    df['polarity_company']     = df['polarity_company'].fillna(config['missing_polarity'])
    df['polarity_company_max'] = df['polarity_company_max'].fillna(config['missing_polarity'])
    df['subjectivity']         = df['subjectivity'].fillna(config['missing_subjectivity'])

    df = df.sort_values('date').reset_index(drop=True)
    return df
