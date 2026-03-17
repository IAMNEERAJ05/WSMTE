# WSMTE — Data Pipeline Specification

## CRITICAL INSTRUCTION
Follow these steps in EXACT ORDER.
Do NOT compute technical indicators on raw prices.
Do NOT fit scaler on validation or test data.
Do NOT shuffle data at any point.

---

## Dataset Column Reference (VERIFIED)

### Kotekar GitHub Dataset (company-level)
```
Columns: datePublished, company, symbol, headline,
         description, articleBody, tags, author, url
NOTE: NO pre-computed polarity or subjectivity
      Must run FinBERT on (headline + first 2 sentences of articleBody)
      Must run mDeBERTa on articleBody (truncated to 512 tokens)
```

### Kaggle Dataset 1 (Jan 2017 – Apr 15, 2021, market-level)
```
Columns: Date, Title, URL, sentiment, confidence
Source:  Economic Times (economictimes.indiatimes.com)
NOTE: sentiment/confidence already exist but we rerun
      FinBERT on Title column for consistency with Dataset 2
Text column to use: Title
```

### Kaggle Dataset 2 (Jan 2022 – Jun 2025, market-level)
```
Columns: Archive, Date, Headline, Headline link
Source:  Economic Times (economictimes.indiatimes.com)
NOTE: No pre-computed sentiment
Text column to use: Headline
```

---

## Overview of Pipeline Steps

```
Step 1:  Download price data (Yahoo Finance)
Step 2:  Load all three news datasets
Step 3:  Run FinBERT on Kotekar → polarity_company
Step 4:  Run mDeBERTa on Kotekar → subjectivity
Step 5:  Run FinBERT on Dataset 1 Title → polarity_market (2020–2021)
Step 6:  Run FinBERT on Dataset 2 Headline → polarity_market (2022–2024)
Step 7:  Aggregate all sentiment to daily level
Step 8:  Merge all sources by trading date
Step 9:  Apply Coif3 wavelet denoising to OHLCV
Step 10: Compute technical indicators on DENOISED prices
Step 11: Build final feature dataframe
Step 12: Verify no missing values
Step 13: Temporal train/val/test split (70/15/15)
Step 14: Fit MinMaxScaler on train only, transform all splits
Step 15: Build sliding windows
Step 16: Generate target variables
Step 17: Check class imbalance
```

---

## Step 1 — Download Price Data

```python
import yfinance as yf
import pandas as pd

nifty = yf.download('^NSEI', start='2020-01-01', end='2024-04-23')
nifty = nifty[['Open', 'High', 'Low', 'Close', 'Volume']]
nifty.index = pd.to_datetime(nifty.index)
nifty = nifty.sort_index()
nifty.to_csv('data/raw/nifty50_ohlcv.csv')
print(f"Price data shape: {nifty.shape}")  # expected ~(1090, 5)
```

---

## Step 2 — Load News Datasets

```python
# Kotekar dataset
kotekar = pd.read_csv('data/raw/kotekar_news.csv')
kotekar['date'] = pd.to_datetime(kotekar['datePublished']).dt.date
kotekar = kotekar[
    (kotekar['date'] >= pd.to_datetime('2020-01-01').date()) &
    (kotekar['date'] <= pd.to_datetime('2024-04-23').date())
]
print(f"Kotekar shape: {kotekar.shape}")

# Kaggle Dataset 1 — text column is 'Title'
kaggle1 = pd.read_csv('data/raw/kaggle_news_1.csv')
kaggle1['date'] = pd.to_datetime(kaggle1['Date']).dt.date
kaggle1 = kaggle1[
    (kaggle1['date'] >= pd.to_datetime('2020-01-01').date()) &
    (kaggle1['date'] <= pd.to_datetime('2021-04-15').date())
]
print(f"Kaggle1 shape: {kaggle1.shape}")

# Kaggle Dataset 2 — text column is 'Headline'
kaggle2 = pd.read_csv('data/raw/kaggle_news_2.csv')
kaggle2['date'] = pd.to_datetime(kaggle2['Date']).dt.date
kaggle2 = kaggle2[
    (kaggle2['date'] >= pd.to_datetime('2022-01-01').date()) &
    (kaggle2['date'] <= pd.to_datetime('2024-04-23').date())
]
print(f"Kaggle2 shape: {kaggle2.shape}")
```

---

## Steps 3–6 — FinBERT + mDeBERTa Inference (Run on Kaggle GPU)

### Load Models

```python
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)
import torch

# FinBERT
FINBERT_MODEL = 'ProsusAI/finbert'
finbert_tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
finbert_model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
finbert_model.eval()

# mDeBERTa for subjectivity
MDEBERTA_MODEL = 'cross-encoder/nli-deberta-v3-small'
deberta_tokenizer = AutoTokenizer.from_pretrained(MDEBERTA_MODEL)
deberta_model = AutoModelForSequenceClassification.from_pretrained(MDEBERTA_MODEL)
deberta_model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
finbert_model = finbert_model.to(device)
deberta_model = deberta_model.to(device)
print(f"Using device: {device}")
```

### FinBERT Polarity Function

```python
def get_finbert_polarity(texts, batch_size=32, max_length=512):
    """
    Compute continuous polarity in [-1, 1].
    polarity = P(positive) - P(negative)
    ProsusAI/finbert label order: positive=0, negative=1, neutral=2
    """
    polarities = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = finbert_tokenizer(
            batch, return_tensors='pt', padding=True,
            truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        polarity = (probs[:, 0] - probs[:, 1]).cpu().numpy()
        polarities.extend(polarity.tolist())
        if i % 500 == 0:
            print(f"  FinBERT processed {i}/{len(texts)}")
    return polarities
```

### mDeBERTa Subjectivity Function

```python
def get_subjectivity(texts, batch_size=16, max_length=512):
    """
    Compute subjectivity score in [0, 1].
    Uses NLI entailment probability as subjectivity proxy.
    Higher score = more subjective/opinionated text.
    """
    scores = []
    hypothesis = "This text expresses a personal opinion or subjective view."
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = deberta_tokenizer(
            batch,
            [hypothesis] * len(batch),
            return_tensors='pt', padding=True,
            truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            outputs = deberta_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        subjectivity = probs[:, 2].cpu().numpy()  # index 2 = entailment
        scores.extend(subjectivity.tolist())
        if i % 200 == 0:
            print(f"  mDeBERTa processed {i}/{len(texts)}")
    return scores
```

### Step 3 — FinBERT on Kotekar (polarity_company)

```python
import re

def prepare_kotekar_text(row):
    """
    headline + first 2 sentences of articleBody.
    Best practice: headline captures core sentiment,
    first 2 sentences provide context.
    Keeps within 512 token limit.
    """
    headline = str(row['headline']).strip()
    body = str(row['articleBody']).strip()
    sentences = re.split(r'(?<=[.!?])\s+', body)
    first_two = ' '.join(sentences[:2])
    return f"{headline}. {first_two}"

print("Preparing Kotekar texts for FinBERT...")
kotekar['text_for_finbert'] = kotekar.apply(prepare_kotekar_text, axis=1)

print("Running FinBERT on Kotekar dataset (~15-20 min)...")
kotekar['polarity_company'] = get_finbert_polarity(
    kotekar['text_for_finbert'].tolist(), batch_size=32)
print(f"Polarity range: {kotekar['polarity_company'].min():.3f} "
      f"to {kotekar['polarity_company'].max():.3f}")
```

### Step 4 — mDeBERTa on Kotekar (subjectivity)

```python
print("Running mDeBERTa on Kotekar articleBody (~30-45 min)...")
kotekar['subjectivity'] = get_subjectivity(
    kotekar['articleBody'].fillna('').tolist(), batch_size=16)
print(f"Subjectivity range: {kotekar['subjectivity'].min():.3f} "
      f"to {kotekar['subjectivity'].max():.3f}")

# Save immediately after computation
kotekar[['date', 'company', 'symbol', 'polarity_company', 'subjectivity']]\
    .to_csv('data/finbert_outputs/kotekar_sentiment.csv', index=False)
print("Saved kotekar_sentiment.csv")
```

### Step 5 — FinBERT on Kaggle Dataset 1 (Title column)

```python
print("Running FinBERT on Kaggle Dataset 1 Title column (~10-15 min)...")
kaggle1['polarity_market'] = get_finbert_polarity(
    kaggle1['Title'].fillna('').tolist(), batch_size=32)
kaggle1[['date', 'polarity_market']]\
    .to_csv('data/finbert_outputs/kaggle1_polarity.csv', index=False)
print(f"Saved kaggle1_polarity.csv. Shape: {kaggle1.shape}")
```

### Step 6 — FinBERT on Kaggle Dataset 2 (Headline column)

```python
print("Running FinBERT on Kaggle Dataset 2 Headline column (~15-20 min)...")
kaggle2['polarity_market'] = get_finbert_polarity(
    kaggle2['Headline'].fillna('').tolist(), batch_size=32)
kaggle2[['date', 'polarity_market']]\
    .to_csv('data/finbert_outputs/kaggle2_polarity.csv', index=False)
print(f"Saved kaggle2_polarity.csv. Shape: {kaggle2.shape}")
```

---

## Step 7 — Aggregate All Sentiment to Daily Level

```python
# Company-level: mean per trading day
company_daily = kotekar.groupby('date').agg(
    polarity_company=('polarity_company', 'mean'),
    subjectivity=('subjectivity', 'mean')
).reset_index()
print(f"Company daily: {company_daily.shape}")

# Market-level: combine Kaggle 1 + 2, no date overlap
# Kaggle1 covers up to Apr 15, 2021, Kaggle2 from Jan 2022
# Gap May–Dec 2021 handled in Step 8
market_combined = pd.concat([
    kaggle1[['date', 'polarity_market']],
    kaggle2[['date', 'polarity_market']]
]).sort_values('date').reset_index(drop=True)

market_daily = market_combined.groupby('date').agg(
    polarity_market=('polarity_market', 'mean')
).reset_index()
print(f"Market daily: {market_daily.shape}")
```

---

## Step 8 — Merge All Sources by Trading Date

```python
price_df = pd.read_csv('data/raw/nifty50_ohlcv.csv')
price_df['date'] = pd.to_datetime(price_df['Date']).dt.date

# Merge company sentiment
df = price_df.merge(company_daily, on='date', how='left')

# Merge market sentiment
df = df.merge(market_daily, on='date', how='left')

# Fill gap period + any other missing
df['polarity_market'] = df['polarity_market'].fillna(0.0)
df['polarity_company'] = df['polarity_company'].fillna(0.0)
df['subjectivity'] = df['subjectivity'].fillna(0.5)

df = df.sort_values('date').reset_index(drop=True)
print(f"Merged shape: {df.shape}")
df.to_csv('data/processed/merged_data.csv', index=False)
```

---

## Step 9 — Wavelet Denoising (MUST happen before Step 10)

```python
import pywt
import numpy as np

def coif3_denoise(series):
    """
    Coif3, level=1, soft thresholding.
    Universal threshold via median absolute deviation.
    """
    coeffs = pywt.wavedec(series, 'coif3', level=1)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(series)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft')
                  for c in coeffs[1:]]
    return pywt.waverec(coeffs, 'coif3')[:len(series)]

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[f'{col}_d'] = coif3_denoise(df[col].values)
    print(f"Denoised {col}")
```

---

## Step 10 — Technical Indicators (on DENOISED prices ONLY)

```python
# RSI(14)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI_d'] = compute_rsi(df['Close_d'])

# MACD histogram
ema12 = df['Close_d'].ewm(span=12, adjust=False).mean()
ema26 = df['Close_d'].ewm(span=26, adjust=False).mean()
df['MACD_d'] = ema12 - ema26

# Bollinger Band Width
sma20 = df['Close_d'].rolling(20).mean()
std20 = df['Close_d'].rolling(20).std()
df['BB_width_d'] = ((sma20 + 2*std20) - (sma20 - 2*std20)) / sma20

# ROC(5)
df['ROC_d'] = df['Close_d'].pct_change(periods=5) * 100
```

---

## Step 11 — Build Final Feature Dataframe

```python
FEATURE_COLUMNS = [
    'Close_d', 'Volume_d', 'RSI_d', 'MACD_d',
    'BB_width_d', 'ROC_d',
    'polarity_company', 'polarity_market', 'subjectivity'
]

# Drop warmup rows (26 days for MACD)
df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
print(f"After warmup drop: {df.shape}")

df[['date'] + FEATURE_COLUMNS + ['Close']]\
    .to_csv('data/processed/final_dataset.csv', index=False)
```

---

## Step 12 — Verify No Missing Values

```python
assert df[FEATURE_COLUMNS].isnull().sum().sum() == 0, \
    "Missing values found — fix pipeline before continuing"
print(f"Verified. Final shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
```

---

## Step 13 — Temporal Split (NO shuffling)

```python
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_df = df.iloc[:train_end].reset_index(drop=True)
val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
test_df  = df.iloc[val_end:].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
assert train_df['date'].max() < val_df['date'].min()
assert val_df['date'].max() < test_df['date'].min()
print("Split verified — no temporal leakage")
```

---

## Step 14 — Scaling (fit on train ONLY)

```python
from sklearn.preprocessing import MinMaxScaler
import joblib

scaler = MinMaxScaler()  # per-feature by default

train_scaled = scaler.fit_transform(train_df[FEATURE_COLUMNS])
val_scaled   = scaler.transform(val_df[FEATURE_COLUMNS])
test_scaled  = scaler.transform(test_df[FEATURE_COLUMNS])

joblib.dump(scaler, 'data/processed/scaler.pkl')
print("Scaler saved")
```

---

## Step 15 — Sliding Windows

```python
WINDOW_SIZE = 5

def create_windows(scaled_data, raw_close, window_size=5):
    X, y_clf, y_reg = [], [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y_clf.append(1 if raw_close[i] > raw_close[i-1] else 0)
        y_reg.append(scaled_data[i][0])  # scaled Close_d
    return np.array(X), np.array(y_clf), np.array(y_reg)

X_train, y_clf_train, y_reg_train = create_windows(
    train_scaled, train_df['Close'].values)
X_val,   y_clf_val,   y_reg_val   = create_windows(
    val_scaled,   val_df['Close'].values)
X_test,  y_clf_test,  y_reg_test  = create_windows(
    test_scaled,  test_df['Close'].values)

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

for name, arr in [
    ('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
    ('y_clf_train', y_clf_train), ('y_clf_val', y_clf_val), ('y_clf_test', y_clf_test),
    ('y_reg_train', y_reg_train), ('y_reg_val', y_reg_val), ('y_reg_test', y_reg_test)
]:
    np.save(f'data/processed/{name}.npy', arr)
print("All arrays saved")
```

---

## Step 16 — Class Imbalance Check

```python
import json
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

counts = Counter(y_clf_train)
ratio = max(counts.values()) / min(counts.values())
print(f"Up: {counts[1]/len(y_clf_train)*100:.1f}% | "
      f"Down: {counts[0]/len(y_clf_train)*100:.1f}% | Ratio: {ratio:.2f}")

if ratio > 1.5:  # exceeds 60:40
    weights = compute_class_weight(
        'balanced', classes=np.array([0, 1]), y=y_clf_train)
    class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}
    print(f"Applying class weights: {class_weight_dict}")
else:
    class_weight_dict = None
    print("Balanced — no class weighting needed")

with open('data/processed/class_weights.json', 'w') as f:
    json.dump(class_weight_dict, f)
```

---

## Final Expected Shapes

```
X_train:      (~730, 5, 9)   float32
X_val:        (~155, 5, 9)   float32
X_test:       (~155, 5, 9)   float32
y_clf_*:      (~n,)          int32    {0, 1}
y_reg_*:      (~n,)          float32
```

---

## FinBERT + mDeBERTa Runtime Estimates (Kaggle T4 2x)

```
Kotekar FinBERT  (~2,719 articles):   15–20 min
Kotekar mDeBERTa (~2,719 articles):   30–45 min
Kaggle1 FinBERT  (~varies):           10–15 min
Kaggle2 FinBERT  (~varies):           15–20 min
──────────────────────────────────────────────
Total:                                70–100 min
```

Save output CSV after EACH dataset — do not wait for all four to finish.
