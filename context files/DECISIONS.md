# WSMTE — Finalized Decisions

## CRITICAL INSTRUCTION
Every decision in this file is FINAL and LOCKED.
Do NOT suggest alternatives. Do NOT change parameters.
Do NOT add components not listed here.
If something seems improvable, note it as a comment but implement as specified.

---

## 1. Framework
- TensorFlow / Keras
- NOT PyTorch
- NOT any other framework

---

## 2. Dataset Decisions

### Price Data
- Source: Yahoo Finance, ticker ^NSEI
- Columns: Open, High, Low, Close, Volume
- Range: Jan 2020 – May 2024

### Company-Level Sentiment — Kotekar GitHub Dataset
- Columns available: datePublished, company, symbol, headline,
  description, articleBody, tags, author, url
- NO pre-computed polarity or subjectivity in this dataset
- Must run FinBERT on: headline + first 2 sentences of articleBody
- Must run mDeBERTa on: full articleBody (truncated to 512 tokens)
- FinBERT model: ProsusAI/finbert
- mDeBERTa model: cross-encoder/nli-deberta-v3-small
- Justification for different input lengths:
  FinBERT: polarity signal is front-loaded in financial articles
  mDeBERTa: subjectivity requires full context to distinguish opinion from fact
- Missing trading days: polarity = 0, subjectivity = 0.5

### Market-Level Sentiment
- Kaggle Dataset 1: Jan 2020 – Apr 2021
  Columns: Date, Title, URL, sentiment, confidence
  Text column to use: Title
  NOTE: existing sentiment/confidence ignored — rerun FinBERT for consistency
- Gap period May 2021 – Dec 2021: polarity_market = 0 (neutral fill)
- Kaggle Dataset 2: Jan 2022 – May 2024
  Columns: Archive, Date, Headline, Headline link
  Text column to use: Headline
- FinBERT model: ProsusAI/finbert (same as company-level)
- Aggregation: mean FinBERT polarity per trading day
- Duplicates at boundary: drop by date combination

### Rejected Datasets
- NifSent50: REJECTED — price-derived labels, irrelevant headlines, duplicates
- Do NOT use NifSent50 for anything

---

## 3. Feature Engineering Decisions

### Wavelet Denoising
- Wavelet: Coif3
- Level: 1
- Thresholding: soft
- Applied to: Open, High, Low, Close, Volume independently
- Applied FIRST — before any technical indicator computation
- Library: pywt

### Technical Indicators
- ALL computed on DENOISED prices, NOT raw prices
- RSI: 14-day period
- MACD histogram: EMA(12) − EMA(26)
- BB_width: (Upper − Lower) / Middle, 20-day SMA
- ROC: 5-day rate of change on denoised Close

### LOCKED Feature Vector (9 features, exact order)
```
[Close_d, Volume_d, RSI_d, MACD_d, BB_width_d, ROC_d,
 Polarity_company, Polarity_market, Subjectivity]
```

### Dropped Features — Do NOT Add These Back
- High_d: BB_width already captures High-Low range
- Low_d: same reason
- Open_d: marginal info beyond Close, High, Low
- EMA(12), EMA(26): already encoded inside MACD
- SMA(10), SMA(20): already used inside BB_width
- Stochastic Oscillator: redundant with RSI

### Sliding Window
- Window size: 5 days
- Input shape per sample: [5, 9]
- Warmup: drop first 26 trading days (MACD needs longest warmup)

### Scaling
- MinMaxScaler
- Per-feature (each of 9 columns scaled independently)
- Fit on training set ONLY
- Apply fitted scaler to val and test sets

### Target Variables
- Classification: y = 1 if Close_t+1 > Close_t else 0
- Regression: y = Close_t+1 (normalized by scaler)

---

## 4. Architecture Decisions

### Encoder Branches (all parallel, NOT sequential)
- LSTM: units=64, activation=tanh, recurrent_activation=sigmoid,
        dropout=0.2, recurrent_dropout=0.0, return_sequences=False
- GRU:  units=64, activation=tanh, recurrent_activation=sigmoid,
        dropout=0.2, recurrent_dropout=0.0, return_sequences=False
- TCN:  filters=64, kernel_size=2, dilations=[1,2,4],
        padding=causal, activation=relu, dropout=0.2,
        use_skip_connections=True, use_batch_norm=False,
        use_layer_norm=False, use_weight_norm=False

### Branch Merging
- Configs A–G: simple concatenation → 192-dim vector
- Config H only: PSO-weighted sum → w1×LSTM + w2×TCN + w3×GRU
  where w1+w2+w3=1, weights found by PSO on validation set

### Shared Dense Layer
- Dense(64, activation=relu)
- Dropout(0.2)

### Regression Head
- Dense(16, activation=relu)
- Dense(1, activation=linear)
- Loss: MSE

### Classification Head
- Dense(16, activation=relu)
- Dense(1, activation=sigmoid)
- Loss: BCE

### Combined Loss (MTL)
- Uncertainty weighting: Kendall et al. CVPR 2018
- Formula: L = (1/2σ₁²)×MSE + (1/2σ₂²)×BCE + log(σ₁) + log(σ₂)
- σ₁ and σ₂ are trainable parameters
- Do NOT use fixed alpha weighting
- Do NOT use Focal loss
- Do NOT use Huber loss

### Excluded Architectures — Do NOT Add These
- TKAN: no stable library, Kotekar's own contribution
- Transformer: needs large dataset, 1090 samples insufficient
- BiLSTM: uses future timesteps, illegal in financial forecasting
- SimpleRNN: vanishing gradient, weakest in Kotekar comparison
- No model-level stacking (no separate meta-learner)

---

## 5. Training Decisions

- Split: 70/15/15 temporal, NO shuffling
- Optimizer: Adam, lr=0.001
- Batch size: 32
- Max epochs: 100
- Early stopping: patience=15, monitor=val_loss, restore_best_weights=True
- LR scheduler: ReduceLROnPlateau(monitor=val_loss, factor=0.5,
                patience=7, min_lr=1e-6)
- Dropout: 0.2 on all branches and shared dense
- Class imbalance: apply class_weight to BCE if label ratio exceeds 60:40
- Config G + H: 30 runs with different random seeds
- Configs A–F: 10 runs with different random seeds
- Checkpoint: monitor val_loss for saving, log val_binary_accuracy separately

---

## 6. PSO Decisions (Config H only)

- Library: pyswarms
- Method: Two-stage (train first, then PSO search, then fine-tune)
- Stage 1: Train full model with simple concatenation
- Stage 2: Freeze encoder, PSO searches for best [w1, w2, w3]
- Stage 3: Unfreeze, fine-tune with PSO weights
- n_particles: 20
- PSO iterations: 50
- options: c1=0.5, c2=0.3, w=0.9
- Fitness function: negative validation accuracy (PSO minimizes)
- Weight normalization: softmax to ensure w1+w2+w3=1
- Only applies to Config H

---

## 7. Evaluation Decisions

### Classification Metrics (all configs)
- Accuracy, Balanced Accuracy, AUC-ROC, Precision, Recall, F1

### Regression Metrics (configs with regression head: E, F, G, H)
- RMSE, MAE, R²

### Statistical Reporting (Config H — 30 runs)
- Mean accuracy, Max accuracy, Std deviation
- Wilcoxon signed-rank test between Config B and Config H

### Must-Do Evaluations
- SHAP: DeepSHAP or GradientExplainer on shared encoder
- Granger causality: statsmodels, lags 1–5,
  test BOTH company-level AND market-level polarity
- Trading simulation: long-only strategy
  (long on predicted up-day, flat on predicted down-day)
  Follow Kotekar Algorithm 1
- Sharpe ratio: (return − 6% annual risk-free) / std daily returns
- Confusion matrix, AUC-ROC curve, loss curves per run

### Skipped (Future Work)
- Walk-forward cross validation
- PSO for configs A–G

---

## 8. Ablation Study

8 configs total. See ARCHITECTURE.md for full table.
- Configs A–G: simple concatenation, 10 runs each
- Config H: PSO weighting, 30 runs (final proposed model)
- All results saved to ablation_results.csv (one row per run)

### Ablation Feature Design (LOCKED)
Config A uses all technical features as baseline — NOT minimal features.
This ensures sentiment contribution is isolated cleanly.

| Config | Features | Purpose |
|--------|----------|---------|
| A | Close_d, Volume_d, RSI_d, MACD_d, BB_width_d, ROC_d | Technicals only — meaningful floor |
| B | A + polarity_company + subjectivity | Company sentiment contribution |
| C | B + polarity_market | Market sentiment contribution (Novelty 2) |
| D | All 9 features | Full feature set single-task confirmation |
| E | All 9 features, classification only | MTL ablation — no regression head |
| F | All 9 features, regression only | MTL ablation — no classification head |
| G | All 9 features, both heads, concat | Full WSMTE without PSO |
| H | All 9 features, both heads, PSO | Final proposed model |

---

## 9. Compute Split

### Runs on Kaggle T4 2x GPU
- notebooks/03_finbert_inference.ipynb
  (FinBERT on Kotekar + Kaggle1 + Kaggle2, mDeBERTa on Kotekar)
- notebooks/04_model_training.ipynb
- notebooks/05_ablation.ipynb

### Runs on Local PC (no GPU needed)
- notebooks/01_data_prep.ipynb
- notebooks/02_feature_engineering.ipynb
- notebooks/06_evaluation.ipynb

### Kaggle Setup
- Clone GitHub repo in Kaggle: !git clone https://github.com/[user]/WSMTE.git
- sys.path.append('/kaggle/working/WSMTE')
- All src/ code imported from repo
- Outputs downloaded to local results/ folder

---

## 10. Paper Decisions
- Target: IEEE conference, 8 pages
- Primary comparison: Config H vs Kotekar 0.5853
- Same dataset, same metric, same 30-run protocol
- Config B = starting point baseline (NOT a replication of Kotekar model)
- Related work: ~10–15 citations compiled during paper writing phase
