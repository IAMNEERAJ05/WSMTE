# WSMTE Project Overview

## Project Name
**WSMTE — Wavelet-Sentiment Multi-Task Ensemble**

## One-Line Summary
A novel deep learning architecture for next-day Nifty50 index return direction
prediction, combining wavelet-denoised technical features with multi-source
financial news sentiment using a parallel LSTM+TCN+GRU encoder with multi-task
regression and classification heads.

---

## Problem Statement
- Task: Binary classification — predict whether Nifty50 closes UP (1) or DOWN (0)
  the next trading day
- Why binary not 3-class: dataset size (~1090 trading days) too small for 3-class
- Date range: January 2020 – May 2024
- Primary benchmark: Kotekar et al. (IEEE Access 2026) best accuracy = 0.5853

---

## Three Reference Papers

### Paper 1 — Kotekar et al. (IEEE Access 2026) [PRIMARY BENCHMARK]
- Task: Binary classification, next-day Nifty50 direction
- Data: 2,719 Moneycontrol articles + Yahoo Finance OHLC
- Sentiment: FinBERT polarity + mDeBERTa subjectivity
- Models: SimpleRNN, GRU, LSTM, TKAN + PSO ensemble
- Best result: accuracy = 0.5853 (LSTM + polarity, 30 runs)
- Key findings: polarity Granger-causes returns at lags 3-5, subjectivity weakest feature
- Relation to WSMTE: direct extension — same dataset, same metric, same 30-run protocol

### Paper 2 — Khan et al. (JIK 2025)
- Task: Regression, next-day Nifty50 closing price
- Data: Yahoo Finance OHLCV
- Features: RSI, EMA, MACD, Stochastic, BB, SMA — computed on RAW prices
- Model: Single hidden layer FFNN (10 neurons, tanh), Levenberg-Marquardt, MATLAB
- Best result: test RMSE = 0.0109
- Relation to WSMTE: we borrow their indicator set but compute on DENOISED prices

### Paper 3 — Singh et al. (IEEE Access 2025)
- Task: Regression, next-day Nifty50 closing price
- Data: Yahoo Finance OHLCV, Jan 2010 – Dec 2023 (13 years)
- Preprocessing: Coif3 wavelet denoising (level 1, soft threshold) on ALL OHLCV
- Model: LSTM + CNN + TCN as base learners, Random Forest meta-learner (stacking)
- Best result: R² = 0.9993, RMSE = 0.003723
- Relation to WSMTE: we borrow Coif3 denoising but add sentiment + technicals + MTL

---

## Five Novelty Claims

1. Coif3 wavelet denoising applied BEFORE technical indicator computation
   → Singh et al. denoised before modeling but never used technicals
   → Khan et al. used technicals but on raw prices
   → WSMTE does both: denoise first, then compute all technicals from clean signal

2. Multi-source sentiment fusion: company-level + market-level
   → Kotekar used only company-specific Moneycontrol articles
   → WSMTE adds market-level sentiment from Kaggle financial headlines via FinBERT

3. Technical indicators fused with sentiment in same model
   → Paper 1 used sentiment only (no technicals)
   → Paper 3 used denoised OHLCV only (no sentiment, no technicals)
   → WSMTE combines all three: denoised prices + technicals + sentiment

4. Multi-task learning: shared encoder with regression + classification heads
   → No prior Nifty50 paper used MTL
   → Regression head teaches encoder price magnitude
   → Classification head teaches encoder direction
   → Both signals shape the shared encoder simultaneously

5. Parallel LSTM+TCN+GRU encoder with PSO branch weighting
   → Paper 3 used sequential stacking (independent models + RF meta-learner)
   → WSMTE uses parallel branches inside one model with PSO-learned weights

---

## Architecture Name Breakdown
- W  = Wavelet denoising (Coif3, level 1, soft threshold)
- S  = Sentiment (multi-source: company + market level)
- M  = Multi-task learning (regression + classification)
- T  = Temporal encoder (parallel LSTM + TCN + GRU)
- E  = Ensemble (PSO-weighted branch combination)

---

## Target Publication
- Venue: IEEE conference (specific venue decided after results)
- Length: 8 pages
- Comparison: Config H (full WSMTE) vs Kotekar 0.5853

---

## Important Notes for Claude Code
- This is a FINAL YEAR B.TECH PROJECT for publication
- ALL architecture and parameter decisions are FINALIZED — do not suggest alternatives
- Do not add models, features, or components not listed in DECISIONS.md
- Do not change hyperparameters without explicit instruction
- Refer to DECISIONS.md before making ANY implementation choice
- Refer to ARCHITECTURE.md for exact layer specifications
- Refer to DATA_PIPELINE.md for exact preprocessing steps
