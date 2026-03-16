# WSMTE — Task Tracker

## How To Use This File
- Check off tasks as completed: change [ ] to [x]
- Never skip a task — each depends on the previous
- If a Kaggle session dies, restart from last unchecked task
- Update this file and push to GitHub after each session

---

## DAY 1 — Data Collection + FinBERT/mDeBERTa Inference

### Local PC Tasks
- [ ] 1.1  Create full folder structure (see FOLDER_STRUCTURE.md)
- [ ] 1.2  Initialize GitHub repo, push empty structure
- [ ] 1.3  Create requirements.txt
- [ ] 1.4  Create config/config.py
- [ ] 1.5  Download Nifty50 OHLCV via yfinance → data/raw/nifty50_ohlcv.csv
- [ ] 1.6  Verify Kotekar dataset columns: datePublished, company, symbol,
           headline, description, articleBody, tags, author, url
- [ ] 1.7  Verify Kaggle Dataset 1 columns: Date, Title, URL, sentiment, confidence
- [ ] 1.8  Verify Kaggle Dataset 2 columns: Archive, Date, Headline, Headline link
- [ ] 1.9  Check for obvious quality issues (blank rows, bad dates) in all 3 datasets
- [ ] 1.10 Push all raw data and config files to GitHub

### Kaggle GPU Tasks (notebooks/03_finbert_inference.ipynb)
- [ ] 1.11 Clone GitHub repo in Kaggle notebook
- [ ] 1.12 Verify GPU enabled (T4 x2)
- [ ] 1.13 Load FinBERT (ProsusAI/finbert) and mDeBERTa models
- [ ] 1.14 Run FinBERT on Kotekar (headline + first 2 sentences of articleBody)
           → polarity_company column
- [ ] 1.15 Run mDeBERTa on Kotekar (full articleBody, truncated 512 tokens)
           → subjectivity column
- [ ] 1.16 Save → data/finbert_outputs/kotekar_sentiment.csv
- [ ] 1.17 Run FinBERT on Kaggle Dataset 1 Title column
           → polarity_market column
- [ ] 1.18 Save → data/finbert_outputs/kaggle1_polarity.csv
- [ ] 1.19 Run FinBERT on Kaggle Dataset 2 Headline column
           → polarity_market column
- [ ] 1.20 Save → data/finbert_outputs/kaggle2_polarity.csv
- [ ] 1.21 Download all 3 output CSVs from Kaggle to local data/finbert_outputs/
- [ ] 1.22 Push finbert outputs to GitHub

### Expected Kaggle runtime: 70–100 minutes total
### Save each output CSV immediately after each dataset — do NOT wait

---

## DAY 2 — Data Merging + Feature Engineering

### Local PC Tasks (notebooks/01_data_prep.ipynb)
- [ ] 2.1  Load kotekar_sentiment.csv (date, company, polarity_company, subjectivity)
- [ ] 2.2  Load kaggle1_polarity.csv and kaggle2_polarity.csv
- [ ] 2.3  Load nifty50_ohlcv.csv
- [ ] 2.4  Aggregate Kotekar to daily: mean polarity_company + subjectivity per date
- [ ] 2.5  Aggregate Kaggle market data to daily: mean polarity_market per date
- [ ] 2.6  Merge all sources on trading date (left join from price data)
- [ ] 2.7  Verify gap period May–Dec 2021 gets polarity_market = 0
- [ ] 2.8  Fill remaining missing: polarity_company=0, subjectivity=0.5
- [ ] 2.9  Save → data/processed/merged_data.csv

### Local PC Tasks (notebooks/02_feature_engineering.ipynb)
- [ ] 2.10 Apply Coif3 wavelet denoising to all OHLCV columns (pywt, level=1, soft)
- [ ] 2.11 Compute RSI(14) on denoised Close
- [ ] 2.12 Compute MACD histogram on denoised Close (EMA12 - EMA26)
- [ ] 2.13 Compute BB_width on denoised Close (20-day)
- [ ] 2.14 Compute ROC(5) on denoised Close
- [ ] 2.15 Drop first 26 rows (MACD warmup period)
- [ ] 2.16 Verify 0 missing values across all 9 feature columns
- [ ] 2.17 Save → data/processed/final_dataset.csv
- [ ] 2.18 Apply 70/15/15 temporal split (no shuffling)
- [ ] 2.19 Verify split order: train dates < val dates < test dates
- [ ] 2.20 Fit MinMaxScaler on train only, transform val and test
- [ ] 2.21 Save scaler → data/processed/scaler.pkl
- [ ] 2.22 Build sliding windows [5×9] for train, val, test
- [ ] 2.23 Check class imbalance — apply class weights if ratio > 60:40
- [ ] 2.24 Save class_weights.json → data/processed/class_weights.json
- [ ] 2.25 Save all .npy arrays → data/processed/
- [ ] 2.26 Push all processed data to GitHub

---

## DAY 3 — Model Implementation

### Local PC Tasks (write all src/ code in VS Code)
- [ ] 3.1  Write src/data/loader.py
           → load_price_data(), load_kotekar_sentiment(),
             load_market_sentiment(), merge_sources()
- [ ] 3.2  Write src/data/preprocessor.py
           → coif3_denoise(), apply_scaler(), handle_missing_values()
- [ ] 3.3  Write src/data/feature_engineering.py
           → compute_rsi(), compute_macd(), compute_bb_width(),
             compute_roc(), create_windows(), generate_targets()
- [ ] 3.4  Write src/sentiment/finbert_inference.py
           → get_finbert_polarity(texts, batch_size, max_length)
           → prepare_kotekar_text(row) — headline + first 2 sentences
- [ ] 3.5  Write src/sentiment/aggregator.py
           → aggregate_daily_polarity(df), fill_gap_period(df)
- [ ] 3.6  Write src/models/encoder.py
           → build_lstm_branch(), build_gru_branch(), build_tcn_branch()
- [ ] 3.7  Write src/models/heads.py
           → build_regression_head(), build_classification_head()
- [ ] 3.8  Write src/models/losses.py
           → uncertainty_weighted_loss(mse, bce, log_sigma1, log_sigma2)
- [ ] 3.9  Write src/models/wsmte.py
           → build_wsmte(config, use_pso=False) — concat version
           → build_wsmte(config, use_pso=True) — PSO version
- [ ] 3.10 Write src/models/pso_weighting.py
           → run_pso_stage(model, X_val, y_val, config)
           → finetune_with_pso_weights(model, weights, data, config)
- [ ] 3.11 Write src/training/trainer.py
           → train_single_run(model, data, config, seed)
           → train_multi_run(config, ablation_cfg, config_name, data, n_runs)
- [ ] 3.12 Write src/training/callbacks.py
           → get_callbacks(config) → [EarlyStopping, ReduceLROnPlateau,
             ModelCheckpoint]
- [ ] 3.13 Write src/evaluation/metrics.py
           → compute_classification_metrics(), compute_regression_metrics(),
             compute_sharpe_ratio()
- [ ] 3.14 Write src/evaluation/shap_analysis.py
           → run_shap_analysis(model, X_test, feature_names)
- [ ] 3.15 Write src/evaluation/granger_test.py
           → run_granger_tests(returns, polarity_company,
             polarity_market, max_lag=5)
- [ ] 3.16 Write src/evaluation/trading_sim.py
           → run_trading_simulation(y_pred_proba, actual_returns,
             risk_free_rate=0.06)
- [ ] 3.17 Write ablation/run_ablation.py
           → loops all 8 configs, saves one row per run to ablation_results.csv
- [ ] 3.18 Write all unit tests (see TESTS_SPEC.md)
- [ ] 3.19 Run tests locally: python -m pytest tests/ -v
           → all tests must pass before Day 4
- [ ] 3.20 Push all src/ code to GitHub

---

## DAY 4 — Training Configs A–G (Kaggle)

### Kaggle GPU Tasks (notebooks/04_model_training.ipynb)
- [ ] 4.1  Pull latest GitHub repo in Kaggle (re-clone)
- [ ] 4.2  Upload processed .npy files as Kaggle dataset: wsmte-processed-data
- [ ] 4.3  Load all data arrays
- [ ] 4.4  Run Config A — 10 seeds — save results to ablation_results_partial.csv
- [ ] 4.5  Run Config B — 10 seeds — save results
- [ ] 4.6  Run Config C — 10 seeds — save results
- [ ] 4.7  Run Config D — 10 seeds — save results
- [ ] 4.8  Run Config E — 10 seeds — save results
- [ ] 4.9  Run Config F — 10 seeds — save results
- [ ] 4.10 Run Config G — 10 seeds — save results
- [ ] 4.11 Save Config G best model weights → config_g_best.h5
- [ ] 4.12 Download ablation_results_AG.csv → local ablation/
- [ ] 4.13 Download config_g_best.h5 → local results/saved_models/
- [ ] 4.14 Push to GitHub

### Local PC Tasks (write paper while training runs)
- [ ] 4.15 Start paper: Introduction + Related Work outline
- [ ] 4.16 Start paper: Methodology section

### Expected Kaggle runtime: 2–4 hours

---

## DAY 5 — Config H PSO Training + Evaluation

### Kaggle GPU Tasks (notebooks/05_ablation.ipynb)
- [ ] 5.1  Pull latest GitHub repo in Kaggle
- [ ] 5.2  Upload config_g_best.h5 as Kaggle dataset: wsmte-models
- [ ] 5.3  Load Config G weights
- [ ] 5.4  Run PSO Stage 2 — search for best [w1, w2, w3]
- [ ] 5.5  Print and record best PSO weights
- [ ] 5.6  Run Config H fine-tuning — 30 seeds
- [ ] 5.7  Save Config H best weights → config_h_best.h5
- [ ] 5.8  Download ablation_results_H.csv → local ablation/
- [ ] 5.9  Download config_h_best.h5 → local results/saved_models/
- [ ] 5.10 Run SHAP analysis on Config H best model (GPU accelerated)
- [ ] 5.11 Download shap_summary.png → local results/figures/
- [ ] 5.12 Push to GitHub

### Local PC Tasks (notebooks/06_evaluation.ipynb)
- [ ] 5.13 Merge ablation_results_AG.csv + ablation_results_H.csv
           → ablation/ablation_results.csv
- [ ] 5.14 Compute mean + max + std per config
- [ ] 5.15 Run Wilcoxon signed-rank test (Config B vs Config H)
- [ ] 5.16 Generate ablation comparison table → results/tables/ablation_summary.csv
- [ ] 5.17 Generate confusion matrix → results/figures/confusion_matrix.png
- [ ] 5.18 Generate AUC-ROC curve → results/figures/auc_roc_curve.png
- [ ] 5.19 Generate loss curves → results/figures/loss_curves.png
- [ ] 5.20 Run Granger causality tests (company + market polarity, lags 1–5)
           → results/tables/granger_results.csv
- [ ] 5.21 Run trading simulation (long-only, Kotekar Algorithm 1)
- [ ] 5.22 Compute Sharpe ratio vs Buy & Hold
           → results/tables/trading_results.csv
- [ ] 5.23 Generate trading simulation plot
           → results/figures/trading_simulation.png
- [ ] 5.24 Optional: wavelet denoising visualization
           → results/figures/wavelet_denoising.png
- [ ] 5.25 Push all results to GitHub

### Expected Kaggle runtime: 1–2 hours

---

## DAY 6 — GitHub Cleanup + Paper Writing

### Local PC Tasks
- [ ] 6.1  Verify all unit tests pass: python -m pytest tests/ -v
- [ ] 6.2  Write README.md (project overview, setup, results summary)
- [ ] 6.3  Verify folder structure matches FOLDER_STRUCTURE.md exactly
- [ ] 6.4  Verify .gitignore excludes raw data, .h5 files, __pycache__
- [ ] 6.5  Final GitHub push — complete, clean repo
- [ ] 6.6  Write paper: Results section
- [ ] 6.7  Write paper: Discussion section
- [ ] 6.8  Write paper: Abstract + Conclusion
- [ ] 6.9  Compile related work citations (~10–15)

---

## DAY 7 — Buffer

- [ ] 7.1  Fix any bugs from evaluation
- [ ] 7.2  Re-run any failed configs
- [ ] 7.3  Finalize paper
- [ ] 7.4  Final GitHub push
- [ ] 7.5  Review all 10 context files still match final implementation
