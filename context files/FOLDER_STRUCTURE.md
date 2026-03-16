# WSMTE — Folder Structure Specification

## CRITICAL INSTRUCTION
Create files EXACTLY as specified here.
Every file has a defined responsibility — do not merge responsibilities.
Do not create files not listed here without explicit instruction.

---

## Complete Folder Tree

```
WSMTE/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   └── config.py                      ← ALL hyperparameters, paths, column names
│
├── data/
│   ├── raw/                           ← original downloaded files (gitignored)
│   │   ├── kotekar_news.csv           ← Kotekar GitHub dataset
│   │   │                                 cols: datePublished, company, symbol,
│   │   │                                 headline, description, articleBody,
│   │   │                                 tags, author, url
│   │   ├── kaggle_news_1.csv          ← Kaggle Dataset 1 (Jan2020–Apr2021)
│   │   │                                 cols: Date, Title, URL,
│   │   │                                 sentiment, confidence
│   │   ├── kaggle_news_2.csv          ← Kaggle Dataset 2 (Jan2022–May2024)
│   │   │                                 cols: Archive, Date, Headline,
│   │   │                                 Headline link
│   │   └── nifty50_ohlcv.csv          ← Yahoo Finance ^NSEI
│   │
│   ├── processed/                     ← generated files, pushed to GitHub
│   │   ├── merged_data.csv            ← after merging all sources by date
│   │   ├── final_dataset.csv          ← after denoising + feature engineering
│   │   ├── class_weights.json         ← null or {0: w0, 1: w1}
│   │   ├── scaler.pkl                 ← fitted MinMaxScaler (gitignored)
│   │   ├── X_train.npy                ← shape (~730, 5, 9) (gitignored)
│   │   ├── X_val.npy                  ← shape (~155, 5, 9) (gitignored)
│   │   ├── X_test.npy                 ← shape (~155, 5, 9) (gitignored)
│   │   ├── y_clf_train.npy            ← shape (~730,) int (gitignored)
│   │   ├── y_clf_val.npy
│   │   ├── y_clf_test.npy
│   │   ├── y_reg_train.npy            ← shape (~730,) float (gitignored)
│   │   ├── y_reg_val.npy
│   │   └── y_reg_test.npy
│   │
│   └── finbert_outputs/               ← downloaded from Kaggle, pushed to GitHub
│       ├── kotekar_sentiment.csv      ← cols: date, company, symbol,
│       │                                 polarity_company, subjectivity
│       ├── kaggle1_polarity.csv       ← cols: date, polarity_market
│       └── kaggle2_polarity.csv       ← cols: date, polarity_market
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  ← load_price_data(),
│   │   │                                 load_kotekar_sentiment(),
│   │   │                                 load_market_sentiment(),
│   │   │                                 merge_sources()
│   │   ├── preprocessor.py            ← coif3_denoise(), apply_scaler(),
│   │   │                                 handle_missing_values()
│   │   └── feature_engineering.py    ← compute_rsi(), compute_macd(),
│   │                                     compute_bb_width(), compute_roc(),
│   │                                     create_windows(), generate_targets()
│   │
│   ├── sentiment/
│   │   ├── __init__.py
│   │   ├── finbert_inference.py       ← get_finbert_polarity(texts, batch_size)
│   │   │                                 get_subjectivity(texts, batch_size)
│   │   │                                 prepare_kotekar_text(row)
│   │   │                                   → headline + first 2 sentences
│   │   └── aggregator.py             ← aggregate_daily_polarity(df),
│   │                                     fill_gap_period(df)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py                 ← build_lstm_branch(),
│   │   │                                 build_gru_branch(),
│   │   │                                 build_tcn_branch()
│   │   ├── heads.py                   ← build_regression_head(),
│   │   │                                 build_classification_head()
│   │   ├── losses.py                  ← uncertainty_weighted_loss(
│   │   │                                 mse, bce, log_sigma1, log_sigma2)
│   │   ├── wsmte.py                   ← build_wsmte(config, use_pso=False)
│   │   │                                 returns compiled Keras model
│   │   └── pso_weighting.py           ← run_pso_stage(model, X_val, y_val, config)
│   │                                     finetune_with_pso_weights(model, weights)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                 ← train_single_run(model, data, config, seed)
│   │   │                                 train_multi_run(config, ablation_cfg,
│   │   │                                 config_name, data, n_runs)
│   │   └── callbacks.py              ← get_callbacks(config) returns
│   │                                     [EarlyStopping, ReduceLROnPlateau,
│   │                                     ModelCheckpoint]
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                 ← compute_classification_metrics(),
│       │                                 compute_regression_metrics(),
│       │                                 compute_sharpe_ratio()
│       ├── shap_analysis.py           ← run_shap_analysis(model, X_test,
│       │                                 feature_names, save_path)
│       ├── granger_test.py            ← run_granger_tests(returns,
│       │                                 polarity_company, polarity_market,
│       │                                 max_lag=5)
│       └── trading_sim.py            ← run_trading_simulation(y_pred_proba,
│                                         actual_returns, risk_free_rate=0.06)
│
├── ablation/
│   ├── run_ablation.py                ← main ablation loop, all 8 configs
│   └── ablation_results.csv          ← one row per run
│                                         cols: config, seed, run, accuracy,
│                                         balanced_accuracy, auc, precision,
│                                         recall, f1, rmse, mae, r2
│
├── notebooks/
│   ├── 01_data_prep.ipynb            ← LOCAL: merge all sources
│   ├── 02_feature_engineering.ipynb  ← LOCAL: denoise, technicals, windows
│   ├── 03_finbert_inference.ipynb    ← KAGGLE GPU: FinBERT + mDeBERTa
│   │                                     on Kotekar, Kaggle1, Kaggle2
│   ├── 04_model_training.ipynb       ← KAGGLE GPU: Configs A–G
│   ├── 05_ablation.ipynb             ← KAGGLE GPU: Config H PSO + SHAP
│   └── 06_evaluation.ipynb           ← LOCAL: all metrics and plots
│
├── results/
│   ├── figures/
│   │   ├── loss_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── auc_roc_curve.png
│   │   ├── shap_summary.png
│   │   ├── ablation_comparison.png
│   │   ├── trading_simulation.png
│   │   └── wavelet_denoising.png     ← optional
│   │
│   ├── tables/
│   │   ├── ablation_summary.csv
│   │   ├── granger_results.csv
│   │   └── trading_results.csv
│   │
│   └── saved_models/                  ← gitignored (large .h5 files)
│       ├── config_g_best.h5
│       └── config_h_best.h5
│
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   └── test_model.py
│
└── logs/
    └── training_logs/                 ← gitignored
```

---

## File Responsibilities

| File | Single Responsibility |
|------|-----------------------|
| config/config.py | All hyperparameters + verified column names |
| src/data/loader.py | Load and merge raw data sources |
| src/data/preprocessor.py | Wavelet denoising and scaling |
| src/data/feature_engineering.py | Technical indicators and sliding windows |
| src/sentiment/finbert_inference.py | FinBERT polarity + mDeBERTa subjectivity |
| src/sentiment/aggregator.py | Daily aggregation and gap fill |
| src/models/encoder.py | LSTM, TCN, GRU branch definitions |
| src/models/heads.py | Regression and classification heads |
| src/models/losses.py | Uncertainty weighting loss |
| src/models/wsmte.py | Full model assembly |
| src/models/pso_weighting.py | PSO two-stage optimization |
| src/training/trainer.py | Multi-seed training loop |
| src/training/callbacks.py | Keras callbacks |
| src/evaluation/metrics.py | All metrics computation |
| src/evaluation/shap_analysis.py | SHAP feature importance |
| src/evaluation/granger_test.py | Granger causality |
| src/evaluation/trading_sim.py | Trading simulation + Sharpe |
| ablation/run_ablation.py | Ablation loop over 8 configs |

---

## GitHub Push vs Gitignore

### Push to GitHub
- All src/ .py files
- All notebooks/ .ipynb files
- config/, requirements.txt, README.md, .gitignore
- data/processed/final_dataset.csv
- data/processed/class_weights.json
- data/finbert_outputs/ (all 3 CSV files)
- ablation/ablation_results.csv
- results/figures/ and results/tables/
- tests/

### Gitignore
```
data/raw/
data/processed/*.npy
data/processed/scaler.pkl
results/saved_models/
logs/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.h5
```
