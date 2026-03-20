# config/config.py
# WSMTE Project — Single Source of Truth for All Hyperparameters
# DO NOT hardcode any of these values in src/ files
# Import everywhere: from config.config import CONFIG

CONFIG = {

    # ─────────────────────────────────────────────
    # PATHS
    # ─────────────────────────────────────────────
    'raw_data_dir':        'data/raw/',
    'processed_data_dir':  'data/processed/',
    'finbert_output_dir':  'data/finbert_outputs/',
    'results_dir':         'results/',
    'figures_dir':         'results/figures/',
    'tables_dir':          'results/tables/',
    'models_dir':          'results/saved_models/',
    'logs_dir':            'logs/training_logs/',
    'ablation_results':    'ablation/ablation_results.csv',

    # ─────────────────────────────────────────────
    # DATA
    # ─────────────────────────────────────────────
    'ticker':              '^NSEI',
    'start_date':          '2020-01-01',
    'end_date':            '2024-05-29',
    'gap_start':           '2021-04-16',  # day after Kaggle1 last entry (Apr 15, 2021)
    'gap_end':             '2021-12-31',

    # ─────────────────────────────────────────────
    # RAW DATASET COLUMN NAMES (verified)
    # ─────────────────────────────────────────────
    # Kotekar dataset
    'kotekar_date_col':       'datePublished',
    'kotekar_text_col':       'articleBody',
    'kotekar_headline_col':   'headline',

    # Kaggle Dataset 1 (Jan 2017 – Apr 15, 2021; filtered from Jan 2020)
    'kaggle1_date_col':       'Date',
    'kaggle1_text_col':       'Title',

    # Kaggle Dataset 2 (Jan 2022 – Dec 2024; filtered to Apr 2024)
    'kaggle2_date_col':       'Date',
    'kaggle2_text_col':       'Headline',

    # ─────────────────────────────────────────────
    # FINBERT OUTPUT FILES
    # ─────────────────────────────────────────────
    'kotekar_sentiment_file': 'data/finbert_outputs/kotekar_sentiment.csv',
    'kaggle1_polarity_file':  'data/finbert_outputs/kaggle1_polarity.csv',
    'kaggle2_polarity_file':  'data/finbert_outputs/kaggle2_polarity.csv',

    # ─────────────────────────────────────────────
    # FEATURES
    # ─────────────────────────────────────────────
    'feature_columns': [
        'Close_d',
        'High_d',
        'Low_d',
        'Open_d',
        'Volume_d',
        'RSI_d',
        'MACD_d',
        'BB_width_d',
        'ROC_d',
        'polarity_company',
        'polarity_company_max',
        'polarity_market',
        'polarity_market_max',
        'subjectivity'
    ],
    'n_features':          14,
    'window_size':         5,
    'warmup_days':         26,

    # ─────────────────────────────────────────────
    # TECHNICAL INDICATOR PARAMETERS
    # ─────────────────────────────────────────────
    'rsi_period':          14,
    'ema_fast':            12,
    'ema_slow':            26,
    'bb_period':           20,
    'bb_std':              2,
    'roc_period':          5,

    # ─────────────────────────────────────────────
    # WAVELET DENOISING
    # ─────────────────────────────────────────────
    'wavelet':             'coif3',
    'wavelet_level':       1,
    'wavelet_mode':        'soft',

    # ─────────────────────────────────────────────
    # SENTIMENT MODELS
    # ─────────────────────────────────────────────
    'finbert_model':          'ProsusAI/finbert',
    'mdeberta_model':         'cross-encoder/nli-deberta-v3-small',
    'finbert_batch_size':     32,
    'mdeberta_batch_size':    16,
    'finbert_max_length':     512,
    'mdeberta_max_length':    512,

    # Kotekar text preparation for FinBERT
    # Uses headline + first 2 sentences of articleBody
    'kotekar_finbert_sentences': 2,

    # mDeBERTa uses full articleBody (truncated at 512 tokens)
    # Subjectivity hypothesis
    'subjectivity_hypothesis': (
        "This text expresses a personal opinion or subjective view."
    ),

    # Fill values for missing sentiment
    'missing_polarity':       0.0,
    'missing_subjectivity':   0.5,

    # ─────────────────────────────────────────────
    # DATA SPLIT
    # ─────────────────────────────────────────────
    'train_ratio':         0.75,
    'val_ratio':           0.10,
    'test_ratio':          0.15,
    'shuffle':             False,

    # ─────────────────────────────────────────────
    # CLASS IMBALANCE
    # ─────────────────────────────────────────────
    'imbalance_threshold': 1.5,

    # ─────────────────────────────────────────────
    # MODEL ARCHITECTURE
    # ─────────────────────────────────────────────
    'lstm_units':             32,
    'lstm_dropout':           0.2,
    'lstm_recurrent_dropout': 0.0,

    'gru_units':              32,
    'gru_dropout':            0.2,
    'gru_recurrent_dropout':  0.0,

    'tcn_filters':            32,
    'tcn_kernel_size':        2,
    'tcn_dilations':          [1, 2, 4],
    'tcn_padding':            'causal',
    'tcn_activation':         'relu',
    'tcn_dropout':            0.2,
    'tcn_use_skip_connections': True,
    'tcn_use_batch_norm':     False,
    'tcn_use_layer_norm':     False,
    'tcn_use_weight_norm':    True,

    'shared_dense_units':     32,
    'shared_dense_activation': 'relu',
    'shared_dense_dropout':   0.2,

    'head_dense_units':       16,
    'head_dense_activation':  'relu',

    # ─────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────
    'optimizer':              'adam',
    'learning_rate':          0.001,
    'batch_size':             32,
    'max_epochs':             100,

    'early_stopping_patience': 15,
    'early_stopping_monitor':  'val_loss',
    'restore_best_weights':    True,

    'lr_reduce_factor':       0.5,
    'lr_reduce_patience':     7,
    'lr_min':                 1e-6,

    'n_runs_final':           30,
    'n_runs_ablation':        30,

    'SEEDS': [23, 47, 0, 1, 2, 7, 13, 17, 21, 29,
              31, 37, 42, 53, 61, 67, 71, 79, 83, 89,
              97, 101, 113, 127, 131, 137, 149, 151, 157, 163],

    # ─────────────────────────────────────────────
    # PSO (Config H only)
    # ─────────────────────────────────────────────
    'pso_n_particles':        20,
    'pso_iterations':         50,
    'pso_c1':                 0.5,
    'pso_c2':                 0.3,
    'pso_w':                  0.9,
    'pso_finetune_epochs':    20,
    'pso_finetune_patience':  5,

    # ─────────────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────────────
    'granger_max_lag':        10,
    'risk_free_rate':         0.06,
    'trading_strategy':       'long_only',

    # ─────────────────────────────────────────────
    # ABLATION CONFIGS
    # ─────────────────────────────────────────────
    'ablation_configs': {
        'A': {
            'features':    ['Close_d', 'High_d', 'Low_d', 'Open_d', 'Volume_d'],
            'heads':       ['classification'],
            'merge':       'concat',
            'use_pso':     False,
            'n_runs':      30,
            'description': 'Raw OHLCV denoised only — price-only floor baseline'
        },
        'B': {
            'features':    ['Close_d', 'High_d', 'Low_d', 'Open_d', 'Volume_d',
                            'polarity_company', 'polarity_market', 'polarity_market_max',
                            'subjectivity'],
            'heads':       ['classification'],
            'merge':       'concat',
            'use_pso':     False,
            'n_runs':      30,
            'description': 'Denoised OHLCV + all sentiment — sentiment on raw prices'
        },
        'C': {
            'features':    ['Close_d', 'Volume_d',
                            'RSI_d', 'MACD_d', 'BB_width_d', 'ROC_d'],
            'heads':       ['classification'],
            'merge':       'concat',
            'use_pso':     False,
            'n_runs':      30,
            'description': 'Technical indicators only — meaningful floor baseline'
        },
        'D': {
            'features':    ['Close_d', 'High_d', 'Low_d', 'Open_d', 'Volume_d',
                            'RSI_d', 'MACD_d', 'BB_width_d', 'ROC_d',
                            'polarity_company', 'polarity_company_max',
                            'polarity_market', 'polarity_market_max',
                            'subjectivity'],
            'heads':       ['classification'],
            'merge':       'concat',
            'use_pso':     False,
            'n_runs':      30,
            'description': 'All 14 features, single-task — full feature set confirmed'
        },
        'E': {
            'features':    ['Close_d', 'High_d', 'Low_d', 'Open_d', 'Volume_d',
                            'RSI_d', 'MACD_d', 'BB_width_d', 'ROC_d',
                            'polarity_company', 'polarity_company_max',
                            'polarity_market', 'polarity_market_max',
                            'subjectivity'],
            'heads':       ['classification', 'regression'],
            'merge':       'concat',
            'use_pso':     False,
            'n_runs':      30,
            'description': 'All 14 features, both heads, concat — full WSMTE without PSO'
        },
        'F': {
            'features':    ['Close_d', 'High_d', 'Low_d', 'Open_d', 'Volume_d',
                            'RSI_d', 'MACD_d', 'BB_width_d', 'ROC_d',
                            'polarity_company', 'polarity_company_max',
                            'polarity_market', 'polarity_market_max',
                            'subjectivity'],
            'heads':       ['classification', 'regression'],
            'merge':       'pso',
            'use_pso':     True,
            'n_runs':      30,
            'description': 'All 14 features, both heads, PSO — FINAL PROPOSED MODEL'
        },
    },

    # ─────────────────────────────────────────────
    # RESULTS COLUMNS
    # ─────────────────────────────────────────────
    'results_columns': [
        'config', 'seed', 'run',
        'accuracy', 'balanced_accuracy', 'auc',
        'precision', 'recall', 'f1',
        'rmse', 'mae', 'r2'
    ],
}
