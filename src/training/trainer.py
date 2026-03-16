"""
src/training/trainer.py
Single-run and multi-run training loops for WSMTE ablation study.
"""

import json
import os
import numpy as np
import tensorflow as tf
import pandas as pd

from src.models.wsmte import build_wsmte
from src.training.callbacks import get_callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def train_single_run(ablation_cfg, data, config, seed, run_id='run'):
    """
    Train one WSMTE model with a fixed random seed and return evaluation results.

    Args:
        ablation_cfg: dict from CONFIG['ablation_configs']['X']
        data:         dict with keys:
                        X_train, X_val, X_test,
                        y_clf_train, y_clf_val, y_clf_test,
                        y_reg_train, y_reg_val, y_reg_test,
                        class_weight (dict or None)
        config:       global CONFIG dict
        seed:         int — random seed for reproducibility
        run_id:       string used in checkpoint filename

    Returns:
        dict with keys from CONFIG['results_columns'] (metrics on test set)
        and 'history' (Keras History object)
    """
    # Set seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)

    heads    = ablation_cfg['heads']
    features = ablation_cfg['features']
    has_both = 'classification' in heads and 'regression' in heads
    has_clf  = 'classification' in heads
    has_reg  = 'regression' in heads

    # Slice feature columns (for configs A, B that use fewer features)
    feat_idx = [config['feature_columns'].index(f) for f in features]
    X_train = data['X_train'][:, :, feat_idx]
    X_val   = data['X_val'][:, :, feat_idx]
    X_test  = data['X_test'][:, :, feat_idx]

    # Build and compile model
    model = build_wsmte(config, ablation_cfg=ablation_cfg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    )

    # Apply class weights for classification tasks
    class_weight = data.get('class_weight')
    if has_clf and class_weight:
        model.set_class_weight(class_weight)

    # Prepare targets
    if has_both:
        y_train = [data['y_reg_train'], data['y_clf_train']]
        y_val   = [data['y_reg_val'],   data['y_clf_val']]
        y_test  = [data['y_reg_test'],  data['y_clf_test']]
    elif has_clf:
        y_train, y_val, y_test = (
            data['y_clf_train'], data['y_clf_val'], data['y_clf_test']
        )
    else:
        y_train, y_val, y_test = (
            data['y_reg_train'], data['y_reg_val'], data['y_reg_test']
        )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['max_epochs'],
        batch_size=config['batch_size'],
        callbacks=get_callbacks(config, run_id=run_id),
        verbose=0,
    )

    # Evaluate on test set
    results = _evaluate(model, X_test, y_test, data, has_both, has_clf, has_reg)
    results['history'] = history
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-run loop
# ─────────────────────────────────────────────────────────────────────────────

def train_multi_run(ablation_cfg, config_name, data, config, n_runs=None,
                    seeds=None, results_path=None):
    """
    Train n_runs models with different seeds, append results to CSV.

    Args:
        ablation_cfg:  dict from CONFIG['ablation_configs']['X']
        config_name:   string like 'A', 'B', ..., 'H'
        data:          dict (same as train_single_run)
        config:        global CONFIG dict
        n_runs:        int; defaults to ablation_cfg['n_runs']
        seeds:         list of ints; if None, uses range(n_runs)
        results_path:  CSV path; defaults to config['ablation_results']

    Returns:
        pd.DataFrame of all runs for this config
    """
    if n_runs is None:
        n_runs = ablation_cfg['n_runs']
    if seeds is None:
        seeds = list(range(n_runs))
    if results_path is None:
        results_path = config['ablation_results']

    all_rows = []
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"Config {config_name} | Run {run_idx+1}/{n_runs} | seed={seed}")
        print('='*50)

        run_id = f'config_{config_name}_run{run_idx:02d}'
        result = train_single_run(ablation_cfg, data, config, seed=seed, run_id=run_id)

        row = {
            'config': config_name,
            'seed':   seed,
            'run':    run_idx + 1,
            **{k: result[k] for k in config['results_columns']
               if k not in ('config', 'seed', 'run') and k in result},
        }
        all_rows.append(row)

        # Append row to CSV immediately (in case session dies)
        _append_to_csv(row, results_path, config)
        print(f"  accuracy={row.get('accuracy', 'N/A'):.4f}  "
              f"auc={row.get('auc', 'N/A'):.4f}")

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(model, X_test, y_test, data, has_both, has_clf, has_reg):
    """Run model on test set and compute all required metrics."""
    from src.evaluation.metrics import (
        compute_classification_metrics,
        compute_regression_metrics,
    )

    preds = model.predict(X_test, verbose=0)

    result = {k: None for k in [
        'accuracy', 'balanced_accuracy', 'auc', 'precision', 'recall', 'f1',
        'rmse', 'mae', 'r2',
    ]}

    if has_both:
        reg_pred = preds[0].flatten()
        clf_pred = preds[1].flatten()
        y_clf    = data['y_clf_test']
        y_reg    = data['y_reg_test']
    elif has_clf:
        clf_pred = preds.flatten()
        y_clf    = data['y_clf_test']
        y_reg    = None
        reg_pred = None
    else:
        reg_pred = preds.flatten()
        y_reg    = data['y_reg_test']
        clf_pred = None
        y_clf    = None

    if clf_pred is not None:
        clf_metrics = compute_classification_metrics(y_clf, clf_pred)
        result.update(clf_metrics)

    if reg_pred is not None:
        reg_metrics = compute_regression_metrics(y_reg, reg_pred)
        result.update(reg_metrics)

    return result


def _append_to_csv(row, path, config):
    """Append a single result row to the ablation CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_row = pd.DataFrame([row])
    # Ensure correct column order; fill missing with None
    for col in config['results_columns']:
        if col not in df_row.columns:
            df_row[col] = None
    df_row = df_row[config['results_columns']]

    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    df_row.to_csv(path, mode='a', header=write_header, index=False)
