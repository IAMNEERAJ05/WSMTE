"""
ablation/run_ablation.py
Main ablation loop — trains all 8 configs (A–H) and saves results.

Usage (Kaggle GPU):
    python ablation/run_ablation.py              # runs all configs
    python ablation/run_ablation.py --configs A B # runs specific configs

Configs A–G: 10 runs each (simple concatenation)
Config  H:   30 runs (PSO-weighted merge, 3-stage training)
"""

import sys
import os
import argparse
import json
import numpy as np
import tensorflow as tf

# Ensure project root is on path (for Kaggle: sys.path.insert after git clone)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CONFIG
from src.training.trainer import train_single_run, train_multi_run


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(config):
    """Load all processed .npy arrays and class_weights."""
    p = config['processed_data_dir']
    data = {
        'X_train':     np.load(p + 'X_train.npy'),
        'X_val':       np.load(p + 'X_val.npy'),
        'X_test':      np.load(p + 'X_test.npy'),
        'y_clf_train': np.load(p + 'y_clf_train.npy'),
        'y_clf_val':   np.load(p + 'y_clf_val.npy'),
        'y_clf_test':  np.load(p + 'y_clf_test.npy'),
        'y_reg_train': np.load(p + 'y_reg_train.npy'),
        'y_reg_val':   np.load(p + 'y_reg_val.npy'),
        'y_reg_test':  np.load(p + 'y_reg_test.npy'),
    }
    cw_path = p + 'class_weights.json'
    with open(cw_path) as f:
        class_weight = json.load(f)
    # JSON stores keys as strings; convert to int
    if class_weight is not None:
        class_weight = {int(k): v for k, v in class_weight.items()}
    data['class_weight'] = class_weight
    print(f"Data loaded. X_train: {data['X_train'].shape}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Config H — PSO training
# ─────────────────────────────────────────────────────────────────────────────

def run_config_h(data, config, n_runs=30, results_path=None):
    """
    Run Config H: 3-stage PSO training × 30 seeds.

    Stage 1: Train with concatenation (same as Config G).
    Stage 2: PSO search for best [w1, w2, w3] on val set.
    Stage 3: Fine-tune PSO model.

    Saves the overall best model (highest test accuracy across all runs) as
    config_h_best.keras and its PSO weights as config_h_best_pso_weights.npy.
    """
    from src.models.wsmte import build_wsmte
    from src.models.pso_weighting import run_pso_stage, finetune_with_pso_weights
    from src.training.callbacks import get_callbacks
    from src.evaluation.metrics import compute_classification_metrics, compute_regression_metrics

    if results_path is None:
        results_path = config['ablation_results']

    ablation_cfg = config['ablation_configs']['H']
    seeds = list(range(n_runs))

    best_accuracy   = -1.0
    best_model_path = os.path.join(config['models_dir'], 'config_h_best.keras')
    best_weights_path = os.path.join(config['models_dir'], 'config_h_best_pso_weights.npy')

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*55}")
        print(f"Config H | Run {run_idx+1}/{n_runs} | seed={seed}")
        print('='*55)

        tf.random.set_seed(seed)
        np.random.seed(seed)

        # ── Stage 1: train with concat ────────────────────────────────────────
        print("Stage 1: Training with concatenation merge...")
        g_cfg = dict(ablation_cfg)
        g_cfg['merge']   = 'concat'
        g_cfg['use_pso'] = False

        stage1_result = train_single_run(
            g_cfg, data, config, seed=seed,
            run_id=f'config_H_stage1_run{run_idx:02d}',
        )
        stage1_model = _get_model_from_checkpoint(
            config, f'config_H_stage1_run{run_idx:02d}'
        )

        # ── Stage 2: PSO weight search ────────────────────────────────────────
        best_weights, _ = run_pso_stage(
            stage1_model,
            data['X_val'], data['y_clf_val'],
            config,
        )

        # ── Stage 3: fine-tune with PSO weights ───────────────────────────────
        pso_model = finetune_with_pso_weights(
            stage1_model, best_weights, data, config
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        preds = pso_model.predict(data['X_test'], verbose=0)
        reg_pred = preds[0].flatten()
        clf_pred = preds[1].flatten()

        clf_m = compute_classification_metrics(data['y_clf_test'], clf_pred)
        reg_m = compute_regression_metrics(data['y_reg_test'], reg_pred)

        import pandas as pd
        row = {
            'config': 'H', 'seed': seed, 'run': run_idx + 1,
            **clf_m, **reg_m,
        }
        # Reorder to results_columns
        ordered = {k: row.get(k) for k in config['results_columns']}
        df_row = pd.DataFrame([ordered])
        write_header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0
        df_row.to_csv(results_path, mode='a', header=write_header, index=False)
        print(f"  accuracy={clf_m['accuracy']:.4f}  auc={clf_m['auc']:.4f}  "
              f"pso_weights={best_weights.round(4)}")

        # ── Save best model across all runs ───────────────────────────────────
        if clf_m['accuracy'] > best_accuracy:
            best_accuracy = clf_m['accuracy']
            os.makedirs(config['models_dir'], exist_ok=True)
            pso_model.save(best_model_path)
            np.save(best_weights_path, best_weights)
            print(f"  New best Config H — accuracy={best_accuracy:.4f} — saved to {best_model_path}")


def _get_model_from_checkpoint(config, run_id):
    """Reload best model saved by ModelCheckpoint."""
    path = os.path.join(config['models_dir'], f'best_{run_id}.keras')
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    raise FileNotFoundError(f"Checkpoint not found: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WSMTE Ablation Study')
    parser.add_argument(
        '--configs', nargs='+', default=list('ABCDEFGH'),
        help='Which configs to run (default: all A–H)'
    )
    args = parser.parse_args()

    data = load_data(CONFIG)

    for cfg_name in args.configs:
        cfg_name = cfg_name.upper()
        if cfg_name not in CONFIG['ablation_configs']:
            print(f"Unknown config {cfg_name}, skipping.")
            continue

        ablation_cfg = CONFIG['ablation_configs'][cfg_name]
        print(f"\n{'#'*55}")
        print(f"# Starting Config {cfg_name}: {ablation_cfg['description']}")
        print('#'*55)

        if cfg_name == 'H':
            run_config_h(data, CONFIG, n_runs=ablation_cfg['n_runs'])
        else:
            train_multi_run(
                ablation_cfg=ablation_cfg,
                config_name=cfg_name,
                data=data,
                config=CONFIG,
                n_runs=ablation_cfg['n_runs'],
            )

    print("\nAblation study complete.")
    print(f"Results saved to: {CONFIG['ablation_results']}")


if __name__ == '__main__':
    main()
