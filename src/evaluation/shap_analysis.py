"""
src/evaluation/shap_analysis.py
SHAP feature importance analysis for the WSMTE shared encoder.
Uses shap.GradientExplainer (GPU-compatible, works with TF models).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import shap


def run_shap_analysis(model, X_test, feature_names, save_path=None,
                      background_size=100, max_display=9):
    """
    Compute SHAP values for the classification head using GradientExplainer,
    generate a summary plot, and optionally save it.

    The explainer uses the shared encoder + classification path.
    A background dataset (random sample of training data) is used as the
    reference distribution.

    Args:
        model:           compiled WSMTEModel (dual-head or clf-only)
        X_test:          np.ndarray [n_test, window_size, n_features]
        feature_names:   list of strings, length n_features
                         (e.g. CONFIG['feature_columns'])
        save_path:       optional path to save the PNG figure
                         (e.g. 'results/figures/shap_summary.png')
        background_size: int — number of background samples for GradientExplainer
        max_display:     int — max features to show in summary plot

    Returns:
        shap_values: np.ndarray — raw SHAP values from explainer
    """
    # Build a sub-model that outputs only the classification logit
    # (GradientExplainer requires a single output)
    import tensorflow as tf

    clf_output = None
    for layer in model.layers:
        if layer.name == 'clf_output':
            clf_output = layer.output
            break

    if clf_output is None:
        raise ValueError("clf_output layer not found. "
                         "Run SHAP on a model that has a classification head.")

    clf_model = tf.keras.Model(inputs=model.input, outputs=clf_output)

    # Background dataset — small random subset of X_test
    np.random.seed(42)
    idx = np.random.choice(len(X_test), size=min(background_size, len(X_test)),
                           replace=False)
    background = X_test[idx]

    print("Running SHAP GradientExplainer...")
    explainer   = shap.GradientExplainer(clf_model, background)
    shap_values = explainer.shap_values(X_test)

    # shap_values shape: [n_test, window_size, n_features]
    # Average over the window dimension for a per-feature importance plot
    if isinstance(shap_values, list):
        sv = shap_values[0]   # single output
    else:
        sv = shap_values
    sv_mean = sv.mean(axis=1)     # [n_test, n_features]

    # SHAP summary plot
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        sv_mean,
        features=X_test.mean(axis=1),   # mean feature values over window
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP summary saved → {save_path}")

    plt.show()
    return shap_values
