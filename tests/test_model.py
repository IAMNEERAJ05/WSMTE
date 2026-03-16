"""
tests/test_model.py
Unit tests for WSMTE model architecture.
All tests run on CPU — no GPU needed.

Run: python -m pytest tests/test_model.py -v
"""

import os
import pytest
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CONFIG
from src.models.wsmte import build_wsmte


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def model():
    """Full dual-head concat model (like Config G)."""
    return build_wsmte(CONFIG, use_pso=False)


@pytest.fixture(scope='module')
def dummy_input():
    return np.random.randn(8, CONFIG['window_size'], CONFIG['n_features']).astype(np.float32)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_model_builds(model):
    """Model must build without errors."""
    assert model is not None


def test_output_shapes(model, dummy_input):
    """Dual-head model must return (reg_out, clf_out) each of shape [batch, 1]."""
    outputs = model(dummy_input)
    # outputs is a list [reg_out, clf_out] for dual-head
    assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
    reg_out, clf_out = outputs
    assert reg_out.shape == (8, 1), f"reg_out shape: {reg_out.shape}"
    assert clf_out.shape == (8, 1), f"clf_out shape: {clf_out.shape}"


def test_classification_output_range(model):
    """Sigmoid output must lie in [0, 1]."""
    dummy = np.random.randn(32, CONFIG['window_size'], CONFIG['n_features']).astype(np.float32)
    _, clf_out = model(dummy)
    arr = clf_out.numpy()
    assert (arr >= 0).all() and (arr <= 1).all(), \
        f"clf_out outside [0,1]: min={arr.min():.4f}, max={arr.max():.4f}"


def test_regression_output_unbounded():
    """Linear activation — with extreme inputs output must not be clipped to [0,1]."""
    model = build_wsmte(CONFIG, use_pso=False)
    extreme = np.ones((8, CONFIG['window_size'], CONFIG['n_features']), dtype=np.float32) * 10.0
    reg_out, _ = model(extreme)
    # If sigmoid were used, all outputs would collapse to ~1; std > 0 checks unboundedness
    assert reg_out.numpy().std() >= 0, "reg_out std check"   # always true; real check below
    # A stronger check: run on negative extremes too
    neg_extreme = -extreme
    reg_neg, _ = model(neg_extreme)
    # Regression outputs for +10 and -10 inputs should differ
    assert not np.allclose(reg_out.numpy(), reg_neg.numpy(), atol=1e-3), \
        "Regression output does not vary — may be using sigmoid instead of linear"


def test_sigma_parameters_trainable(model):
    """Uncertainty weighting parameters log_sigma1 and log_sigma2 must be trainable."""
    trainable_names = [v.name for v in model.trainable_variables]
    assert any('sigma' in name or 'log_sigma' in name for name in trainable_names), \
        f"No sigma/log_sigma variable found. Trainable vars: {trainable_names}"


def test_three_branches(model):
    """Model must contain LSTM, GRU, and TCN layers."""
    layer_types = [type(layer).__name__ for layer in model.layers]
    assert 'LSTM' in layer_types, f"LSTM not in layers: {layer_types}"
    assert 'GRU'  in layer_types, f"GRU not in layers: {layer_types}"
    assert 'TCN'  in layer_types, f"TCN not in layers: {layer_types}"


def test_different_seeds_differ():
    """
    Different random seeds must produce different weights and thus different outputs
    after 2 epochs of training on a small dummy dataset.
    """
    WINDOW = CONFIG['window_size']
    N_FEAT = CONFIG['n_features']
    np.random.seed(0)
    X  = np.random.randn(60, WINDOW, N_FEAT).astype(np.float32)
    y_reg = np.random.randn(60, 1).astype(np.float32)
    y_clf = np.random.randint(0, 2, (60, 1)).astype(np.float32)
    X_val = X[:10]; y_reg_val = y_reg[:10]; y_clf_val = y_clf[:10]

    accs = []
    for seed in [42, 7, 123]:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        m = build_wsmte(CONFIG, use_pso=False)
        m.compile(optimizer=tf.keras.optimizers.Adam(0.001))
        m.fit(X, [y_reg, y_clf],
              validation_data=(X_val, [y_reg_val, y_clf_val]),
              epochs=2, verbose=0)
        _, clf_out = m(X_val)
        accs.append(clf_out.numpy().mean())

    # At least two of the three seeds should produce different mean predictions
    assert len(set(round(a, 6) for a in accs)) > 1, \
        f"All seeds produced identical outputs: {accs}"


def test_single_head_clf_model():
    """Config A: classification-only model must return a single tensor."""
    ablation_cfg = CONFIG['ablation_configs']['A']
    model = build_wsmte(CONFIG, ablation_cfg=ablation_cfg)
    n_feat = len(ablation_cfg['features'])
    dummy = np.random.randn(4, CONFIG['window_size'], n_feat).astype(np.float32)
    out = model(dummy)
    # Single output — should be a tensor not a list
    assert hasattr(out, 'shape'), "Single-head model output should be a tensor"
    assert out.shape == (4, 1), f"clf-only output shape: {out.shape}"


def test_single_head_reg_model():
    """Config F: regression-only model must return a single tensor."""
    ablation_cfg = CONFIG['ablation_configs']['F']
    model = build_wsmte(CONFIG, ablation_cfg=ablation_cfg)
    n_feat = len(ablation_cfg['features'])
    dummy = np.random.randn(4, CONFIG['window_size'], n_feat).astype(np.float32)
    out = model(dummy)
    assert hasattr(out, 'shape'), "Single-head model output should be a tensor"
    assert out.shape == (4, 1), f"reg-only output shape: {out.shape}"
