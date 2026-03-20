"""
src/models/pso_weighting.py
PSO two-stage branch weight optimization for Config F (final proposed model).

Stage 2: Freeze encoder, PSO searches for best [w1, w2, w3] on validation set.
Stage 3: Build new PSO-merge model, transfer branch weights, fine-tune.

Reference: DECISIONS.md section 6, ARCHITECTURE.md PSO two-stage process.
"""

import numpy as np
import tensorflow as tf
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — PSO weight search
# ─────────────────────────────────────────────────────────────────────────────

def _extract_branch_outputs(model, X):
    """
    Extract LSTM, TCN, GRU branch outputs from a trained concat model.

    Returns:
        (out_lstm, out_tcn, out_gru) each np.ndarray of shape [n, 64]
    """
    branch_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer('lstm_branch').output,
            model.get_layer('tcn_branch').output,
            model.get_layer('gru_branch').output,
        ],
        name='branch_extractor',
    )
    out_lstm, out_tcn, out_gru = branch_model.predict(X, verbose=0)
    return out_lstm, out_tcn, out_gru


def _build_downstream_model(model, config):
    """
    Build a frozen sub-model: [batch, 192] → classification output.
    Input is the concatenation of scaled LSTM + TCN + GRU outputs (3 × 64 = 192).
    This matches the pretrained shared_dense weight shape from the concat model.
    """
    concat_dim = (
        config['lstm_units'] + config['tcn_filters'] + config['gru_units']
    )  # 64 + 64 + 64 = 192
    shared_input = tf.keras.Input(shape=(concat_dim,), name='downstream_input')
    x = model.get_layer('shared_dense')(shared_input)
    x = model.get_layer('shared_dropout')(x, training=False)
    x = model.get_layer('clf_dense')(x)
    clf_out = model.get_layer('clf_output')(x)

    downstream = tf.keras.Model(
        inputs=shared_input, outputs=clf_out, name='downstream'
    )
    downstream.trainable = False
    return downstream


def run_pso_stage(model, X_val, y_clf_val, config):
    """
    Stage 2: Freeze all model weights, run PSO to find best [w1, w2, w3].

    Args:
        model:     trained concat WSMTE model (WSMTEModel, concat merge)
        X_val:     np.ndarray [n_val, 5, n_features]
        y_clf_val: np.ndarray [n_val,] int32
        config:    CONFIG dict

    Returns:
        best_weights: np.ndarray [3,] softmax-normalised (sum=1)
        best_cost:    float (negative accuracy at best weights)
    """
    print("Stage 2: Extracting branch outputs...")
    out_lstm, out_tcn, out_gru = _extract_branch_outputs(model, X_val)
    downstream = _build_downstream_model(model, config)

    y_true = y_clf_val.astype(np.float32)

    def fitness(particles):
        """
        particles: np.ndarray [n_particles, 3]
        Returns: cost array [n_particles,] — negative accuracy (PSO minimises)

        Each branch output is scaled by its PSO weight, then the three scaled
        outputs are concatenated → [n, 192].  This preserves the 192-dim input
        that the pretrained shared_dense layer expects (built after Concatenate).
        """
        costs = []
        for particle in particles:
            w = _softmax(particle)
            # Scale each branch and concatenate to match shared_dense input dim
            merged = np.concatenate([
                w[0] * out_lstm,
                w[1] * out_tcn,
                w[2] * out_gru,
            ], axis=1)  # [n_val, 192]
            clf_pred = downstream.predict(merged, verbose=0).flatten()
            acc = np.mean((clf_pred > 0.5).astype(float) == y_true)
            costs.append(-acc)  # minimise negative accuracy
        return np.array(costs)

    options = {
        'c1': config['pso_c1'],
        'c2': config['pso_c2'],
        'w':  config['pso_w'],
    }
    bounds = (
        np.full(3, -5.0),   # lower bounds for raw logits
        np.full(3,  5.0),   # upper bounds
    )

    optimizer = GlobalBestPSO(
        n_particles=config['pso_n_particles'],
        dimensions=3,
        options=options,
        bounds=bounds,
    )

    print(f"Stage 2: Running PSO ({config['pso_n_particles']} particles, "
          f"{config['pso_iterations']} iterations)...")
    best_cost, best_pos = optimizer.optimize(
        fitness,
        iters=config['pso_iterations'],
        verbose=True,
    )

    best_weights = _softmax(best_pos)
    print(f"Stage 2 done. Best weights: w1(LSTM)={best_weights[0]:.4f}, "
          f"w2(TCN)={best_weights[1]:.4f}, w3(GRU)={best_weights[2]:.4f}")
    print(f"Best val accuracy: {-best_cost:.4f}")
    return best_weights, best_cost


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Fine-tune with PSO weights
# ─────────────────────────────────────────────────────────────────────────────

def finetune_with_pso_weights(trained_model, pso_weights, data, config,
                              callbacks=None):
    """
    Stage 3: Build PSO-merge model, transfer encoder weights, fine-tune.

    Architecture change: merge goes from concat [batch,192] → weighted sum [batch,64].
    Only the LSTM, TCN, GRU branch weights are transferred; shared dense and heads
    are re-initialised (different input dim).

    Args:
        trained_model: Stage 1 concat WSMTE model
        pso_weights:   np.ndarray [3,] softmax-normalised weights [w1, w2, w3]
        data:          dict with X_train, y_clf_train, y_reg_train, X_val, y_clf_val, y_reg_val
        config:        CONFIG dict
        callbacks:     list of Keras callbacks (if None, uses fine-tune defaults)

    Returns:
        pso_model: fine-tuned WSMTEModel with PSO merge
    """
    from src.models.wsmte import build_wsmte_pso
    from src.training.callbacks import get_finetune_callbacks

    # Build PSO model
    pso_model = build_wsmte_pso(config, pso_weights)

    # Transfer branch encoder weights
    for layer_name in ['lstm_branch', 'tcn_branch', 'gru_branch']:
        try:
            src_layer = trained_model.get_layer(layer_name)
            dst_layer = pso_model.get_layer(layer_name)
            dst_layer.set_weights(src_layer.get_weights())
        except Exception as e:
            print(f"Warning: could not transfer {layer_name} weights: {e}")

    # Compile
    pso_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    )

    y_train = [data['y_reg_train'], data['y_clf_train']]
    y_val   = [data['y_reg_val'],   data['y_clf_val']]

    cb = callbacks or get_finetune_callbacks(config)

    print(f"Stage 3: Fine-tuning with PSO weights for up to "
          f"{config['pso_finetune_epochs']} epochs...")
    pso_model.fit(
        data['X_train'], y_train,
        validation_data=(data['X_val'], y_val),
        epochs=config['pso_finetune_epochs'],
        batch_size=config['batch_size'],
        callbacks=cb,
        verbose=1,
    )
    return pso_model


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x):
    """Numerically stable softmax for a 1-D array."""
    e = np.exp(x - np.max(x))
    return e / e.sum()
