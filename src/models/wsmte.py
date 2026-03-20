"""
src/models/wsmte.py
Full WSMTE model assembly using Keras Functional API.

build_wsmte(config, use_pso=False, ablation_cfg=None)
  → returns WSMTEModel instance

WSMTEModel subclasses tf.keras.Model with functional inputs/outputs so that:
  - model.layers exposes LSTM, GRU, TCN layers (required by tests)
  - Custom train_step applies fixed-weight MTL loss (0.3×MSE + 0.7×BCE)
  - Custom train_step / test_step compute the correct loss per head config
"""

import tensorflow as tf
import numpy as np

from src.models.encoder import build_lstm_branch, build_gru_branch, build_tcn_branch
from src.models.heads import build_regression_head, build_classification_head
from src.models.losses import fixed_weighted_loss


@tf.keras.utils.register_keras_serializable(package='src.models.wsmte')
class WSMTEModel(tf.keras.Model):

    """
    WSMTE model with fixed-weight MTL loss (0.3 × MSE + 0.7 × BCE).

    For dual-head configs (E, F):
      - Custom train_step / test_step apply fixed weighting

    For single-head configs (A–D):
      - Custom train_step computes task loss directly
      - class_weight support: pass sample_weight via model.set_class_weight()

    Initialised via the functional API (inputs= / outputs= kwargs to super()),
    so model.layers correctly exposes LSTM, GRU, TCN layers.
    """

    def __init__(self, functional_inputs, functional_outputs, heads, **kwargs):
        super().__init__(
            inputs=functional_inputs,
            outputs=functional_outputs,
            **kwargs,
        )
        self.heads = heads
        self.has_both = ('classification' in heads and 'regression' in heads)
        self.has_clf  = 'classification' in heads
        self.has_reg  = 'regression' in heads

        # Optional per-class sample weights (set via set_class_weight)
        self._class_weight = None

    # ── Public API ────────────────────────────────────────────────────────────

    def set_class_weight(self, class_weight_dict):
        """
        Store class weights for use during training.
        class_weight_dict: {0: w0, 1: w1} or None
        """
        self._class_weight = class_weight_dict

    # ── Loss computation ──────────────────────────────────────────────────────

    def _compute_loss(self, y, preds, training=False):
        """Compute appropriate loss based on head configuration."""
        if self.has_both:
            y_reg  = tf.reshape(tf.cast(y[0], tf.float32), [-1])
            y_clf  = tf.reshape(tf.cast(y[1], tf.float32), [-1])
            p_reg  = tf.reshape(preds[0], [-1])
            p_clf  = tf.reshape(preds[1], [-1])
            mse = tf.reduce_mean(tf.square(y_reg - p_reg))
            bce = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_clf, p_clf)
            )
            return fixed_weighted_loss(mse, bce)

        elif self.has_clf:
            y_clf = tf.reshape(tf.cast(y, tf.float32), [-1])
            p_clf = tf.reshape(preds, [-1])
            # Apply sample weights if class_weight is set
            sample_w = self._get_sample_weights(y_clf)
            bce_per_sample = tf.keras.losses.binary_crossentropy(y_clf, p_clf)
            if sample_w is not None:
                return tf.reduce_mean(bce_per_sample * sample_w)
            return tf.reduce_mean(bce_per_sample)

        else:  # regression only
            y_reg = tf.reshape(tf.cast(y, tf.float32), [-1])
            p_reg = tf.reshape(preds, [-1])
            return tf.reduce_mean(tf.square(y_reg - p_reg))

    def _get_sample_weights(self, y_clf_tensor):
        """Convert class_weight_dict to per-sample weights tensor."""
        if self._class_weight is None:
            return None
        y_int = tf.cast(tf.round(y_clf_tensor), tf.int32)
        w0 = float(self._class_weight.get(0, 1.0))
        w1 = float(self._class_weight.get(1, 1.0))
        weights = tf.where(tf.equal(y_int, 1), w1, w0)
        return tf.cast(weights, tf.float32)

    # ── Training / evaluation steps ───────────────────────────────────────────

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            loss  = self._compute_loss(y, preds, training=True)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return self._collect_metrics(loss, y, preds)

    def test_step(self, data):
        x, y = data
        preds = self(x, training=False)
        loss  = self._compute_loss(y, preds, training=False)
        return self._collect_metrics(loss, y, preds)

    def _collect_metrics(self, loss, y, preds):
        results = {'loss': loss}

        # Classification accuracy (always reported when clf head present)
        if self.has_clf:
            clf_pred = tf.reshape(preds[1] if self.has_both else preds, [-1])
            clf_true = tf.reshape(tf.cast(y[1] if self.has_both else y, tf.float32), [-1])
            acc = tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.cast(clf_pred > 0.5, tf.float32), clf_true),
                    tf.float32,
                )
            )
            results['binary_accuracy'] = acc

        # Regression MSE (reported separately for dual-head / reg-only)
        if self.has_reg:
            reg_pred = tf.reshape(preds[0] if self.has_both else preds, [-1])
            reg_true = tf.reshape(tf.cast(y[0] if self.has_both else y, tf.float32), [-1])
            mse = tf.reduce_mean(tf.square(reg_true - reg_pred))
            results['mse'] = mse

        return results

    @property
    def metrics(self):
        # Reset compiled metric state between epochs
        return []

    def get_config(self):
        config = super().get_config()
        config['heads'] = list(self.heads)
        return config

    @classmethod
    def from_config(cls, config):
        heads = config.pop('heads', ['classification', 'regression'])
        # Reconstruct the functional graph as a plain Model, then re-wrap
        inner = tf.keras.Model.from_config(config)
        outputs = inner.outputs[0] if len(inner.outputs) == 1 else inner.outputs
        return cls(
            functional_inputs=inner.input,
            functional_outputs=outputs,
            heads=heads,
            name=inner.name,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Builder function
# ─────────────────────────────────────────────────────────────────────────────

def build_wsmte(config, use_pso=False, ablation_cfg=None):
    """
    Build and return a WSMTEModel.

    Args:
        config:      global CONFIG dict
        use_pso:     if True, returns concat model (Stage 1); PSO merge is
                     applied externally in pso_weighting.finetune_with_pso_weights()
        ablation_cfg: one entry from CONFIG['ablation_configs'] (e.g. CONFIG['ablation_configs']['A'])
                      If None, defaults to full 9-feature dual-head concat model (like Config G).

    Returns:
        WSMTEModel instance (not yet compiled)
    """
    if ablation_cfg is None:
        ablation_cfg = {
            'features': config['feature_columns'],   # all 9
            'heads':    ['classification', 'regression'],
            'merge':    'concat',
            'use_pso':  False,
        }

    n_features = len(ablation_cfg['features'])
    heads       = ablation_cfg['heads']

    # ── Input ─────────────────────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(config['window_size'], n_features), name='input')

    # ── Three parallel encoder branches ───────────────────────────────────────
    lstm_out = build_lstm_branch(inputs, config)   # [batch, 64]
    tcn_out  = build_tcn_branch(inputs, config)    # [batch, 64]
    gru_out  = build_gru_branch(inputs, config)    # [batch, 64]

    # ── Merge ─────────────────────────────────────────────────────────────────
    # Configs A–G and Stage 1 of H: simple concatenation → [batch, 192]
    # Config H PSO merge is handled externally by pso_weighting module
    merged = tf.keras.layers.Concatenate(name='merge')(
        [lstm_out, tcn_out, gru_out]
    )  # [batch, 192]

    # ── Shared dense ──────────────────────────────────────────────────────────
    x = tf.keras.layers.Dense(
        config['shared_dense_units'],
        activation=config['shared_dense_activation'],
        name='shared_dense',
    )(merged)
    x = tf.keras.layers.Dropout(
        config['shared_dense_dropout'], name='shared_dropout'
    )(x)  # [batch, 64]

    # ── Output heads ──────────────────────────────────────────────────────────
    outputs = []
    if 'regression' in heads:
        outputs.append(build_regression_head(x, config))
    if 'classification' in heads:
        outputs.append(build_classification_head(x, config))

    # Single output — unwrap list so model(x) returns tensor not [tensor]
    if len(outputs) == 1:
        functional_outputs = outputs[0]
    else:
        functional_outputs = outputs

    model = WSMTEModel(
        functional_inputs=inputs,
        functional_outputs=functional_outputs,
        heads=heads,
        name='wsmte',
    )
    return model


def build_wsmte_pso(config, pso_weights, ablation_cfg=None):
    """
    Build WSMTE model with PSO-weighted branch merge (Config F — final proposed model).
    The shared dense layer now takes [batch, 64] input (weighted sum),
    NOT [batch, 192] (concatenation).

    pso_weights: array-like [w1, w2, w3] (softmax-normalised, sum=1)
    """
    if ablation_cfg is None:
        ablation_cfg = config['ablation_configs']['F']

    n_features = len(ablation_cfg['features'])
    heads       = ablation_cfg['heads']
    w = np.asarray(pso_weights, dtype=np.float32)

    inputs   = tf.keras.Input(shape=(config['window_size'], n_features), name='input')
    lstm_out = build_lstm_branch(inputs, config)
    # keras-tcn's return_sequences=False uses an internal Lambda slicer, producing a
    # non-standard KerasTensor that breaks arithmetic ops. Wrap through Activation('linear')
    # (identity) to create a clean KerasTensor before scaling.
    tcn_out  = tf.keras.layers.Activation('linear', name='tcn_norm')(build_tcn_branch(inputs, config))
    gru_out  = build_gru_branch(inputs, config)

    # PSO-weighted sum → [batch, 64]
    lstm_scaled = tf.keras.layers.Rescaling(scale=float(w[0]), offset=0., name='lstm_scale')(lstm_out)
    tcn_scaled  = tf.keras.layers.Rescaling(scale=float(w[1]), offset=0., name='tcn_scale')(tcn_out)
    gru_scaled  = tf.keras.layers.Rescaling(scale=float(w[2]), offset=0., name='gru_scale')(gru_out)
    merged = tf.keras.layers.Add(name='pso_merge')([lstm_scaled, tcn_scaled, gru_scaled])

    x = tf.keras.layers.Dense(
        config['shared_dense_units'],
        activation=config['shared_dense_activation'],
        name='shared_dense',
    )(merged)
    x = tf.keras.layers.Dropout(
        config['shared_dense_dropout'], name='shared_dropout'
    )(x)

    outputs = []
    if 'regression' in heads:
        outputs.append(build_regression_head(x, config))
    if 'classification' in heads:
        outputs.append(build_classification_head(x, config))

    functional_outputs = outputs[0] if len(outputs) == 1 else outputs

    model = WSMTEModel(
        functional_inputs=inputs,
        functional_outputs=functional_outputs,
        heads=heads,
        name='wsmte_pso',
    )
    return model
