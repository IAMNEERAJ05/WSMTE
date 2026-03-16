"""
src/models/heads.py
Regression and classification output heads for WSMTE.
Both take a [batch, 64] shared representation as input.
"""

import tensorflow as tf


def build_regression_head(x, config):
    """
    Regression head: Dense(16, relu) → Dense(1, linear)
    Loss: MSE  |  Output: next-day normalized Close_d

    Args:
        x:      tensor of shape [batch, 64] from shared dense layer
        config: CONFIG dict

    Returns:
        tensor of shape [batch, 1]
    """
    x = tf.keras.layers.Dense(
        config['head_dense_units'],
        activation=config['head_dense_activation'],
        name='reg_dense',
    )(x)
    out = tf.keras.layers.Dense(1, activation='linear', name='reg_output')(x)
    return out


def build_classification_head(x, config):
    """
    Classification head: Dense(16, relu) → Dense(1, sigmoid)
    Loss: BCE  |  Output: P(next-day close UP) ∈ [0, 1]

    Args:
        x:      tensor of shape [batch, 64] from shared dense layer
        config: CONFIG dict

    Returns:
        tensor of shape [batch, 1]
    """
    x = tf.keras.layers.Dense(
        config['head_dense_units'],
        activation=config['head_dense_activation'],
        name='clf_dense',
    )(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='clf_output')(x)
    return out
