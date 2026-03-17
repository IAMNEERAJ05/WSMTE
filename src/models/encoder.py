"""
src/models/encoder.py
LSTM, TCN, and GRU branch builders for the parallel WSMTE encoder.
All return output tensors of shape [batch, 64].
"""

import tensorflow as tf
from tcn import TCN


def build_lstm_branch(inputs, config):
    """
    LSTM branch.
    units=64, tanh, sigmoid gates, dropout=0.2, recurrent_dropout=0.0,
    return_sequences=False → output shape [batch, 64]
    """
    x = tf.keras.layers.LSTM(
        units=config['lstm_units'],
        activation='tanh',
        recurrent_activation='sigmoid',
        dropout=config['lstm_dropout'],
        recurrent_dropout=config['lstm_recurrent_dropout'],
        return_sequences=False,
        name='lstm_branch',
    )(inputs)
    return x


def build_gru_branch(inputs, config):
    """
    GRU branch.
    units=64, tanh, sigmoid gates, dropout=0.2, recurrent_dropout=0.0,
    return_sequences=False → output shape [batch, 64]
    """
    x = tf.keras.layers.GRU(
        units=config['gru_units'],
        activation='tanh',
        recurrent_activation='sigmoid',
        dropout=config['gru_dropout'],
        recurrent_dropout=config['gru_recurrent_dropout'],
        return_sequences=False,
        name='gru_branch',
    )(inputs)
    return x


def build_tcn_branch(inputs, config):
    """
    TCN branch via philipperemy/keras-tcn.
    filters=64, kernel_size=2, dilations=[1,2,4], causal padding,
    relu, dropout=0.2, skip_connections=True, no batch/layer/weight norm.
    Receptive field = 1 + (2-1)*(1+2+4) = 8 > window_size=5 ✓
    Output shape [batch, 64]
    """
    x = TCN(
        nb_filters=config['tcn_filters'],
        kernel_size=config['tcn_kernel_size'],
        dilations=config['tcn_dilations'],
        padding=config['tcn_padding'],
        activation=config['tcn_activation'],
        dropout_rate=config['tcn_dropout'],
        use_skip_connections=config['tcn_use_skip_connections'],
        use_batch_norm=config['tcn_use_batch_norm'],
        use_layer_norm=config['tcn_use_layer_norm'],
        return_sequences=False,
        name='tcn_branch',
    )(inputs)
    return x
