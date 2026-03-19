"""
src/models/losses.py
Fixed-weight multi-task loss for WSMTE.

Formula:
  L = 0.3 × MSE + 0.7 × BCE

No trainable uncertainty parameters.
"""

import tensorflow as tf


def fixed_weighted_loss(mse, bce):
    """
    Compute the combined MTL loss with fixed weights.

    Args:
        mse: scalar tensor — mean squared error (regression task)
        bce: scalar tensor — binary cross-entropy (classification task)

    Returns:
        scalar tensor — total loss
    """
    return 0.3 * mse + 0.7 * bce
