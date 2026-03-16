"""
src/models/losses.py
Uncertainty-weighted multi-task loss (Kendall et al., CVPR 2018).

Formula:
  L = (exp(-log_sigma1)) * MSE  + log_sigma1
    + (exp(-log_sigma2)) * BCE  + log_sigma2

Where log_sigma1, log_sigma2 are trainable tf.Variable parameters
initialized to 0.0 (i.e., sigma = exp(0) = 1.0).

Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses for
           Scene Geometry and Semantics", Kendall et al., CVPR 2018.
"""

import tensorflow as tf


def uncertainty_weighted_loss(mse, bce, log_sigma1, log_sigma2):
    """
    Compute the combined MTL loss with learnable uncertainty weighting.

    Args:
        mse:        scalar tensor — mean squared error (regression task)
        bce:        scalar tensor — binary cross-entropy (classification task)
        log_sigma1: tf.Variable — log noise for regression (trainable)
        log_sigma2: tf.Variable — log noise for classification (trainable)

    Returns:
        scalar tensor — total loss
    """
    reg_term = tf.exp(-log_sigma1) * mse  + log_sigma1
    clf_term = tf.exp(-log_sigma2) * bce  + log_sigma2
    return reg_term + clf_term
