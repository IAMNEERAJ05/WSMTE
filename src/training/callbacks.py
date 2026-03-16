"""
src/training/callbacks.py
Keras callbacks for WSMTE training runs.
"""

import os
import tensorflow as tf


def get_callbacks(config, run_id='run', save_dir=None):
    """
    Return standard training callbacks: EarlyStopping, ReduceLROnPlateau,
    and ModelCheckpoint.

    Args:
        config:   CONFIG dict
        run_id:   string identifier appended to checkpoint filename
        save_dir: directory for checkpoint; defaults to config['models_dir']

    Returns:
        list of Keras callbacks
    """
    if save_dir is None:
        save_dir = config['models_dir']
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, f'best_{run_id}.keras')

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=config['early_stopping_monitor'],
        patience=config['early_stopping_patience'],
        restore_best_weights=config['restore_best_weights'],
        verbose=1,
    )

    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=config['early_stopping_monitor'],
        factor=config['lr_reduce_factor'],
        patience=config['lr_reduce_patience'],
        min_lr=config['lr_min'],
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=config['early_stopping_monitor'],
        save_best_only=True,
        verbose=0,
    )

    return [early_stop, lr_reduce, checkpoint]


def get_finetune_callbacks(config, run_id='pso_finetune', save_dir=None):
    """
    Callbacks for Stage 3 PSO fine-tuning (shorter patience).

    Returns:
        list of Keras callbacks
    """
    if save_dir is None:
        save_dir = config['models_dir']
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, f'best_{run_id}.keras')

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=config['early_stopping_monitor'],
        patience=config['pso_finetune_patience'],
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=config['early_stopping_monitor'],
        save_best_only=True,
        verbose=0,
    )

    return [early_stop, checkpoint]
