"""
utils/data_utils.py
Centralised split and anti-leakage normalisation utilities for all BCI experiment paradigms.
Ensures no data leakage and a consistent test set across experiments.
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def three_way_split(X, y, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Partition data into Train, Validation, and Test subsets using stratified sampling.
    Falls back to random splitting if stratification fails due to severely imbalanced classes.
    """
    test_size_actual = test_ratio
    val_relative_ratio = val_ratio / (1.0 - test_size_actual)

    try:
        # Attempt 1: Stratified split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size_actual, random_state=random_state, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_ratio, random_state=random_state, stratify=y_temp
        )

    except ValueError:
        # Fallback: non-stratified split when a class has only one sample
        print("      [WARNING] Data too imbalanced for stratification. Falling back to random split.")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state, stratify=None
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_ratio, random_state=random_state, stratify=None
        )

    return X_train, X_val, X_test, y_train, y_val, y_test

def fit_and_apply_scaler(X_train, X_val, X_test, save_path=None):
    """
    Fit a StandardScaler exclusively on X_train and transform all three splits.
    Supports both 3-D (classical ML) and 4-D (EEGNet) input arrays.
    Optionally serialises the fitted scaler to disk for later inference use.
    """
    original_shape_train = X_train.shape
    original_shape_val = X_val.shape
    original_shape_test = X_test.shape

    scaler = StandardScaler()

    # Flatten to 2-D (samples × features) so StandardScaler can operate on any input rank
    X_train_2d = X_train.reshape(len(X_train), -1)
    X_val_2d = X_val.reshape(len(X_val), -1)
    X_test_2d = X_test.reshape(len(X_test), -1)

    # Fit only on training data, then transform all splits independently
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(original_shape_train)
    X_val_scaled = scaler.transform(X_val_2d).reshape(original_shape_val)
    X_test_scaled = scaler.transform(X_test_2d).reshape(original_shape_test)

    if save_path:
        parent_dir = os.path.dirname(os.path.abspath(save_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
