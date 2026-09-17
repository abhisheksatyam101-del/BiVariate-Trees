"""
preprocessing.py
----------------
Dataset loading, inspection, cleaning, and preprocessing for the
Bivariate Decision Trees project.

Dataset Used:
  Breast Cancer Wisconsin (UCI/sklearn)
  - N = 569 samples
  - D = 30 features (geometric measurements of cell nuclei)
  - K = 2 classes: Malignant (0), Benign (1)
  - Source: sklearn.datasets.load_breast_cancer()

This is one of the exact datasets used in the paper (Table 1):
  "Breast Cancer (455, 30, 2)" — 455 training, 30 features, 2 classes.

Paper reference:
  Section 5.1.1: "Breast Cancer UCI dataset. This is a binary
  classification task into malignant and benign tumors. Each input
  instance contains 30 features extracted from a collection of cells."
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    """
    Load the Breast Cancer Wisconsin dataset.

    Returns
    -------
    X : np.ndarray of shape (569, 30)
        Feature matrix.
    y : np.ndarray of shape (569,)
        Labels: 0 = Malignant, 1 = Benign.
    feature_names : list of str
        Names of the 30 features.
    class_names : list of str
        ['malignant', 'benign']
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y, list(data.feature_names), list(data.target_names)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Inspect Dataset
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset(X, y, feature_names, class_names):
    """
    Print a summary of the dataset.
    """
    print("=" * 60)
    print("DATASET: Breast Cancer Wisconsin (UCI)")
    print("=" * 60)
    print(f"  Total samples   : {X.shape[0]}")
    print(f"  Features        : {X.shape[1]}")
    print(f"  Classes         : {class_names}")
    print()
    print("Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls} ({class_names[cls]}): {cnt} samples "
              f"({cnt/len(y)*100:.1f}%)")
    print()

    df = pd.DataFrame(X, columns=feature_names)
    print("Feature statistics (first 5 features):")
    print(df.iloc[:, :5].describe().round(3).to_string())
    print()

    # Missing value check
    missing = np.sum(np.isnan(X))
    print(f"Missing values: {missing}  (none expected for this dataset)")
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and test sets.

    Implementation choice: 80/20 split, seed=42.
    This matches the paper's cross-validation approach (paper uses 3
    repeated random train/validation splits).

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    test_size : float, default=0.2
    random_state : int, default=42

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test  samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Standardization
# ─────────────────────────────────────────────────────────────────────────────

def standardize(X_train, X_test):
    """
    Standardize features using training-set statistics ONLY.

    Formula: z = (x - μ) / σ

    WHY SCALING IS IMPORTANT FOR BIVARIATE TREES:
    Feature scaling is particularly important for bivariate trees because
    the orientation search projects two features as:
        z = x_j * cos(θ) + x_k * sin(θ)
    If x_j and x_k are on very different scales, the projection will be
    dominated by the larger feature regardless of θ, making the angle
    search meaningless. After standardization, both features have
    mean=0 and std=1, so the geometric interpretation of angle θ is valid.

    Note: Only μ and σ from TRAINING data are used to avoid data leakage.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    X_test  : np.ndarray of shape (n_test, n_features)

    Returns
    -------
    X_train_scaled, X_test_scaled, mean_, std_
    """
    mean_ = X_train.mean(axis=0)
    std_ = X_train.std(axis=0)
    std_[std_ == 0] = 1.0  # Avoid division by zero

    X_train_scaled = (X_train - mean_) / std_
    X_test_scaled = (X_test - mean_) / std_

    print(f"Feature mean (train): min={mean_.min():.3f}, max={mean_.max():.3f}")
    print(f"Feature std  (train): min={std_.min():.3f}, max={std_.max():.3f}")
    print(f"After scaling — Train range: [{X_train_scaled.min():.2f}, "
          f"{X_train_scaled.max():.2f}]")

    return X_train_scaled, X_test_scaled, mean_, std_


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Run full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def get_data(test_size=0.2, random_state=42, scale=True, verbose=True):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, class_names
    """
    X, y, feature_names, class_names = load_dataset()
    if verbose:
        inspect_dataset(X, y, feature_names, class_names)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    if scale:
        X_train, X_test, _, _ = standardize(X_train, X_test)

    return X_train, X_test, y_train, y_test, feature_names, class_names
