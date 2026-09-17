"""
cart.py
-------
Standard univariate CART (Classification and Regression Trees) implemented
completely from scratch.

WHAT IS CART?
  CART (Breiman et al., 1984) builds a binary decision tree by greedily
  choosing the best split at each node. A split is of the form:
      x_j < t
  where x_j is a single feature and t is a threshold.

  This is called a UNIVARIATE split because it uses only ONE feature.

GINI IMPURITY:
  The split quality is measured by the Gini impurity:
      Gini(S) = 1 - Σ_k p_k²
  where p_k = proportion of class k samples in set S.

  Lower Gini = purer node = better split.
  Perfect purity: Gini = 0 (all samples same class).
  Maximum impurity (binary): Gini = 0.5 (50/50 split).

WEIGHTED GINI:
  For a split that divides S into S_left and S_right:
      Weighted_Gini = (|S_left|/|S|) * Gini(S_left)
                    + (|S_right|/|S|) * Gini(S_right)

  The best split minimizes the weighted Gini.

Paper Reference:
  Kairgeldin & Carreira-Perpiñán, KDD 2024.
  Section 4.2: "Our faster algorithm [BiCART] is based on CART."
  This CART implementation serves as the baseline (Model 1).

  The paper compares bivariate trees against univariate CART (Table 1),
  showing that bivariate trees are both more accurate AND smaller.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from node import Node


# ─────────────────────────────────────────────────────────────────────────────
# GINI IMPURITY
# ─────────────────────────────────────────────────────────────────────────────

def gini_impurity(y):
    """
    Compute Gini impurity of a label array.

    Formula (paper uses this in the context of CART):
        Gini(S) = 1 - Σ_k p_k²

    where p_k = proportion of class k in S.

    Parameters
    ----------
    y : np.ndarray of shape (n,)
        Class labels.

    Returns
    -------
    float : Gini impurity in [0, 0.5] for binary, [0, (K-1)/K] for K-class.

    Example
    -------
    y = [0, 0, 1, 1]  → p_0=0.5, p_1=0.5 → Gini = 1 - 0.25 - 0.25 = 0.5
    y = [0, 0, 0, 0]  → p_0=1.0           → Gini = 1 - 1.0       = 0.0
    """
    if len(y) == 0:
        return 0.0
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs ** 2)


def weighted_gini(y_left, y_right):
    """
    Compute the weighted Gini impurity of a split.

    Formula:
        WG = (|S_left| / |S|) * Gini(S_left)
           + (|S_right| / |S|) * Gini(S_right)

    Parameters
    ----------
    y_left  : np.ndarray — labels of left partition
    y_right : np.ndarray — labels of right partition

    Returns
    -------
    float : weighted Gini impurity
    """
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    w_left = len(y_left) / n
    w_right = len(y_right) / n
    return w_left * gini_impurity(y_left) + w_right * gini_impurity(y_right)


# ─────────────────────────────────────────────────────────────────────────────
# UNIVARIATE SPLIT SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def find_best_univariate_split(X, y):
    """
    Search for the best univariate split: x_j < t.

    Algorithm:
    1. For each feature j:
       a. Sort samples by x_j.
       b. Consider all N-1 midpoint thresholds between consecutive values.
       c. Compute weighted Gini for each threshold.
       d. Track the threshold with minimum weighted Gini.
    2. Return the (feature, threshold, gini) with the global minimum.

    Mapping to paper:
      - This is the standard CART split search.
      - BiCART uses this for univariate candidate splits.
      - BiTAO also evaluates a univariate solution (Solution 2).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)

    Returns
    -------
    best_feature : int or None
    best_threshold : float or None
    best_gini : float
    """
    n_samples, n_features = X.shape
    best_feature = None
    best_threshold = None
    best_gini = float('inf')

    for j in range(n_features):
        values = X[:, j]
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_y = y[sorted_idx]

        # Consider midpoints between consecutive distinct values
        for i in range(1, n_samples):
            if sorted_values[i] == sorted_values[i - 1]:
                continue  # Skip duplicate values

            threshold = (sorted_values[i] + sorted_values[i - 1]) / 2.0
            y_left = sorted_y[:i]
            y_right = sorted_y[i:]

            gini = weighted_gini(y_left, y_right)

            if gini < best_gini:
                best_gini = gini
                best_feature = j
                best_threshold = threshold

    return best_feature, best_threshold, best_gini


# ─────────────────────────────────────────────────────────────────────────────
# CART CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class CARTClassifier:
    """
    Standard univariate CART classification tree, built from scratch.

    This is Model 1 in our project — the baseline.

    Split condition at each node:
        x_j < t  →  go LEFT
        x_j >= t →  go RIGHT

    Stored as a Node tree where each decision node has:
        feature_j = j  (feature index)
        bias = -t       (so: weight_j*x_j + bias < 0 ↔ x_j < t)
        weight_j = 1.0

    Parameters
    ----------
    max_depth : int, default=10
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum samples required in each leaf.

    Implementation choice:
        These stopping criteria are not specified by the paper for CART
        (the paper cross-validates depth). We use max_depth=10 as default
        for the baseline experiment.
    """

    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes_ = None

    def fit(self, X, y):
        """
        Build the CART tree from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_train, n_features)
        y : np.ndarray of shape (n_train,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self.n_classes_ = len(np.unique(y))
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        """
        Recursively build the CART tree.

        Pseudocode:
        ─────────────────────────────────────────────
        CART_BUILD(X, y, depth):
          node = new Node
          node.n_samples = len(y)
          node.depth = depth

          IF stopping criteria:
            node.is_leaf = True
            node.label = majority_class(y)
            return node

          feature, threshold, gini = FIND_BEST_SPLIT(X, y)

          IF no valid split found:
            node.is_leaf = True
            node.label = majority_class(y)
            return node

          Split X, y using feature < threshold
          node.feature_j = feature
          node.weight_j = 1.0
          node.bias = -threshold
          node.node_type = "univariate"

          node.left  = CART_BUILD(X_left,  y_left,  depth+1)
          node.right = CART_BUILD(X_right, y_right, depth+1)
          return node
        ─────────────────────────────────────────────
        """
        node = Node()
        node.depth = depth
        node.n_samples = len(y)

        # ── Stopping criteria ──────────────────────────────────────────────
        if (depth >= self.max_depth
                or len(y) < self.min_samples_split
                or gini_impurity(y) == 0.0):
            node.is_leaf = True
            node.label = _majority_class(y)
            node.node_type = "leaf"
            return node

        # ── Find best split ────────────────────────────────────────────────
        feature, threshold, gini = find_best_univariate_split(X, y)

        if feature is None:
            node.is_leaf = True
            node.label = _majority_class(y)
            node.node_type = "leaf"
            return node

        # ── Apply split ────────────────────────────────────────────────────
        mask_left = X[:, feature] < threshold
        mask_right = ~mask_left

        if (mask_left.sum() < self.min_samples_leaf
                or mask_right.sum() < self.min_samples_leaf):
            node.is_leaf = True
            node.label = _majority_class(y)
            node.node_type = "leaf"
            return node

        # ── Store split in paper's parameterization ────────────────────────
        # Condition: w_j * x_j + b < 0  ↔  x_j < threshold
        # So: w_j = 1.0, b = -threshold
        node.feature_j = feature
        node.weight_j = 1.0
        node.bias = -threshold
        node.node_type = "univariate"

        # ── Recurse ────────────────────────────────────────────────────────
        node.left = self._build(X[mask_left], y[mask_left], depth + 1)
        node.right = self._build(X[mask_right], y[mask_right], depth + 1)

        return node

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_test, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_test,)
        """
        X = np.array(X, dtype=float)
        return np.array([self.root.predict_sample(x) for x in X])

    def print_tree(self):
        """Print the tree structure."""
        if self.root:
            self.root.print_tree()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _majority_class(y):
    """Return the most frequent class in y."""
    classes, counts = np.unique(y, return_counts=True)
    return int(classes[np.argmax(counts)])
