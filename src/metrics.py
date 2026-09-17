"""
metrics.py
----------
Evaluation metrics for CART, BiCART, and BiTAO models.

Implemented from scratch (no sklearn metric functions used for the
main classification metrics, only numpy).

Metrics implemented:
  - Accuracy
  - Precision (per class + macro)
  - Recall (per class + macro)
  - F1-score (per class + macro)
  - Confusion Matrix

Tree complexity metrics:
  - Total nodes
  - Decision (internal) nodes
  - Leaf nodes
  - Tree depth
  - Number of bivariate nodes
  - Number of univariate nodes
  - Number of features used
"""

import numpy as np
import time


# ─────────────────────────────────────────────────────────────────────────────
# Classification Metrics
# ─────────────────────────────────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred, n_classes=None):
    """
    Compute confusion matrix.

    C[i, j] = number of samples with true label i predicted as j.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    y_pred : array-like of shape (n,)
    n_classes : int, optional

    Returns
    -------
    C : np.ndarray of shape (n_classes, n_classes)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if n_classes is None:
        n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    C = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[int(t), int(p)] += 1
    return C


def accuracy(y_true, y_pred):
    """
    Accuracy = correct predictions / total predictions.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    float in [0, 1]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true, y_pred, n_classes=None):
    """
    Compute per-class and macro-averaged Precision, Recall, F1.

    For each class k:
      Precision_k = TP_k / (TP_k + FP_k)
      Recall_k    = TP_k / (TP_k + FN_k)
      F1_k        = 2 * P_k * R_k / (P_k + R_k)

    Macro average: simple mean over all classes.

    Returns
    -------
    dict with keys: 'precision', 'recall', 'f1', 'macro_precision',
                    'macro_recall', 'macro_f1'
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if n_classes is None:
        n_classes = len(np.unique(np.concatenate([y_true, y_pred])))

    C = confusion_matrix(y_true, y_pred, n_classes)

    precisions, recalls, f1s = [], [], []
    for k in range(n_classes):
        tp = C[k, k]
        fp = C[:, k].sum() - tp
        fn = C[k, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        'precision': precisions,
        'recall': recalls,
        'f1': f1s,
        'macro_precision': float(np.mean(precisions)),
        'macro_recall': float(np.mean(recalls)),
        'macro_f1': float(np.mean(f1s)),
    }


def compute_all_metrics(y_true, y_pred, n_classes=None):
    """
    Compute all classification metrics.

    Returns
    -------
    dict with all metrics
    """
    acc = accuracy(y_true, y_pred)
    prf = precision_recall_f1(y_true, y_pred, n_classes)
    cm = confusion_matrix(y_true, y_pred, n_classes)

    results = {
        'accuracy': acc,
        'macro_precision': prf['macro_precision'],
        'macro_recall': prf['macro_recall'],
        'macro_f1': prf['macro_f1'],
        'per_class_precision': prf['precision'],
        'per_class_recall': prf['recall'],
        'per_class_f1': prf['f1'],
        'confusion_matrix': cm,
    }
    return results


def print_metrics(name, y_true, y_pred, class_names=None):
    """
    Print a nicely formatted metrics report.
    """
    metrics = compute_all_metrics(y_true, y_pred)
    n_classes = len(metrics['per_class_precision'])
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    print(f"\n{'='*50}")
    print(f"  Results for: {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  "
          f"({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision : {metrics['macro_precision']:.4f}  (macro)")
    print(f"  Recall    : {metrics['macro_recall']:.4f}  (macro)")
    print(f"  F1-Score  : {metrics['macro_f1']:.4f}  (macro)")
    print()
    print("  Per-class metrics:")
    for k in range(n_classes):
        print(f"    Class {k} ({class_names[k]}): "
              f"P={metrics['per_class_precision'][k]:.3f}  "
              f"R={metrics['per_class_recall'][k]:.3f}  "
              f"F1={metrics['per_class_f1'][k]:.3f}")
    print()
    print("  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    header = "         " + "  ".join(f"Pred-{c[:4]:4s}" for c in class_names)
    print(header)
    for k in range(n_classes):
        row = f"  True-{class_names[k][:4]:4s}: " + "  ".join(
            f"{cm[k, j]:8d}" for j in range(n_classes)
        )
        print(row)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Tree Complexity Metrics
# ─────────────────────────────────────────────────────────────────────────────

def tree_statistics(tree_root):
    """
    Compute tree complexity metrics.

    Paper motivation: The paper emphasizes that bivariate trees are
    *smaller* than univariate trees. We measure this with:
      - Total nodes
      - Internal (decision) nodes
      - Leaf nodes
      - Tree depth (Δ in paper notation)
      - Number of bivariate decision nodes
      - Number of univariate decision nodes
      - Number of unique features used

    Parameters
    ----------
    tree_root : Node

    Returns
    -------
    dict
    """
    total, dec, leaf = tree_root.count_nodes()
    depth = tree_root.max_depth()
    biv, univ = tree_root.count_bivariate()

    # Collect all features used
    features_used = set()
    _collect_features(tree_root, features_used)

    stats = {
        'total_nodes': total,
        'decision_nodes': dec,
        'leaf_nodes': leaf,
        'tree_depth': depth,
        'bivariate_nodes': biv,
        'univariate_nodes': univ,
        'zero_nodes': dec - biv - univ,
        'unique_features_used': len(features_used),
        'features_list': sorted(features_used),
    }
    return stats


def _collect_features(node, features_set):
    """Recursively collect all feature indices used in decision nodes."""
    if node.is_leaf:
        return
    if node.feature_j is not None:
        features_set.add(node.feature_j)
    if node.feature_k is not None:
        features_set.add(node.feature_k)
    _collect_features(node.left, features_set)
    _collect_features(node.right, features_set)


def print_tree_stats(name, stats):
    """Print tree complexity statistics."""
    print(f"\n  Tree Statistics — {name}")
    print(f"  {'─'*40}")
    print(f"  Total nodes       : {stats['total_nodes']}")
    print(f"  Decision nodes    : {stats['decision_nodes']}")
    print(f"  Leaf nodes        : {stats['leaf_nodes']}")
    print(f"  Tree depth (Δ)    : {stats['tree_depth']}")
    print(f"  Bivariate nodes   : {stats['bivariate_nodes']}")
    print(f"  Univariate nodes  : {stats['univariate_nodes']}")
    print(f"  Zero-variate nodes: {stats['zero_nodes']}")
    print(f"  Unique features   : {stats['unique_features_used']}")


# ─────────────────────────────────────────────────────────────────────────────
# Timing Utility
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

    def __str__(self):
        return f"{self.elapsed:.4f}s"
