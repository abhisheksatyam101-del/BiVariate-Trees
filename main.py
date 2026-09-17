"""
main.py
-------
Main experiment runner for the Bivariate Decision Trees project.

Runs all three experiments (CART, BiCART, BiTAO) and produces results.

Usage:
    python main.py
"""

import numpy as np
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import get_data
from cart import CARTClassifier
from bicart import BiCARTClassifier
from bitao import BiTAOClassifier
from metrics import (compute_all_metrics, print_metrics,
                     tree_statistics, print_tree_stats, Timer)
from visualization import (
    plot_class_distribution, plot_confusion_matrix,
    plot_accuracy_comparison, plot_f1_comparison,
    plot_tree_size_comparison, plot_training_time,
    plot_decision_boundary_2d, plot_bivariate_split_illustration,
    plot_objective_convergence, plot_summary_table
)


def run_experiment():
    """Run the complete experiment."""
    os.makedirs("results/figures", exist_ok=True)

    print("=" * 70)
    print("  BIVARIATE DECISION TREES: SEMINAR PROJECT")
    print("  Paper: Kairgeldin & Carreira-Perpiñán, KDD 2024")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # HYPERPARAMETERS
    # ─────────────────────────────────────────────────────────────────────
    # Document which come from paper and which are implementation choices.

    # Paper-specified or dataset-driven:
    RANDOM_SEED = 42          # Implementation choice
    TEST_SIZE = 0.2            # ~80/20 split (paper: 3 repeated splits)

    # CART hyperparameters
    CART_MAX_DEPTH = 7         # Implementation choice (paper cross-validates)
    CART_MIN_SAMPLES = 2       # Implementation choice

    # BiCART hyperparameters
    BICART_MAX_DEPTH = 5       # Implementation choice
    BICART_H = 36              # H=36 orientations (5° steps) — Impl. choice
    BICART_MIN_SAMPLES = 2     # Implementation choice

    # BiTAO hyperparameters
    BITAO_INIT_DEPTH = 4       # Initial CART depth — Implementation choice
    BITAO_LAMBDA = 1.0         # λ=1.0 — implementation choice (paper: cross-val)
    BITAO_C = 1.5              # C=1.5 — paper recommends C ∈ [1.1, 1.5]
    BITAO_H = 36               # H=36 — Implementation choice
    BITAO_MAX_ITER = 15        # Max TAO iterations — Implementation choice

    print("\n[HYPERPARAMETERS]")
    print(f"  CART:    max_depth={CART_MAX_DEPTH}, min_samples={CART_MIN_SAMPLES}")
    print(f"  BiCART:  max_depth={BICART_MAX_DEPTH}, H={BICART_H}")
    print(f"  BiTAO:   init_depth={BITAO_INIT_DEPTH}, lam={BITAO_LAMBDA}, "
          f"C={BITAO_C}, H={BITAO_H}, max_iter={BITAO_MAX_ITER}")
    print(f"  Split:   test_size={TEST_SIZE}, seed={RANDOM_SEED}")

    # ─────────────────────────────────────────────────────────────────────
    # LOAD DATA
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[PHASE 1] Loading and preprocessing data...")
    print("─" * 70)

    X_train, X_test, y_train, y_test, feature_names, class_names = get_data(
        test_size=TEST_SIZE, random_state=RANDOM_SEED, scale=True
    )

    print(f"\nDataset: Breast Cancer Wisconsin (UCI)")
    print(f"  Used in paper: Table 1, Section 5.1.1")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test : {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {class_names}")

    # Plot class distribution
    plot_class_distribution(y_train, class_names)

    # ─────────────────────────────────────────────────────────────────────
    # MODEL 1: STANDARD CART
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[PHASE 2] Model 1: Standard Univariate CART (Baseline)")
    print("─" * 70)

    cart = CARTClassifier(
        max_depth=CART_MAX_DEPTH,
        min_samples_split=CART_MIN_SAMPLES
    )
    with Timer() as t_cart:
        cart.fit(X_train, y_train)
    time_cart = t_cart.elapsed

    print(f"  Training time: {time_cart:.4f}s")

    with Timer() as tp_cart:
        y_pred_cart = cart.predict(X_test)
    time_pred_cart = tp_cart.elapsed

    metrics_cart = print_metrics("CART", y_test, y_pred_cart, class_names)
    stats_cart = tree_statistics(cart.root)
    print_tree_stats("CART", stats_cart)

    # ─────────────────────────────────────────────────────────────────────
    # MODEL 2: BiCART
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[PHASE 3] Model 2: Bivariate CART (BiCART)")
    print("  Paper reference: Section 4.2")
    print("─" * 70)

    bicart = BiCARTClassifier(
        max_depth=BICART_MAX_DEPTH,
        H=BICART_H,
        min_samples_split=BICART_MIN_SAMPLES
    )
    with Timer() as t_bicart:
        bicart.fit(X_train, y_train)
    time_bicart = t_bicart.elapsed

    print(f"  Training time: {time_bicart:.4f}s")

    with Timer() as tp_bicart:
        y_pred_bicart = bicart.predict(X_test)
    time_pred_bicart = tp_bicart.elapsed

    metrics_bicart = print_metrics("BiCART", y_test, y_pred_bicart, class_names)
    stats_bicart = tree_statistics(bicart.root)
    print_tree_stats("BiCART", stats_bicart)

    # ─────────────────────────────────────────────────────────────────────
    # MODEL 3: BiTAO
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[PHASE 4] Model 3: Bivariate TAO (BiTAO)")
    print("  Paper reference: Section 4.1")
    print("─" * 70)

    bitao = BiTAOClassifier(
        max_depth=BITAO_INIT_DEPTH,
        lam=BITAO_LAMBDA,
        C=BITAO_C,
        H=BITAO_H,
        max_iter=BITAO_MAX_ITER
    )
    with Timer() as t_bitao:
        bitao.fit(X_train, y_train)
    time_bitao = t_bitao.elapsed

    print(f"  Training time: {time_bitao:.4f}s")

    with Timer() as tp_bitao:
        y_pred_bitao = bitao.predict(X_test)
    time_pred_bitao = tp_bitao.elapsed

    metrics_bitao = print_metrics("BiTAO", y_test, y_pred_bitao, class_names)
    stats_bitao = tree_statistics(bitao.root)
    print_tree_stats("BiTAO", stats_bitao)

    # ─────────────────────────────────────────────────────────────────────
    # RESULTS SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON: CART vs BiCART vs BiTAO")
    print("=" * 70)

    all_results = {
        'CART': metrics_cart,
        'BiCART': metrics_bicart,
        'BiTAO': metrics_bitao,
    }
    all_stats = {
        'CART': stats_cart,
        'BiCART': stats_bicart,
        'BiTAO': stats_bitao,
    }
    all_times = {
        'CART': time_cart,
        'BiCART': time_bicart,
        'BiTAO': time_bitao,
    }
    pred_times = {
        'CART': time_pred_cart,
        'BiCART': time_pred_bicart,
        'BiTAO': time_pred_bitao,
    }

    # Print comparison table
    print(f"\n{'Metric':<25} {'CART':>12} {'BiCART':>12} {'BiTAO':>12}")
    print("─" * 65)
    for metric, key in [
        ('Accuracy', 'accuracy'),
        ('Macro Precision', 'macro_precision'),
        ('Macro Recall', 'macro_recall'),
        ('Macro F1', 'macro_f1'),
    ]:
        vals = [all_results[m][key] for m in ['CART', 'BiCART', 'BiTAO']]
        print(f"{metric:<25} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}")

    print("─" * 65)
    for metric, key in [
        ('Total Nodes', 'total_nodes'),
        ('Tree Depth (Δ)', 'tree_depth'),
        ('Bivariate Nodes', 'bivariate_nodes'),
        ('Unique Features', 'unique_features_used'),
    ]:
        vals = [all_stats[m][key] for m in ['CART', 'BiCART', 'BiTAO']]
        print(f"{metric:<25} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    print("─" * 65)
    print(f"{'Train Time (s)':<25} {time_cart:>12.4f} "
          f"{time_bicart:>12.4f} {time_bitao:>12.4f}")
    print(f"{'Pred Time (s)':<25} {time_pred_cart:>12.4f} "
          f"{time_pred_bicart:>12.4f} {time_pred_bitao:>12.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # SAVE RESULTS CSV
    # ─────────────────────────────────────────────────────────────────────
    import csv
    with open("results/results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1',
                         'Total_Nodes', 'Depth', 'Bivariate_Nodes',
                         'Features_Used', 'Train_Time', 'Pred_Time'])
        for m in ['CART', 'BiCART', 'BiTAO']:
            r, s = all_results[m], all_stats[m]
            writer.writerow([
                m,
                f"{r['accuracy']:.4f}",
                f"{r['macro_precision']:.4f}",
                f"{r['macro_recall']:.4f}",
                f"{r['macro_f1']:.4f}",
                s['total_nodes'],
                s['tree_depth'],
                s['bivariate_nodes'],
                s['unique_features_used'],
                f"{all_times[m]:.4f}",
                f"{pred_times[m]:.4f}",
            ])
    print("\n  Results saved to: results/results.csv")

    # ─────────────────────────────────────────────────────────────────────
    # VISUALIZATIONS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("[PHASE 5] Generating visualizations...")
    print("─" * 70)

    # Confusion matrices
    for m, y_pred, name in [
        ('CART', y_pred_cart, 'CART'),
        ('BiCART', y_pred_bicart, 'BiCART'),
        ('BiTAO', y_pred_bitao, 'BiTAO'),
    ]:
        cm = all_results[m]['confusion_matrix']
        plot_confusion_matrix(cm, class_names, f'Confusion Matrix — {name}')

    # Comparison charts
    plot_accuracy_comparison(all_results)
    plot_f1_comparison(all_results)
    plot_tree_size_comparison(all_stats)
    plot_training_time(all_times)

    # Bivariate split illustration
    plot_bivariate_split_illustration()

    # Decision boundaries (use first two features for illustration)
    for model, name in [
        (cart, 'CART'), (bicart, 'BiCART'), (bitao, 'BiTAO')
    ]:
        plot_decision_boundary_2d(
            model, X_test, y_test,
            feature_j=0, feature_k=1,
            feature_names=feature_names,
            model_name=name, class_names=class_names
        )

    # BiTAO convergence
    if bitao.objective_history:
        plot_objective_convergence(bitao.objective_history)

    # Summary table
    plot_summary_table(all_results, all_stats, all_times)

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE!")
    print(f"  Figures saved to: results/figures/")
    print(f"  Results saved to: results/results.csv")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
