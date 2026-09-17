"""
visualization.py
----------------
All visualizations for the Bivariate Decision Trees project.

Creates:
  1. Class distribution bar chart
  2. Confusion matrices (CART, BiCART, BiTAO)
  3. Accuracy comparison bar chart
  4. F1 comparison bar chart
  5. Tree-size comparison bar chart
  6. Training-time comparison bar chart
  7. 2D decision boundary for selected feature pair
  8. Bivariate split illustration
  9. Objective convergence (BiTAO)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

# Create results directory
os.makedirs("results/figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'CART': '#4e8ef7',
    'BiCART': '#f7934e',
    'BiTAO': '#5ec96e',
    'malignant': '#e05252',
    'benign': '#52a0e0',
    'bg': '#1a1a2e',
    'fg': '#eaeaea',
}


def _save(fig, name):
    path = f"results/figures/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(y, class_names):
    """Bar chart of class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#16213e')
    ax.set_facecolor('#0f3460')

    unique, counts = np.unique(y, return_counts=True)
    colors = [COLORS['malignant'], COLORS['benign']]
    bars = ax.bar([class_names[i] for i in unique], counts,
                  color=colors, width=0.5, edgecolor='white', linewidth=0.8)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                f'{cnt}\n({cnt/len(y)*100:.1f}%)',
                ha='center', va='bottom', color='white', fontsize=11)

    ax.set_title('Class Distribution — Breast Cancer Dataset',
                 color='white', fontsize=13, pad=12)
    ax.set_ylabel('Number of Samples', color='white')
    ax.tick_params(colors='white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#555')
    ax.set_ylim(0, max(counts) * 1.3)

    return _save(fig, 'class_distribution')


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, title):
    """Heatmap-style confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#16213e')
    ax.set_facecolor('#0f3460')

    cmap = LinearSegmentedColormap.from_list(
        'biv', ['#0f3460', '#4e8ef7', '#ffffff']
    )
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, color='white')
    ax.set_yticklabels(class_names, color='white')
    ax.set_xlabel('Predicted Label', color='white')
    ax.set_ylabel('True Label', color='white')
    ax.set_title(title, color='white', fontsize=12, pad=10)

    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='black' if cm[i, j] > thresh else 'white',
                    fontsize=14, fontweight='bold')

    name = title.lower().replace(' ', '_').replace('—', '').replace('  ', '_')
    return _save(fig, f'cm_{name}')


# ─────────────────────────────────────────────────────────────────────────────
# 3. ACCURACY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(results_dict):
    """Bar chart comparing test accuracy of all three models."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#16213e')
    ax.set_facecolor('#0f3460')

    models = list(results_dict.keys())
    accs = [results_dict[m]['accuracy'] * 100 for m in models]
    colors = [COLORS[m] for m in models]

    bars = ax.bar(models, accs, color=colors, width=0.5,
                  edgecolor='white', linewidth=0.8)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{acc:.2f}%', ha='center', va='bottom',
                color='white', fontsize=12, fontweight='bold')

    ax.set_ylim(min(accs) - 3, 100)
    ax.set_ylabel('Test Accuracy (%)', color='white', fontsize=12)
    ax.set_title('Test Accuracy Comparison\nCART vs BiCART vs BiTAO',
                 color='white', fontsize=13, pad=12)
    ax.tick_params(colors='white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#555')

    return _save(fig, 'accuracy_comparison')


# ─────────────────────────────────────────────────────────────────────────────
# 4. F1 COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_f1_comparison(results_dict):
    """Bar chart comparing macro F1 of all three models."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#16213e')
    ax.set_facecolor('#0f3460')

    models = list(results_dict.keys())
    f1s = [results_dict[m]['macro_f1'] for m in models]
    colors = [COLORS[m] for m in models]

    bars = ax.bar(models, f1s, color=colors, width=0.5,
                  edgecolor='white', linewidth=0.8)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{f1:.4f}', ha='center', va='bottom',
                color='white', fontsize=12, fontweight='bold')

    ax.set_ylim(min(f1s) - 0.05, 1.02)
    ax.set_ylabel('Macro F1-Score', color='white', fontsize=12)
    ax.set_title('F1-Score Comparison\nCART vs BiCART vs BiTAO',
                 color='white', fontsize=13, pad=12)
    ax.tick_params(colors='white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#555')

    return _save(fig, 'f1_comparison')


# ─────────────────────────────────────────────────────────────────────────────
# 5. TREE SIZE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_tree_size_comparison(tree_stats_dict):
    """Grouped bar chart for tree complexity metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor='#16213e')
    fig.suptitle('Tree Complexity Comparison',
                 color='white', fontsize=14, y=1.02)

    models = list(tree_stats_dict.keys())
    metrics_labels = ['Total Nodes', 'Tree Depth', 'Bivariate Nodes']
    metrics_keys = ['total_nodes', 'tree_depth', 'bivariate_nodes']

    for ax, label, key in zip(axes, metrics_labels, metrics_keys):
        ax.set_facecolor('#0f3460')
        vals = [tree_stats_dict[m].get(key, 0) for m in models]
        colors = [COLORS[m] for m in models]
        bars = ax.bar(models, vals, color=colors, width=0.5,
                      edgecolor='white', linewidth=0.8)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(v), ha='center', va='bottom',
                    color='white', fontsize=11, fontweight='bold')

        ax.set_title(label, color='white', fontsize=11)
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#555')

    fig.patch.set_facecolor('#16213e')
    return _save(fig, 'tree_size_comparison')


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING TIME COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_time(times_dict):
    """Horizontal bar chart for training times."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#16213e')
    ax.set_facecolor('#0f3460')

    models = list(times_dict.keys())
    times = [times_dict[m] for m in models]
    colors = [COLORS[m] for m in models]

    bars = ax.barh(models, times, color=colors,
                   edgecolor='white', linewidth=0.8)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max(times) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{t:.3f}s', va='center', color='white',
                fontsize=11, fontweight='bold')

    ax.set_xlabel('Training Time (seconds)', color='white', fontsize=12)
    ax.set_title('Training Time Comparison', color='white', fontsize=13, pad=10)
    ax.tick_params(colors='white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#555')

    return _save(fig, 'training_time')


# ─────────────────────────────────────────────────────────────────────────────
# 7. DECISION BOUNDARY (2D)
# ─────────────────────────────────────────────────────────────────────────────

def plot_decision_boundary_2d(model, X, y, feature_j, feature_k,
                               feature_names, model_name, class_names):
    """
    Visualize the decision boundary in 2D for a selected feature pair.

    NOTE: This represents only the 2D projection onto features j and k.
    The actual model uses all features. This is purely for illustration.

    Parameters
    ----------
    model : fitted classifier (with predict method)
    X : full feature matrix (standardized)
    y : true labels
    feature_j, feature_k : int — feature indices to plot
    feature_names : list
    model_name : str
    class_names : list
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#16213e')
    ax.set_facecolor('#0f3460')

    # Create 2D grid
    x1 = X[:, feature_j]
    x2 = X[:, feature_k]
    margin = 0.5
    x1_min, x1_max = x1.min() - margin, x1.max() + margin
    x2_min, x2_max = x2.min() - margin, x2.max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, 200),
        np.linspace(x2_min, x2_max, 200)
    )

    # Create full-dimension grid (all other features set to 0 = mean)
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_full = np.zeros((grid_2d.shape[0], X.shape[1]))
    grid_full[:, feature_j] = grid_2d[:, 0]
    grid_full[:, feature_k] = grid_2d[:, 1]

    Z = model.predict(grid_full).reshape(xx.shape)

    # Background color map
    cmap_bg = LinearSegmentedColormap.from_list(
        'bg', ['#2d4a8a', '#8a2d4a'], N=2
    )
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, colors='white', linewidths=1.5, alpha=0.7,
               levels=[0.5])

    # Scatter plot
    scatter_colors = [COLORS['malignant'] if yi == 0 else COLORS['benign']
                      for yi in y]
    ax.scatter(x1, x2, c=scatter_colors, s=30, alpha=0.7,
               edgecolors='white', linewidths=0.3)

    ax.set_xlabel(feature_names[feature_j], color='white', fontsize=11)
    ax.set_ylabel(feature_names[feature_k], color='white', fontsize=11)
    ax.set_title(
        f'Decision Boundary — {model_name}\n'
        f'(2D projection: {feature_names[feature_j][:15]} vs '
        f'{feature_names[feature_k][:15]})\n'
        f'Note: Only this feature pair shown; model uses all features.',
        color='white', fontsize=10, pad=10
    )
    ax.tick_params(colors='white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#555')

    patches = [
        mpatches.Patch(color=COLORS['malignant'], label=class_names[0]),
        mpatches.Patch(color=COLORS['benign'], label=class_names[1]),
    ]
    ax.legend(handles=patches, loc='upper right',
              facecolor='#1a1a2e', labelcolor='white', framealpha=0.8)

    name = model_name.lower().replace(' ', '_')
    return _save(fig, f'boundary_{name}')


# ─────────────────────────────────────────────────────────────────────────────
# 8. BIVARIATE SPLIT ILLUSTRATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_bivariate_split_illustration():
    """
    Illustrate how a bivariate split works geometrically.

    Shows:
      - A univariate split (axis-aligned line)
      - A bivariate split (diagonal line at angle θ)
      - The projection z = x_j*cos(θ) + x_k*sin(θ)

    This is a toy example (not real data).
    """
    np.random.seed(42)
    n = 30

    # Class 0: bottom-left cluster
    X0 = np.random.randn(n, 2) * 0.5 + np.array([-1, -1])
    # Class 1: top-right cluster
    X1 = np.random.randn(n, 2) * 0.5 + np.array([1, 1])

    X_toy = np.vstack([X0, X1])
    y_toy = np.array([0] * n + [1] * n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#16213e')
    fig.suptitle('Univariate Split vs Bivariate Split',
                 color='white', fontsize=14, y=1.02)

    for ax, title, show_bivariate in zip(
        axes,
        ['Univariate Split (x₁ < t)', 'Bivariate Split (x₁cos(θ) + x₂sin(θ) < t)'],
        [False, True]
    ):
        ax.set_facecolor('#0f3460')
        ax.scatter(X0[:, 0], X0[:, 1], c=COLORS['malignant'], s=40,
                   label='Class 0', edgecolors='white', linewidths=0.3)
        ax.scatter(X1[:, 0], X1[:, 1], c=COLORS['benign'], s=40,
                   label='Class 1', edgecolors='white', linewidths=0.3)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='#555', linewidth=0.5)
        ax.axvline(0, color='#555', linewidth=0.5)

        if not show_bivariate:
            # Univariate split: x1 < 0
            ax.axvline(x=0.0, color='#f7934e', linewidth=2.5,
                       linestyle='--', label='Split: x₁ < 0')
            ax.fill_betweenx([-3, 3], -3, 0, alpha=0.1,
                             color=COLORS['malignant'])
            ax.fill_betweenx([-3, 3], 0, 3, alpha=0.1,
                             color=COLORS['benign'])
        else:
            # Bivariate split: x1*cos(45°) + x2*sin(45°) < 0
            theta = np.pi / 4  # 45°
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x_line = np.linspace(-3, 3, 100)
            # Split line: cos_t*x + sin_t*y = 0 → y = -cos_t/sin_t * x
            y_line = -(cos_t / sin_t) * x_line
            ax.plot(x_line, y_line, color='#5ec96e', linewidth=2.5,
                    linestyle='--',
                    label=f'Split: x₁cos(45°) + x₂sin(45°) < 0')

            # Draw orientation arrow
            ax.annotate('', xy=(cos_t, sin_t), xytext=(0, 0),
                        arrowprops=dict(color='yellow', arrowstyle='->',
                                        lw=2.0))
            ax.text(cos_t + 0.1, sin_t + 0.1, 'θ=45°',
                    color='yellow', fontsize=10)

        ax.set_xlabel('Feature x₁', color='white')
        ax.set_ylabel('Feature x₂', color='white')
        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#555')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

    fig.patch.set_facecolor('#16213e')
    return _save(fig, 'bivariate_split_illustration')


# ─────────────────────────────────────────────────────────────────────────────
# 9. BiTAO OBJECTIVE CONVERGENCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_objective_convergence(objective_history):
    """Plot the BiTAO objective E(Θ) over iterations."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#16213e')
    ax.set_facecolor('#0f3460')

    iters = list(range(1, len(objective_history) + 1))
    ax.plot(iters, objective_history, color=COLORS['BiTAO'],
            linewidth=2.5, marker='o', markersize=6,
            markerfacecolor='white', markeredgecolor=COLORS['BiTAO'])

    ax.fill_between(iters, objective_history,
                    min(objective_history), alpha=0.2,
                    color=COLORS['BiTAO'])

    ax.set_xlabel('TAO Iteration', color='white', fontsize=12)
    ax.set_ylabel('Objective E(Θ) = 0/1 Loss + λ·Regularization',
                  color='white', fontsize=11)
    ax.set_title('BiTAO Objective Convergence\n'
                 '"until E(Θ) does not strictly decrease" (paper criterion)',
                 color='white', fontsize=12, pad=10)
    ax.tick_params(colors='white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#555')

    return _save(fig, 'bitao_convergence')


# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY COMPARISON TABLE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_table(results_dict, tree_stats_dict, times_dict):
    """Render a summary table as a matplotlib figure."""
    metrics = ['Accuracy', 'Macro Prec.', 'Macro Recall', 'Macro F1',
               'Total Nodes', 'Tree Depth', 'Bivariate Nodes',
               'Unique Features', 'Train Time (s)']

    models = list(results_dict.keys())
    data = []
    for m in models:
        r = results_dict[m]
        s = tree_stats_dict[m]
        row = [
            f"{r['accuracy']*100:.2f}%",
            f"{r['macro_precision']:.4f}",
            f"{r['macro_recall']:.4f}",
            f"{r['macro_f1']:.4f}",
            str(s['total_nodes']),
            str(s['tree_depth']),
            str(s['bivariate_nodes']),
            str(s['unique_features_used']),
            f"{times_dict[m]:.3f}",
        ]
        data.append(row)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#16213e')
    ax.set_facecolor('#16213e')
    ax.axis('off')

    table = ax.table(
        cellText=data,
        rowLabels=models,
        colLabels=metrics,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4e8ef7')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == -1:
            cell.set_facecolor(COLORS.get(models[row - 1], '#333'))
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#0f3460')
            cell.set_text_props(color='white')
        cell.set_edgecolor('#333')

    ax.set_title('Complete Results Summary: CART vs BiCART vs BiTAO',
                 color='white', fontsize=13, pad=15, y=0.95)

    return _save(fig, 'summary_table')
