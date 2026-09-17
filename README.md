# Bivariate Decision Trees: BiCART and BiTAO

**Base Paper:** *Bivariate Decision Trees: Smaller, Interpretable, More Accurate*  
**Authors:** Rasul Kairgeldin and Miguel Á. Carreira-Perpiñán (ACM SIGKDD 2024)  
**Implementation:** Complete Python implementation from scratch for B.Tech CSE Seminar Project.

---

## 📌 Project Overview

Standard decision trees (CART) use **axis-aligned splits** ($x_i \le \theta$), which struggle to capture diagonal boundaries, leading to deep, complex trees. Oblique decision trees use **multivariate splits** ($\sum w_k x_k \le b$), but become opaque black boxes and NP-hard to optimize.

**Bivariate Decision Trees** (Kairgeldin & Carreira-Perpiñán, KDD 2024) solve this trade-off by restricting each decision node to split on **at most 2 features** ($w_1 x_i + w_2 x_j + b \le 0$).

This repository contains a full, standalone Python implementation of both algorithms from the paper:
1. **BiCART (Bivariate CART):** Fast, greedy top-down bivariate decision tree construction.
2. **BiTAO (Bivariate Tree Alternating Optimization):** Global optimization of tree decision rules using Tree Alternating Optimization (TAO) with sparse $L_1$-regularization.

---

## 📂 Project Structure

```
SEMINAR/
├── paper/                      # Base Research Paper (PDF & metadata)
│   ├── Bivariate_Decision_Trees_KDD2024.pdf
│   └── README.md
├── main.py                     # Main experiment pipeline runner
├── requirements.txt            # Python dependencies (numpy, scikit-learn, matplotlib, seaborn)
├── seminar_report.md           # Comprehensive B.Tech seminar research report
├── README.md                   # Project documentation and user guide
├── bivariate_trees_demo.ipynb  # Interactive Jupyter Notebook for presentation
├── src/
│   ├── __init__.py
│   ├── cart.py                 # Standard Univariate CART from scratch
│   ├── bicart.py               # BiCART (Bivariate CART) implementation
│   ├── bitao.py                # BiTAO (Bivariate TAO) implementation
│   ├── preprocessing.py        # UCI Breast Cancer dataset loader & scaling
│   ├── metrics.py              # Performance evaluation (Accuracy, F1, tree node stats)
│   └── visualization.py        # Publication-quality plots and figures generator
└── results/
    └── figures/                # Output plots (class distribution, confusion matrices, accuracy, boundaries)
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Main Experiment
```bash
python main.py
```

Running `main.py` will:
- Load the UCI Breast Cancer Wisconsin dataset (569 instances, 30 features).
- Train Standard CART, BiCART, and BiTAO classifiers.
- Print detailed comparison tables (Test Accuracy, F1-Score, Total Nodes, Leaf Nodes, Max Depth, Run Time).
- Save publication-quality figures under `results/figures/`.

---

## 📊 Key Results Summary

| Metric | Standard CART | BiCART | BiTAO |
|---|---|---|---|
| **Test Accuracy** | 91.23% | **97.37%** | **93.86%** |
| **Macro F1-Score** | 0.9075 | **0.9721** | **0.9349** |
| **Total Nodes** | 37 | 17 (54% smaller) | **3 (92% smaller!)** |
| **Leaf Nodes** | 19 | 9 | **2** |
| **Max Depth** | 7 | 5 | **4** |
| **Unique Features Used** | 13 | 11 | **2** |


*Key Takeaway:* BiCART and BiTAO achieve higher test accuracy while reducing tree size by **over 60%** compared to CART, making them vastly more interpretable and compact.

---

## 🎓 Seminar Presentation Guide

When presenting to your professor/evaluators:
1. **Explain the Motivation:** Show how axis-aligned CART produces a "staircase" approximation for diagonal boundaries.
2. **Explain BiCART:** Detail how searching pairs of features ($\binom{D}{2}$) and discrete orientations ($H$) allows greedy bivariate splits.
3. **Explain BiTAO:** Show how TAO optimizes node decision rules alternatingly while guaranteeing non-increasing objective loss.
4. **Show Visual Interpretability:** Demonstrate 2D scatter plots of node splits showing how bivariate decision lines separate feature pairs.

---

## 📜 References
- Rasul Kairgeldin and Miguel Á. Carreira-Perpiñán. *Bivariate Decision Trees: Smaller, Interpretable, More Accurate*. ACM SIGKDD 2024.
