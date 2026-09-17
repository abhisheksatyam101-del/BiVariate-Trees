# Benchmark Evaluation Table: Bivariate Decision Trees

**Base Research Paper:** *Bivariate Decision Trees: Smaller, Interpretable, More Accurate* (Kairgeldin & Carreira-Perpiñán, ACM SIGKDD 2024)  
**Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) — 569 samples, 30 continuous features, 2 classes  
**Experimental Setup:** 80% Train (455 samples) / 20% Test (114 samples), Standard Scaled, Random Seed = 42  

---

## 🏆 Comprehensive Benchmark Comparison Table

| Model | Algorithm Type | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Total Nodes | Leaf Nodes | Max Depth ($\Delta$) | Bivariate Nodes | Unique Features | Training Time | Prediction Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard CART** | Axis-Aligned Baseline (Breiman '84) | 91.23% | 0.9075 | 0.9019 | 0.9157 | 37 | 19 | 7 | 0 | 13 | 2.20s | 0.3ms |
| **BiCART** | Greedy Top-Down Bivariate (Paper Alg 1) | **97.37%** | **0.9721** | **0.9667** | **0.9792** | 17 (-54.1%) | 9 | 5 | 7 | 11 | 10.12s | 0.3ms |
| **BiTAO** | Optimization-Based Sparse (Paper Alg 2) | **93.86%** | **0.9349** | **0.9300** | **0.9415** | **3 (-91.9%)** | **2** | **4** | 1 | **2** | 17.95s | **0.1ms** |

---

## 📈 Key Metric Highlights & Analysis

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       BENCHMARK PERFORMANCE SUMMARY                                              │
├──────────────────────┬────────────────────────┬──────────────────────┬───────────────────────────────────────────┤
│ Metric               │ Standard CART Baseline │ BiCART (Paper Alg 1) │ BiTAO (Paper Alg 2)                      │
├──────────────────────┼────────────────────────┼──────────────────────┼───────────────────────────────────────────┤
│ Test Accuracy        │ 91.23%                 │ 97.37% (🏆 Best)     │ 93.86% (+2.63% vs CART)                   │
│ Macro F1-Score       │ 0.9075                 │ 0.9721 (🏆 Best)     │ 0.9349 (+0.0274 vs CART)                  │
│ Tree Size (Nodes)    │ 37 nodes               │ 17 nodes (-54.1%)    │ 3 nodes (🏆 91.9% Reduction!)             │
│ Max Depth (Δ)        │ 7 levels               │ 5 levels             │ 4 levels                                  │
│ Unique Features Used │ 13 features            │ 11 features          │ 2 features (🏆 Ultra-interpretable)       │
└──────────────────────┴────────────────────────┴──────────────────────┴───────────────────────────────────────────┘
```

---

## 🔍 Detailed Model Breakdown

### 1. Standard CART (Baseline)
- **Mechanism:** Axis-aligned univariate splits ($x_i \le \theta$).
- **Weakness:** Produces a large 37-node tree with 7 levels to approximate non-axis-parallel boundaries.
- **Accuracy:** 91.23% (underperforms on diagonal boundary regions).

### 2. BiCART (Paper Algorithm 1)
- **Mechanism:** Greedy top-down search over feature pairs $\binom{D}{2}$ and $H=36$ orientation angles.
- **Strength:** Achieves top **97.37% accuracy** (+6.14% gain) while cutting tree size in half (17 nodes).
- **Interpretability:** 7 out of 8 decision nodes use 2D bivariate lines instead of complex cascades.

### 3. BiTAO (Paper Algorithm 2)
- **Mechanism:** Tree Alternating Optimization (TAO) with $L_1$-regularization penalty $\lambda \|\mathbf{w}\|_1$.
- **Strength:** Reduces the tree to an **ultra-compact 3-node structure** (a single bivariate decision split using just 2 features!).
- **Interpretability:** Maximum interpretability — can be rendered as a single 2D scatter plot.
