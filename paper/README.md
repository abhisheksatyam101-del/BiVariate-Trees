# Research Paper Reference

**Title:** Bivariate Decision Trees: Smaller, Interpretable, More Accurate  
**Authors:** Rasul Kairgeldin and Miguel Á. Carreira-Perpiñán  
**Institution:** University of California, Merced  
**Conference:** ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2024)  
**DOI:** [https://doi.org/10.1145/3637528.3671903](https://doi.org/10.1145/3637528.3671903)  

---

## 📄 Included Files

- [`Bivariate_Decision_Trees_KDD2024.pdf`](./Bivariate_Decision_Trees_KDD2024.pdf) — Original 12-page base research paper PDF published at KDD '24.

---

## 📌 Paper Abstract

Univariate decision trees, commonly used since the 1950s, predict by asking questions about a single feature in each decision node. While they are interpretable, they often lack competitive predictive accuracy due to their inability to model feature correlations. Multivariate (oblique) trees use multiple features in each node, capturing high-dimensional correlations better, but sometimes they can be difficult to interpret. We advocate for a model that strikes a useful middle ground: bivariate decision trees, which use two features in each node. This typically produces trees that not only are more accurate than univariate trees, but much smaller, which offsets the small increase in node complexity and keeps them interpretable. They also help data mining by constructing new features that are useful for discrimination, and by providing a form of supervised, hierarchical 2D visualization that reveals patterns such as clusters or linear structure. We give two new algorithms to learn bivariate trees: a fast one based on CART (BiCART); and a slower one based on alternating optimization with a feature regularization term (BiTAO), which produces the best trees while still scaling to large datasets.
