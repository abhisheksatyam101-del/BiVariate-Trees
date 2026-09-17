"""
node.py
-------
Defines the Node class used by CART, BiCART, and BiTAO.

Paper Reference:
  Kairgeldin & Carreira-Perpiñán, "Bivariate Decision Trees: Smaller,
  Interpretable, More Accurate", KDD 2024.

  From the paper (Section 4):
  "We use bivariate decision nodes where routing function makes hard
   decisions f_i(x; θ_i) = left_i if w_ij*x_j + w_ik*x_k + b_i < 0,
   otherwise right_i, and the learnable parameters are θ_i = {w_i, b_i},
   where ||w_i||_0 ≤ 2 ensures splits of no more than 2 features."
"""


class Node:
    """
    Represents a single node in a decision tree (CART, BiCART, or BiTAO).

    For a DECISION NODE, the split condition is:
        w_j * x_j + w_k * x_k + b < 0  → go LEFT
        otherwise                        → go RIGHT

    For a UNIVARIATE node: only one feature is used (w_k = 0).
    For a BIVARIATE node:  both features are used.
    For a ZERO-VARIATE node: no feature is used (node is redundant).
    For a LEAF node: outputs a class label.

    Node types (paper terminology):
        "univariate"  — one feature used  (||w_i||_0 = 1)
        "bivariate"   — two features used (||w_i||_0 = 2)
        "zero"        — no feature used   (||w_i||_0 = 0), redundant
        "leaf"        — output node

    Parameters
    ----------
    is_leaf : bool
        True if this node is a leaf.
    label : int or None
        Class label (only for leaf nodes). The majority class of samples
        reaching this leaf.
    feature_j : int or None
        Index of the first (or only) feature used for splitting.
    feature_k : int or None
        Index of the second feature (bivariate split only).
    weight_j : float
        Weight of feature_j (w_ij in the paper).
    weight_k : float
        Weight of feature_k (w_ik in the paper). Zero for univariate.
    bias : float
        Bias term b_i. The split is: w_j*x_j + w_k*x_k + b < 0 → LEFT.
    node_type : str
        One of "univariate", "bivariate", "zero", "leaf".
    left : Node or None
        Left child.
    right : Node or None
        Right child.
    depth : int
        Depth of this node in the tree (root = 0).
    n_samples : int
        Number of training samples that reached this node.
    """

    def __init__(self):
        self.is_leaf = False
        self.label = None            # Majority class (leaf nodes)

        # Split parameters (decision nodes)
        self.feature_j = None        # First feature index
        self.feature_k = None        # Second feature index (bivariate)
        self.weight_j = 0.0          # w_ij
        self.weight_k = 0.0          # w_ik
        self.bias = 0.0              # b_i

        # Paper terminology: node type
        self.node_type = "leaf"      # "univariate", "bivariate", "zero", "leaf"

        # Tree structure
        self.left = None
        self.right = None
        self.depth = 0
        self.n_samples = 0

    def route(self, x):
        """
        Route a single sample x through this decision node.

        Implements the paper's routing function:
            f_i(x; θ_i) = LEFT  if w_ij*x_j + w_ik*x_k + b_i < 0
                         = RIGHT otherwise

        For a ZERO-VARIATE node (w_i = 0):
            bias = -1 → all samples go LEFT
            bias = +1 → all samples go RIGHT

        Parameters
        ----------
        x : array-like of shape (n_features,)

        Returns
        -------
        "left" or "right"
        """
        if self.is_leaf:
            raise ValueError("Cannot route through a leaf node.")

        val = self.bias
        if self.feature_j is not None:
            val += self.weight_j * x[self.feature_j]
        if self.feature_k is not None:
            val += self.weight_k * x[self.feature_k]

        return "left" if val < 0 else "right"

    def predict_sample(self, x):
        """
        Traverse the subtree rooted at this node to get a prediction for x.

        Parameters
        ----------
        x : array-like of shape (n_features,)

        Returns
        -------
        int : predicted class label
        """
        if self.is_leaf:
            return self.label

        direction = self.route(x)
        if direction == "left":
            return self.left.predict_sample(x)
        else:
            return self.right.predict_sample(x)

    def count_nodes(self):
        """Return (total_nodes, decision_nodes, leaf_nodes)."""
        if self.is_leaf:
            return 1, 0, 1
        lt, ld, ll = self.left.count_nodes()
        rt, rd, rl = self.right.count_nodes()
        return lt + rt + 1, ld + rd + 1, ll + rl

    def max_depth(self):
        """Return the maximum depth of the subtree rooted here."""
        if self.is_leaf:
            return self.depth
        return max(self.left.max_depth(), self.right.max_depth())

    def count_bivariate(self):
        """Count bivariate and univariate decision nodes."""
        if self.is_leaf:
            return 0, 0
        biv = 1 if self.node_type == "bivariate" else 0
        univ = 1 if self.node_type == "univariate" else 0
        lb, lu = self.left.count_bivariate()
        rb, ru = self.right.count_bivariate()
        return biv + lb + rb, univ + lu + ru

    def print_tree(self, indent=0, prefix="Root"):
        """Pretty-print the tree structure."""
        pad = "  " * indent
        if self.is_leaf:
            print(f"{pad}{prefix} [LEAF] → class={self.label}  (n={self.n_samples})")
        else:
            if self.node_type == "bivariate":
                cond = (f"w{self.weight_j:.2f}*x{self.feature_j} + "
                        f"w{self.weight_k:.2f}*x{self.feature_k} + "
                        f"b{self.bias:.2f} < 0")
            elif self.node_type == "univariate":
                cond = f"w{self.weight_j:.2f}*x{self.feature_j} + b{self.bias:.2f} < 0"
            else:
                cond = f"ZERO-VARIATE (b={self.bias:.1f})"
            print(f"{pad}{prefix} [{self.node_type.upper()}] "
                  f"{cond}  (n={self.n_samples})")
            self.left.print_tree(indent + 1, "LEFT")
            self.right.print_tree(indent + 1, "RIGHT")
