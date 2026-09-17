"""
bitao.py
--------
BiTAO: Bivariate Tree Alternating Optimization — implemented from scratch.

Paper Reference:
  Kairgeldin & Carreira-Perpiñán, KDD 2024.
  Section 4.1: "The better algorithm: bivariate TAO"

═══════════════════════════════════════════════════════════════════════════════
WHAT IS BiTAO?
═══════════════════════════════════════════════════════════════════════════════

Unlike BiCART (greedy, builds tree top-down), BiTAO is NON-GREEDY.
It takes an EXISTING tree and OPTIMIZES all its nodes iteratively.

The paper's objective function (Equation 1):

  E(Θ) = Σ_{n=1}^{N} L(y_n, T(x_n; Θ))  +  λ * Σ_{i ∈ Ndec} φ(w_i)

Where:
  - L(·,·) is the 0/1 loss (number of misclassifications)
  - T(x; Θ) is the tree's prediction for sample x
  - λ ≥ 0 is the regularization hyperparameter
  - φ(w_i) is the feature cost regularization (Equation 2):
      φ(w_i) = C         if ||w_i||_0 = 2  (bivariate node)
      φ(w_i) = ||w_i||_0 if ||w_i||_0 < 2  (univariate: 1, zero: 0)

So the total regularization cost per node is:
  - Zero-variate node  (0 features): cost = λ * 0 = 0
  - Univariate node    (1 feature):  cost = λ * 1
  - Bivariate node     (2 features): cost = λ * C

This means:
  - Zero-variate nodes are cheapest → they get "pruned" (node is redundant)
  - Bivariate nodes cost λC → only used when they reduce loss enough
  - The user controls tree complexity via λ (larger λ → simpler tree)
  - The user controls uni/bivariate ratio via C (larger C → more univariate)

═══════════════════════════════════════════════════════════════════════════════
THE REDUCED PROBLEM (RP) — Core of BiTAO
═══════════════════════════════════════════════════════════════════════════════

For each decision node i, define:
  R_i = reduced set = training samples that REACH node i.

The RP over a decision node (Equation 3) is:
  E_i(w_i, b_i) = Σ_{n ∈ R_i} L(ȳ_n, f_i(x_n; w_i, b_i)) + λ φ(w_i)

Where ȳ_n ∈ {left, right} is the PSEUDOLABEL of sample n:
  "Which child gives lower total loss for sample n?"

The RP is solved by comparing THREE candidate solutions:

  Solution 1 (Bivariate):    L_biv  = 0/1 loss + λ*C
  Solution 2 (Univariate):   L_univ = 0/1 loss + λ
  Solution 3 (Zero-variate): L_0    = 0/1 loss (no cost)

The best solution is:
  if L_biv + λC < min(L_univ + λ, L_0):   use bivariate
  if L_univ + λ < min(L_biv + λC, L_0):   use univariate
  if L_0 ≤ min(L_biv + λC, L_univ + λ):  use zero-variate (prune)

═══════════════════════════════════════════════════════════════════════════════
ALTERNATING OPTIMIZATION LOOP
═══════════════════════════════════════════════════════════════════════════════

From paper Figure 3 (pseudocode — exact transcription):

  repeat:
    for d = Δ downto 0:              ← reverse BFS order (deepest first)
      for i in nodes at depth d:     ← can be done in parallel
        if i is leaf:
          c_i ← majority class in R_i
        else:
          solve reduced problem (3) for node i
          choose best of bivariate / univariate / zero-variate
  until E(Θ) does not strictly decrease

  remove redundant nodes (zero-feature solution)
  return trained tree

═══════════════════════════════════════════════════════════════════════════════
FIDELITY STATEMENT (required by project rules)
═══════════════════════════════════════════════════════════════════════════════

PAPER COMPONENT          | OUR IMPLEMENTATION            | NOTES
─────────────────────────|───────────────────────────────|──────────────
Objective function E(Θ) | Exactly reproduced            | Eq. (1)
Regularization φ(w_i)  | Exactly reproduced            | Eq. (2)
Reduced problem (3)     | Exactly reproduced            | Eq. (3)
Bivariate solution      | H-orientation search          | Paper's approx.
Univariate solution     | Exact threshold search        | Exact
Zero-variate solution   | All-left vs all-right         | Exact
Solution selection rule | Exact (L_biv+λC vs L_univ+λ vs L_0) | Exact
TAO order               | Reverse BFS (deepest first)   | Exact
Leaf update             | Majority class of R_i         | Exact
Stopping condition      | E(Θ) not strictly decreasing  | Exact
Redundant node removal  | Remove zero-variate nodes     | Exact

SIMPLIFICATION:
  - The paper computes the FULL regularization path over λ = {0,1,...,N-N₁}.
    We use a FIXED λ specified by the user (or try a small grid).
    Reason: Full regularization path adds significant complexity.
    This is clearly documented here and in the report.

  - Starting tree: Paper uses "binary axis-aligned tree with given structure."
    We initialize with a CART tree of fixed depth, then run BiTAO on it.
    This is an implementation choice made for our reproduction.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from node import Node
from cart import gini_impurity, _majority_class, CARTClassifier
from bicart import generate_orientations, project_features


# ─────────────────────────────────────────────────────────────────────────────
# PSEUDOLABEL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_pseudolabels(node, X, y):
    """
    Compute pseudolabels ȳ_n for samples in the reduced set of a node.

    From the paper (Section 4.1):
    "ȳ_n ∈ {left, right} is a pseudolabel assigned to training instance
     x_n to indicate the child that yields a lower loss value. The loss
     is computed by propagating x_n down the corresponding child."

    For each sample n in the reduced set R_i:
      - Temporarily send n to LEFT → compute loss from LEFT subtree
      - Temporarily send n to RIGHT → compute loss from RIGHT subtree
      - ȳ_n = the child with lower loss

    In a binary 0/1-loss tree:
      Loss from sending sample n to LEFT child  = 0 if the leaf at the
      end of the LEFT path predicts correctly, else 1.
      Similarly for RIGHT child.

    Parameters
    ----------
    node : Node — current decision node
    X : np.ndarray of shape (|R_i|, n_features)
    y : np.ndarray of shape (|R_i|,)

    Returns
    -------
    pseudolabels : np.ndarray of shape (|R_i|,)
        0 = LEFT preferred, 1 = RIGHT preferred
    """
    n = len(y)
    pseudolabels = np.zeros(n, dtype=int)

    for idx in range(n):
        x_n = X[idx]
        y_n = y[idx]

        # Loss if sent to LEFT child
        pred_left = node.left.predict_sample(x_n)
        loss_left = int(pred_left != y_n)

        # Loss if sent to RIGHT child
        pred_right = node.right.predict_sample(x_n)
        loss_right = int(pred_right != y_n)

        # ȳ_n = child with lower loss (prefer left on tie)
        pseudolabels[idx] = 0 if loss_left <= loss_right else 1

    return pseudolabels


# ─────────────────────────────────────────────────────────────────────────────
# REDUCED PROBLEM: THREE SOLUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def solve_zero_variate(pseudolabels):
    """
    Solution 3 from paper: Zero-variate solution.

    From paper (Section 4.1.1):
    "Solution 3: L_0 (zero-variate solution) of eq. (3) s.t. ||w_i||_0 = 0,
     b_i ∈ {-1, +1}. This indicates that all samples in R_i are sent to
     the left (b_0_i = -1) or to the right (b_0_i = 1)."

    All samples go to the same child:
      - If all-left: loss = number of samples that prefer RIGHT
      - If all-right: loss = number of samples that prefer LEFT

    Parameters
    ----------
    pseudolabels : np.ndarray — 0=prefer left, 1=prefer right

    Returns
    -------
    L_0 : float — minimum 0/1 loss (no regularization for zero-variate)
    bias_0 : float — -1 (all-left) or +1 (all-right)
    """
    n = len(pseudolabels)
    n_prefer_left = np.sum(pseudolabels == 0)
    n_prefer_right = np.sum(pseudolabels == 1)

    if n_prefer_left >= n_prefer_right:
        return float(n_prefer_right), -1.0  # all-left is better
    else:
        return float(n_prefer_left), 1.0   # all-right is better


def solve_univariate(X, pseudolabels, lam):
    """
    Solution 2 from paper: Univariate solution (Vectorized).
    """
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return 0.0, 0, 0.0, 1.0, 0.0

    best_loss = float('inf')
    best_feature = 0
    best_threshold = 0.0

    total_ones = np.sum(pseudolabels)

    for j in range(n_features):
        values = X[:, j]
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_pl = pseudolabels[sorted_idx]

        # Vectorized loss computation
        cum_ones = np.cumsum(sorted_pl[:-1]) # left child errors
        right_zeros = (np.arange(n_samples - 1, 0, -1)) - (total_ones - cum_ones) # right child errors
        losses = (cum_ones + right_zeros).astype(float)

        valid_mask = sorted_values[1:] > sorted_values[:-1]
        if not np.any(valid_mask):
            continue

        losses[~valid_mask] = float('inf')
        min_idx = np.argmin(losses)
        min_l = losses[min_idx]

        if min_l < best_loss:
            best_loss = min_l
            best_feature = j
            best_threshold = (sorted_values[min_idx] + sorted_values[min_idx + 1]) / 2.0

    return float(best_loss), best_feature, best_threshold, 1.0, 0.0


def solve_bivariate(X, pseudolabels, lam, C, H=36):
    """
    Solution 1 from paper: Bivariate solution (Vectorized).
    """
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return 0.0, 0, 1, 1.0, 0.0, 0.0

    _, W = generate_orientations(H)

    best_loss = float('inf')
    best_j = 0
    best_k = 1
    best_cos = 1.0
    best_sin = 0.0
    best_threshold = 0.0

    total_ones = np.sum(pseudolabels)

    for j in range(n_features):
        for k in range(j + 1, n_features):
            X_pair = X[:, [j, k]] # (N, 2)
            Z = X_pair @ W.T       # (N, H)

            for h in range(H):
                z = Z[:, h]
                sorted_idx = np.argsort(z)
                sorted_z = z[sorted_idx]
                sorted_pl = pseudolabels[sorted_idx]

                cum_ones = np.cumsum(sorted_pl[:-1])
                right_zeros = (np.arange(n_samples - 1, 0, -1)) - (total_ones - cum_ones)
                losses = (cum_ones + right_zeros).astype(float)

                valid_mask = sorted_z[1:] > sorted_z[:-1]
                if not np.any(valid_mask):
                    continue

                losses[~valid_mask] = float('inf')
                min_idx = np.argmin(losses)
                min_l = losses[min_idx]

                if min_l < best_loss:
                    best_loss = min_l
                    best_j = j
                    best_k = k
                    best_cos = W[h, 0]
                    best_sin = W[h, 1]
                    best_threshold = (sorted_z[min_idx] + sorted_z[min_idx + 1]) / 2.0

    return float(best_loss), best_j, best_k, best_cos, best_sin, best_threshold


def solve_reduced_problem(X, pseudolabels, lam, C, H=36):
    """
    Solve the Reduced Problem (RP) over a single decision node.

    Implements the selection rule from paper (after equations 3 and 4):

      if L_biv + λC < min(L_univ + λ, L_0):   → θ_biv
      if L_univ + λ < min(L_biv + λC, L_0):   → θ_univ
      if L_0 ≤ min(L_biv + λC, L_univ + λ):  → θ_0

    Ties are broken in favor of the model with fewer parameters.

    Parameters
    ----------
    X : np.ndarray of shape (|R_i|, n_features)
    pseudolabels : np.ndarray
    lam : float — regularization λ
    C : float   — bivariate cost C
    H : int     — orientations

    Returns
    -------
    solution_type : str — "bivariate", "univariate", or "zero"
    params : dict — the optimal parameters for this node
    total_cost : float — regularized objective value
    """
    # Solution 3: Zero-variate
    L0, bias_0 = solve_zero_variate(pseudolabels)
    cost_0 = L0  # No regularization cost for zero-variate nodes

    # Solution 2: Univariate
    L_univ, feat_u, thresh_u, cos_u, sin_u = solve_univariate(
        X, pseudolabels, lam
    )
    cost_univ = L_univ + lam * 1  # λ * ||w||_0 = λ * 1

    # Solution 1: Bivariate
    if X.shape[1] >= 2:
        L_biv, feat_j, feat_k, cos_b, sin_b, thresh_b = solve_bivariate(
            X, pseudolabels, lam, C, H
        )
        cost_biv = L_biv + lam * C  # λ * C
    else:
        cost_biv = float('inf')
        L_biv = float('inf')
        feat_j = feat_k = cos_b = sin_b = thresh_b = None

    # ── Selection rule (from paper equations after eq. 3) ─────────────────
    # Ties broken in favor of simpler model (paper: "lower number of params")
    min_cost = min(cost_0, cost_univ, cost_biv)

    if cost_0 <= min_cost:
        # Zero-variate wins (simplest)
        return "zero", {
            'weight_j': 0.0, 'weight_k': 0.0, 'bias': bias_0,
            'feature_j': None, 'feature_k': None
        }, cost_0

    elif cost_univ <= min_cost:
        # Univariate wins
        return "univariate", {
            'feature_j': feat_u, 'feature_k': None,
            'weight_j': 1.0, 'weight_k': 0.0,
            'bias': -thresh_u if thresh_u is not None else 0.0
        }, cost_univ

    else:
        # Bivariate wins
        return "bivariate", {
            'feature_j': feat_j, 'feature_k': feat_k,
            'weight_j': cos_b, 'weight_k': sin_b,
            'bias': -thresh_b if thresh_b is not None else 0.0
        }, cost_biv


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE OBJECTIVE E(Θ)
# ─────────────────────────────────────────────────────────────────────────────

def compute_objective(tree_root, X, y, lam, C):
    """
    Compute the full objective E(Θ) = data_loss + λ * regularization.

    From paper, Equation (1):
      E(Θ) = Σ_{n=1}^{N} L(y_n, T(x_n; Θ)) + λ * Σ_{i∈Ndec} φ(w_i)

    Where L is the 0/1 loss (1 if misclassified, 0 if correct).

    Parameters
    ----------
    tree_root : Node
    X, y : training data
    lam, C : regularization parameters

    Returns
    -------
    float : total objective value
    """
    # Data loss term
    y_pred = np.array([tree_root.predict_sample(x) for x in X])
    data_loss = float(np.sum(y_pred != y))

    # Regularization term
    reg = _compute_regularization(tree_root, lam, C)

    return data_loss + reg


def _compute_regularization(node, lam, C):
    """Recursively sum regularization over all decision nodes."""
    if node.is_leaf:
        return 0.0
    if node.node_type == "bivariate":
        cost = lam * C
    elif node.node_type == "univariate":
        cost = lam * 1
    else:  # zero-variate
        cost = 0.0
    return cost + _compute_regularization(node.left, lam, C) + \
           _compute_regularization(node.right, lam, C)


# ─────────────────────────────────────────────────────────────────────────────
# REDUCED SET COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def get_reduced_sets(tree_root, X, y):
    """
    Compute reduced sets R_i for all nodes by routing all samples.

    From the paper:
    "Define the reduced set R_i ⊂ {1,...,N} as the training instances
     that reach the node i ∈ N."

    Returns
    -------
    dict mapping node_id → (X_node, y_node, indices)
    """
    # Assign integer IDs to nodes
    node_map = {}
    _assign_ids(tree_root, node_map, [0])

    # Collect samples per node by routing
    reduced = {integer_id: {'X': [], 'y': [], 'idx': []} for integer_id in node_map.values()}

    for i, (x, yi) in enumerate(zip(X, y)):
        node = tree_root
        while True:
            nid = node_map[id(node)]
            reduced[nid]['X'].append(x)
            reduced[nid]['y'].append(yi)
            reduced[nid]['idx'].append(i)
            if node.is_leaf:
                break
            direction = node.route(x)
            node = node.left if direction == "left" else node.right

    # Convert to arrays
    for nid in reduced:
        if reduced[nid]['X']:
            reduced[nid]['X'] = np.array(reduced[nid]['X'])
            reduced[nid]['y'] = np.array(reduced[nid]['y'])
        else:
            reduced[nid]['X'] = np.zeros((0, X.shape[1]))
            reduced[nid]['y'] = np.array([])

    return reduced, node_map


def _assign_ids(node, node_map, counter):
    """Assign unique integer IDs to all nodes."""
    node_map[id(node)] = counter[0]
    counter[0] += 1
    if not node.is_leaf:
        _assign_ids(node.left, node_map, counter)
        _assign_ids(node.right, node_map, counter)


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT NODES IN REVERSE BFS ORDER
# ─────────────────────────────────────────────────────────────────────────────

def get_nodes_by_depth(tree_root):
    """
    Collect all nodes organized by depth level (BFS order).

    Paper: "We optimize node parameters in the reverse breadth-first
    search order starting with deepest nodes and moving all the way
    to the root."

    Returns
    -------
    levels : list of lists — levels[d] contains nodes at depth d
    max_depth : int
    """
    from collections import deque
    levels = []
    queue = deque([(tree_root, 0)])

    while queue:
        node, d = queue.popleft()
        while len(levels) <= d:
            levels.append([])
        levels[d].append(node)
        if not node.is_leaf:
            queue.append((node.left, d + 1))
            queue.append((node.right, d + 1))

    return levels


# ─────────────────────────────────────────────────────────────────────────────
# REDUNDANT NODE REMOVAL
# ─────────────────────────────────────────────────────────────────────────────

def remove_redundant_nodes(node):
    """
    Remove zero-variate nodes from the tree.

    From the paper:
    "If a node uses no features (||w_i||_0 = 0) then it sends all points
     either right or left (depending on b_i), so it is redundant and can
     be pruned at the end of the algorithm."

    A zero-variate node sends all samples to ONE child. We replace it
    with that child.

    This is done AFTER the alternating optimization converges.

    Parameters
    ----------
    node : Node — root of subtree

    Returns
    -------
    Node — simplified subtree root
    """
    if node.is_leaf:
        return node

    # First recurse to simplify children
    node.left = remove_redundant_nodes(node.left)
    node.right = remove_redundant_nodes(node.right)

    # Now check if THIS node is zero-variate
    if node.node_type == "zero":
        if node.bias < 0:
            # All samples go LEFT — replace this node with its left child
            return node.left
        else:
            # All samples go RIGHT — replace this node with its right child
            return node.right

    return node


# ─────────────────────────────────────────────────────────────────────────────
# BiTAO CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class BiTAOClassifier:
    """
    Bivariate Tree Alternating Optimization (BiTAO) — built from scratch.

    This is Model 3 in our project.

    Algorithm overview (from paper Figure 3):
    ─────────────────────────────────────────────────────────────────
    INPUT:  Training set {x_n, y_n}, initial tree T, λ, C, H

    REPEAT:
      1. Compute reduced sets R_i for all nodes.
      2. For d = Δ downto 0 (deepest nodes first):
         For each node i at depth d:
           IF leaf:
             c_i ← majority class in R_i
           ELSE (decision node):
             Compute pseudolabels ȳ_n for n ∈ R_i
             Solve reduced problem (eq. 3):
               - L_biv: bivariate 0/1 loss (H orientations × all pairs)
               - L_univ: univariate 0/1 loss
               - L_0: zero-variate 0/1 loss
             Select: θ_i* = argmin over {θ_biv, θ_univ, θ_0} of cost
    UNTIL E(Θ) does not strictly decrease

    Remove redundant (zero-variate) nodes.
    Return trained tree.
    ─────────────────────────────────────────────────────────────────

    Parameters
    ----------
    max_depth : int, default=4
        Depth of the INITIAL tree (before BiTAO optimization).
        Implementation choice: Paper starts with "binary axis-aligned
        tree with given structure." We use a CART tree of this depth.
    lam : float, default=1.0
        Regularization parameter λ. Controls tree size.
        Larger λ → simpler tree (fewer nodes).
    C : float, default=1.5
        Bivariate cost. Paper recommends C ∈ [1.1, 1.5].
        C > 1: bivariate splits only used when they sufficiently reduce loss.
    H : int, default=36
        Number of orientation angles. Implementation choice.
    max_iter : int, default=20
        Maximum TAO iterations. Implementation choice.
    min_samples_split : int, default=2
        Minimum samples to keep a decision node.
    """

    def __init__(self, max_depth=4, lam=1.0, C=1.5, H=36,
                 max_iter=20, min_samples_split=2):
        self.max_depth = max_depth
        self.lam = lam
        self.C = C
        self.H = H
        self.max_iter = max_iter
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes_ = None
        self.objective_history = []

    def fit(self, X, y):
        """
        Train BiTAO on the given data.

        Step 1: Initialize tree using CART.
        Step 2: Run alternating optimization (BiTAO).
        Step 3: Remove redundant nodes.

        Parameters
        ----------
        X : np.ndarray of shape (n_train, n_features)
        y : np.ndarray of shape (n_train,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self.n_classes_ = len(np.unique(y))

        # ── Step 1: Initialize with CART tree ──────────────────────────────
        # Implementation choice: paper says "binary axis-aligned tree with
        # given structure." We use CART of fixed depth as initialization.
        print("  [BiTAO] Step 1: Initializing with CART tree "
              f"(depth={self.max_depth})...")
        cart = CARTClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
        cart.fit(X, y)
        self.root = cart.root

        # ── Step 2: Alternating optimization ──────────────────────────────
        print("  [BiTAO] Step 2: Running alternating optimization "
              f"(λ={self.lam}, C={self.C}, H={self.H})...")
        self.root = self._run_tao(X, y)

        # ── Step 3: Remove redundant nodes ─────────────────────────────────
        # Paper: "remove redundant nodes (empty features solution)"
        print("  [BiTAO] Step 3: Removing redundant nodes...")
        self.root = remove_redundant_nodes(self.root)

        return self

    def _run_tao(self, X, y):
        """
        Run the BiTAO alternating optimization loop.

        From paper (Figure 3 pseudocode — exact):
          repeat:
            for d = Δ downto 0:
              for i in nodes at depth d (can be done in parallel):
                if i ∈ Nleaf:
                  c_i ← majority class in R_i
                else:
                  [solve reduced problem]
          until E(Θ) does not strictly decrease
        """
        prev_obj = float('inf')

        for iteration in range(self.max_iter):
            # ── Compute reduced sets ──────────────────────────────────────
            reduced, node_map = get_reduced_sets(self.root, X, y)

            # ── Get nodes in BFS order ────────────────────────────────────
            levels = get_nodes_by_depth(self.root)
            max_depth = len(levels) - 1

            # ── Reverse BFS: deepest nodes first ──────────────────────────
            # Paper: "We optimize node parameters in the reverse BFS order
            # starting with deepest nodes and moving all the way to the root."
            for d in range(max_depth, -1, -1):
                for node in levels[d]:
                    nid = node_map[id(node)]
                    X_node = reduced[nid]['X']
                    y_node = reduced[nid]['y']

                    if len(y_node) == 0:
                        # Empty node — make leaf
                        node.is_leaf = True
                        node.node_type = "leaf"
                        node.label = 0
                        continue

                    if node.is_leaf:
                        # ── Leaf update (RP for leaf) ────────────────────
                        # Paper: "Exact solution: the majority class of
                        # the samples in R_i"
                        node.label = _majority_class(y_node)
                        node.n_samples = len(y_node)

                    else:
                        # ── Decision node update (RP) ─────────────────────
                        if len(y_node) < self.min_samples_split:
                            # Too few samples — convert to leaf
                            node.is_leaf = True
                            node.node_type = "leaf"
                            node.label = _majority_class(y_node)
                            node.n_samples = len(y_node)
                            continue

                        # Compute pseudolabels
                        pseudolabels = compute_pseudolabels(node, X_node, y_node)

                        # Solve RP: compare bivariate, univariate, zero
                        sol_type, params, cost = solve_reduced_problem(
                            X_node, pseudolabels,
                            self.lam, self.C, self.H
                        )

                        # Update node parameters
                        node.node_type = sol_type
                        node.feature_j = params['feature_j']
                        node.feature_k = params['feature_k']
                        node.weight_j = params['weight_j']
                        node.weight_k = params['weight_k']
                        node.bias = params['bias']
                        node.n_samples = len(y_node)

            # ── Check stopping condition ──────────────────────────────────
            # Paper: "until E(Θ) does not strictly decrease"
            curr_obj = compute_objective(self.root, X, y, self.lam, self.C)
            self.objective_history.append(curr_obj)

            print(f"  [BiTAO] Iteration {iteration+1:3d}: "
                  f"E(Θ) = {curr_obj:.2f}  "
                  f"(prev={prev_obj:.2f})")

            if curr_obj >= prev_obj:
                print(f"  [BiTAO] Converged: E(Θ) did not decrease.")
                break

            prev_obj = curr_obj

        return self.root

    def predict(self, X):
        """
        Predict class labels.

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
        """Print the optimized tree structure."""
        if self.root:
            self.root.print_tree()
