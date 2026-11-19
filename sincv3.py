import numpy as np
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm
import itertools
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, log_loss
from copy import deepcopy
import seaborn as sns
import igraph as ig
import leidenalg as la
import time
import math
from collections import defaultdict


# ---------------------------------------------------------
# Phase 1: The Map (Interaction Network Construction)
# --------------------------------------------------------


# ---------------------------------------------------------
# Abstract Base Model Wrapper
# ---------------------------------------------------------

class ModelWrapper(ABC):
    """
    Abstract Base Class to standardize model predictions for SINC.
    """
    
    def __init__(self, target_index=1):
        """
        :param target_index: For classification, which class probability to return.
                             (Default 1 for binary classification positive class).
                             Ignored for single-output regression.
        """
        self.target_index = target_index
        self.device = "cpu"

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Input: 2D Numpy Array (Batch Size, Features)
        Output: 1D Numpy Array (Batch Size, ) - Scalar predictions
        """
        pass

    def to_device(self, device_id):
        """
        Moves the model to a specific device (e.g., 'cuda:0').
        Crucial for Multi-GPU worker distribution.
        """
        self.device = device_id
        return self

# ---------------------------------------------------------
# Scikit-Learn / XGBoost / LightGBM Wrapper
# ---------------------------------------------------------
class SklearnWrapper(ModelWrapper):
    def __init__(self, model, model_type='regression', target_index=1):
        super().__init__(target_index)
        self.model = model
        self.model_type = model_type

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.model_type == 'classification':
            
            preds = self.model.predict_proba(X)
            return preds[:, self.target_index]
        else:
           
            return self.model.predict(X)

# ---------------------------------------------------------
# PyTorch Wrapper
# ---------------------------------------------------------
class PyTorchWrapper(ModelWrapper):
    def __init__(self, model, target_index=0, is_classification=False):
        super().__init__(target_index)
        if torch is None:
            raise ImportError("PyTorch is not installed.")
        
        self.model = model
        self.is_classification = is_classification
        self.model.eval() 

    def to_device(self, device_id):
        """
        Actual implementation of moving the internal model to GPU.
        """
        self.device = device_id
        if torch.cuda.is_available() and 'cuda' in str(device_id):
            self.model = self.model.to(device_id)
        else:
            self.model = self.model.cpu()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 1. Convert Numpy -> Tensor
        tensor_X = torch.from_numpy(X).float()
        
        # 2. Move to correct device (handled by to_device)
        if 'cuda' in str(self.device):
            tensor_X = tensor_X.to(self.device)

        # 3. Inference (No Grad for speed/memory)
        with torch.no_grad():
            output = self.model(tensor_X)

            # 4. Handle Output Shape
        
            if output.ndim > 1 and output.shape[1] > 1:

                output = output[:, self.target_index]
            
            
            if self.is_classification:
                output = torch.sigmoid(output) 

        # 5. Return to CPU Numpy
        return output.cpu().numpy().flatten()
    
# ---------------------------------------------------------
# Marginal Partial Dependence Computation
# ---------------------------------------------------------

def get_background_data(X, subsample_size=100):
    """Creates a consistent reference set for all PD calculations."""
    N = X.shape[0]
    if N > subsample_size:
        idx = np.random.choice(N, subsample_size, replace=False)
        return X[idx]
    return X

def compute_marginal_pds(model_wrapper, X_background, feature_indices=None, grid_resolution=50):
    """
    Step 1: Pre-computes centered PDs using the consistent background set.
    """
    N, M = X_background.shape
    feature_indices = feature_indices if feature_indices is not None else range(M)
    pds = {}
    
    for f_idx in tqdm(feature_indices, desc="Computing Marginals"):
        # 1. Create Grid based on percentiles of the background (or full X if preferred)
        unique_vals = np.unique(X_background[:, f_idx])
        if len(unique_vals) <= grid_resolution:
            grid = unique_vals
        else:
            percentiles = np.linspace(0, 100, num=grid_resolution)
            grid = np.unique(np.percentile(X_background[:, f_idx], percentiles))

        # 2. Calculate PD values
        pd_values = []
        for val in grid:
            X_temp = X_background.copy()
            X_temp[:, f_idx] = val
            preds = model_wrapper.predict(X_temp)
            pd_values.append(np.mean(preds))
            
        pd_values = np.array(pd_values)
        # 3. CENTER the PD (Subtract the mean of the PD values)
        pd_centered = pd_values - np.mean(pd_values)

        pds[f_idx] = {'grid': grid, 'values': pd_centered}
        
    return pds

# ---------------------------------------------------------
# Lookup for Partial Dependence Values
# ---------------------------------------------------------

def get_pd_for_X(X_column, pd_dict_item):
    """
    Interpolates PD values for the actual data points.
    
    Args:
        X_column (np.array): The actual feature values from the dataset (N,)
        pd_dict_item (dict): The dictionary created above {'grid':..., 'values':...}
    
    Returns:
        np.array: The PD effect for every row in X (N,)
    """
    return np.interp(X_column, pd_dict_item['grid'], pd_dict_item['values'])

# ---------------------------------------------------------
# H-statistic for Pairwise Interactions
# ---------------------------------------------------------

def compute_pairwise_interactions(model_wrapper, X_background, pds, feature_indices=None, batch_size=2048):
    """
    Step 2: Computes Friedman's H-statistic for pairs using the same background.
    """
    N_bg, M = X_background.shape
    feature_indices = feature_indices if feature_indices is not None else list(pds.keys())
    pairs = list(itertools.combinations(feature_indices, 2))
    interaction_scores = {}

    for (idx_j, idx_k) in tqdm(pairs, desc="Computing Interactions"):
        grid_j, grid_k = pds[idx_j]['grid'], pds[idx_k]['grid']
        
        # 1. Create 50x50 Meshgrid (2,500 points)
        mesh_j, mesh_k = np.meshgrid(grid_j, grid_k, indexing='ij')
        flat_j, flat_k = mesh_j.flatten(), mesh_k.flatten()
        n_grid_points = len(flat_j)

        # 2. Construct Evaluation Matrix (n_grid_points * subsample_size)
        X_eval = np.tile(X_background, (n_grid_points, 1))
        X_eval[:, idx_j] = np.repeat(flat_j, N_bg)
        X_eval[:, idx_k] = np.repeat(flat_k, N_bg)

        # 3. Batch Prediction
        all_preds = []
        for start in range(0, X_eval.shape[0], batch_size):
            all_preds.append(model_wrapper.predict(X_eval[start : start + batch_size]))
        
        # 4. Compute Joint PD and Center it
        pd_joint = np.mean(np.concatenate(all_preds).reshape(n_grid_points, N_bg), axis=1)
        pd_joint_centered = pd_joint - np.mean(pd_joint)

        # 5. Interpolate Marginals (Look up the 1D values for our 2D grid)
        pd_j_vals = np.interp(flat_j, pds[idx_j]['grid'], pds[idx_j]['values'])
        pd_k_vals = np.interp(flat_k, pds[idx_k]['grid'], pds[idx_k]['values'])

        # 6. H-Statistic Calculation
        # H^2 = sum( [PD_jk - (PD_j + PD_k)]^2 ) / sum( PD_jk^2 )
        numer = np.sum((pd_joint_centered - (pd_j_vals + pd_k_vals))**2)
        denom = np.sum(pd_joint_centered**2)
        
        interaction_scores[(idx_j, idx_k)] = np.sqrt(numer / denom) if denom > 1e-9 else 0.0

    return interaction_scores

# ---------------------------------------------------------
# Total Interaction Index (alternative to the H-statistic)
# ---------------------------------------------------------

def compute_total_interactions(model_wrapper, X, n_samples=2000, feature_indices=None,
                               batch_size=8192, seed=0):
    """
    Total interaction index TII_ij (Fruth, Roustant & Kuhnt, 2014) for every feature pair,
    normalised by the output variance.

    TII_ij is the variance of ALL functional-ANOVA terms that contain both x_i and x_j,
    at any order. Unlike the H-statistic, which compares partial dependence functions
    (averaging out every other feature first), it cannot miss a higher-order interaction:
    every pair inside a k-way term gets TII > 0, so the term appears as a k-clique.

    Pick-freeze estimator with two independent samples A and B from X:
        TII_ij = 1/4 * E[(f(A) - f(A^i) - f(A^j) + f(A^ij))^2]
    where A^S is A with the columns in S replaced by those of B.

    Cost: n_samples * (1 + F + F(F-1)/2) predictions, with F = number of features.

    Returns:
        dict {(i, j): TII_ij / Var(f)}, i.e. the share of output variance carried by
        interactions involving both i and j. Already on the "squared" scale, so pass
        square=False to build_interaction_network.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    n = min(n_samples, len(X) // 2)
    idx = rng.permutation(len(X))
    A, B = X[idx[:n]], X[idx[n:2 * n]]      # two disjoint, independent samples
    feature_indices = list(feature_indices) if feature_indices is not None else list(range(X.shape[1]))

    def predict(M):
        return np.concatenate([model_wrapper.predict(M[s:s + batch_size])
                               for s in range(0, len(M), batch_size)])

    f_A = predict(A)
    var_f = np.var(f_A)
    if var_f < 1e-12:
        return {pair: 0.0 for pair in itertools.combinations(feature_indices, 2)}

    # f with a single feature resampled (reused for every pair containing it)
    f_single = {}
    for i in tqdm(feature_indices, desc="TII: single resamples"):
        A_i = A.copy()
        A_i[:, i] = B[:, i]
        f_single[i] = predict(A_i)

    scores = {}
    for i, j in tqdm(list(itertools.combinations(feature_indices, 2)), desc="TII: pairs"):
        A_ij = A.copy()
        A_ij[:, [i, j]] = B[:, [i, j]]
        tii = 0.25 * np.mean((f_A - f_single[i] - f_single[j] + predict(A_ij)) ** 2)
        scores[(i, j)] = tii / var_f
    return scores

# ---------------------------------------------------------
# Process Interaction Scores into DataFrame and Matrix
# ---------------------------------------------------------

def build_interaction_network(interaction_scores, feature_names, tau=0.0, square=True):
    """
    Phase 1: The Map (Interaction Network Construction)
    Populates the symmetric adjacency matrix W based on H-statistics or TII.

    Args:
        interaction_scores (dict): {(i, j): score}. From compute_pairwise_interactions these
                                   are H (sqrt), so they are squared to H^2 (square=True).
                                   From compute_total_interactions they are already on the
                                   variance scale (square=False).
        feature_names (list): List of feature names.
        tau (float): Noise threshold. Edges with weight < tau are set to 0.
        square (bool): Square the scores before thresholding. Default True (H-statistic).

    Returns:
        pd.DataFrame: The Weight Matrix W (D x D).
    """
    D = len(feature_names)
    W = pd.DataFrame(0.0, index=feature_names, columns=feature_names)

    print(f"Constructing Network Map with threshold tau={tau}...")

    for (i, j), h_value in interaction_scores.items():

        h_squared = h_value ** 2 if square else h_value

        name_i = feature_names[i]
        name_j = feature_names[j]

        if h_squared < tau:
            weight = 0.0
        else:
            weight = h_squared

        # Populate Symmetric Matrix
        W.at[name_i, name_j] = weight
        W.at[name_j, name_i] = weight

    return W


def plot_interaction_heatmap(W, feature_names, title='Pairwise Interaction Network (H-Statistics)', cmap='hot'):
    """
    Plots a raw heatmap of the feature-to-feature interaction matrix (W).
    
    Args:
        W (pd.DataFrame or np.ndarray): The symmetric weight matrix from the pipeline.
        feature_names (list): List of feature names for axis labeling.
        title (str): Title of the plot.
        cmap (str): Colormap to use (e.g., 'hot', 'viridis', 'magma').
    """

    plt.figure(figsize=(10, 8))
    
    # We use W.values if it's a DataFrame, otherwise assume it's an array
    matrix_data = W.values if hasattr(W, 'values') else W
    
    # Plotting using imshow for that classic "hot" look
    img = plt.imshow(matrix_data, cmap=cmap, aspect='auto')
    plt.colorbar(img, label='Interaction Strength ($H^2$)')
    
    # Labels and Ticks
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title(title)
    
    # Map the ticks to the feature names
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), feature_names)
    
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------
# Phase 2: The Discovery (Community Detection)
#---------------------------------------------------------

class InteractionGraph:
    """
    Phase 2: The Discovery (Community Detection)
    Converts the Interaction Matrix W into a Graph and finds clusters.
    Supports Louvain, Leiden, Modularity, and CPM with iterative refinement.
    """
    
    def __init__(self, adjacency_matrix_df):
        self.matrix = adjacency_matrix_df
        self.G = self._build_graph()
        self.communities = []
        
    def _build_graph(self):
        G = nx.from_pandas_adjacency(self.matrix)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    def _get_igraph(self):
        """Helper to convert NetworkX graph to igraph for Leiden/CPM."""
        import igraph as ig
        edges = [tuple(e) for e in self.G.edges()]
        weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        
        g_ig = ig.Graph()
        g_ig.add_vertices(list(self.G.nodes()))
        g_ig.add_edges(edges)
        g_ig.es['weight'] = weights
        return g_ig

    def detect_communities(self, algorithm='leiden', method='cpm', resolution=1.0, seed=42,
                           n_iterations=-1, alpha=0.2, min_weight=0.0, max_cliques=1000):
        """
        Detects communities using various algorithms and objective functions.

        :param algorithm: 'louvain', 'leiden', 'leiden_overlap' or 'cliques'.
                          'leiden_overlap' runs Leiden on the line graph of the interaction
                          network (i.e. it partitions the interactions/edges), so a feature
                          can belong to several communities.
                          'cliques' returns all maximal cliques of the thresholded interaction
                          network (Bron-Kerbosch): groups in which EVERY pair of features
                          interacts. Cliques may overlap. Ignores method/resolution/alpha.
        :param method: 'modularity' or 'cpm'
        :param resolution: Resolution parameter (>0). NOTE: for 'leiden_overlap' the line-graph
                           weights are on a different scale than H^2, so gamma must be re-tuned.
        :param seed: Random seed.
        :param n_iterations: (Leiden only) Number of iterations. -1 runs until stable.
        :param alpha: ('leiden_overlap' only) A feature joins a community if at least this share
                      of its total interaction strength lies in that community. It always
                      joins its strongest community.
        :param min_weight: ('cliques' only) Extra edge threshold w0 on top of tau. With the
                           default 0, every edge that survived tau is used.
        :param max_cliques: ('cliques' only) Safety limit; raises an error if the graph has
                            more maximal cliques than this (raise tau if it triggers).
        """
        # --- OPTION 4: MAXIMAL CLIQUES ---
        if algorithm.lower() == 'cliques':
            raw_communities = self._maximal_cliques(min_weight, max_cliques)
            self.communities = sorted([sorted(c) for c in raw_communities],
                                      key=lambda c: (-len(c), c))
            return self.communities

        partition_map = {
            'modularity': la.RBConfigurationVertexPartition,
            'cpm': la.CPMVertexPartition
        }

        # --- OPTION 3: OVERLAPPING LEIDEN (line graph) ---
        if algorithm.lower() == 'leiden_overlap':
            if method.lower() not in partition_map:
                raise ValueError(f"Method '{method}' not supported. Use 'modularity' or 'cpm'.")
            raw_communities = self._overlapping_leiden(
                partition_map[method.lower()], resolution, seed, n_iterations, alpha
            )
            self.communities = sorted([sorted(c) for c in raw_communities],
                                      key=lambda c: (-len(c), c))
            return self.communities

        # --- OPTION 1: LOUVAIN (via NetworkX) ---
        if algorithm.lower() == 'louvain':
            # NetworkX Louvain is simpler but lacks the refinement phase of Leiden
            louvain_sets = nx.community.louvain_communities(
                self.G, weight='weight', resolution=resolution, seed=seed
            )
            raw_communities = [list(c) for c in louvain_sets]

        # --- OPTION 2: LEIDEN (via leidenalg) ---
        elif algorithm.lower() == 'leiden':
            g_ig = self._get_igraph()
            
            # Map method to the specific partition class
            # Note: RBConfigurationVertexPartition is used for Modularity to support resolution_parameter
            partition_map = {
                'modularity': la.RBConfigurationVertexPartition,
                'cpm': la.CPMVertexPartition
            }
            
            if method.lower() not in partition_map:
                raise ValueError(f"Method '{method}' not supported. Use 'modularity' or 'cpm'.")

            # Leiden includes a refinement step in each iteration
            # This is significantly more robust than Louvain for finding global optima
            partition = la.find_partition(
                g_ig, 
                partition_map[method.lower()], 
                weights='weight', 
                resolution_parameter=resolution,
                seed=seed,
                n_iterations=n_iterations
            )
            
            node_names = list(self.G.nodes())
            raw_communities = [[node_names[idx] for idx in comm] for comm in partition]

        else:
            raise ValueError("Algorithm must be 'louvain', 'leiden', 'leiden_overlap' or 'cliques'")

        # Post-process: sort by size and store
        self.communities = sorted([sorted(c) for c in raw_communities], key=len, reverse=True)
        return self.communities

    def _maximal_cliques(self, min_weight=0.0, max_cliques=1000):
        """
        Maximal weighted cliques (Bron & Kerbosch, 1973; as in Amolochitis et al., 2012).

        A maximal clique of minimum weight w0 is a set of features in which every pair is
        joined by an edge of weight >= w0, and which cannot be extended by another feature
        without breaking that property. A k-way interaction term produces exactly a
        k-clique in the pairwise H^2 network, and a feature can belong to several cliques,
        so overlapping interaction groups are recovered without extra parameters.

        Features without any edge >= w0 are returned as singletons (isolated nodes are
        maximal cliques of size 1).
        """
        # 1. Keep only the interactions strong enough to count (weight > 0 already means
        #    the edge survived tau when the matrix was built)
        G = nx.Graph()
        G.add_nodes_from(self.G.nodes())
        G.add_edges_from((u, v) for u, v, w in self.G.edges(data='weight')
                         if w > 0 and w >= min_weight)

        # 2. Enumerate all maximal cliques (networkx uses Bron-Kerbosch with pivoting).
        #    The number of maximal cliques can grow exponentially in dense graphs, so stop
        #    early instead of hanging if tau is set too low.
        cliques = []
        for clique in nx.find_cliques(G):
            cliques.append(list(clique))
            if len(cliques) > max_cliques:
                raise RuntimeError(
                    f"More than {max_cliques} maximal cliques: the interaction network is too "
                    f"dense. Increase tau (or min_weight), or raise max_cliques."
                )
        return cliques

    def get_clique_weights(self, agg=min):
        """
        Interaction strength of each current community: `agg` (default min) over the H^2
        weights of all pairs inside it. With min, this is the largest w0 at which the group
        would still be a clique. Singletons get 0.
        """
        weights = []
        for comm in self.communities:
            pairs = list(itertools.combinations(comm, 2))
            weights.append(agg(self.matrix.at[u, v] for u, v in pairs) if pairs else 0.0)
        return weights

    def _overlapping_leiden(self, partition_class, resolution, seed, n_iterations, alpha):
        """
        Overlapping communities via Leiden on the weighted line graph
        (Evans & Lambiotte, 2009). Every interaction (edge) is assigned to exactly one
        community; features inherit membership from their interactions.
        """
        nodes = list(self.G.nodes())

        # Interactions = edges that survived the tau threshold
        edges = [(u, v, d.get('weight', 1.0)) for u, v, d in self.G.edges(data=True)
                 if d.get('weight', 1.0) > 0]
        if not edges:
            return [[n] for n in nodes]

        strength = defaultdict(float)
        incident = defaultdict(list)
        for k, (u, v, w) in enumerate(edges):
            strength[u] += w
            strength[v] += w
            incident[u].append(k)
            incident[v].append(k)

        # Line graph: two interactions are linked if they share a feature n.
        # Weight w_a * w_b / s_n down-weights links routed through high-strength hub features.
        lg_weights = {}
        for n, ks in incident.items():
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    a, b = ks[i], ks[j]
                    lg_weights[(a, b)] = edges[a][2] * edges[b][2] / strength[n]

        g = ig.Graph(n=len(edges), edges=list(lg_weights.keys()))
        g.es['weight'] = list(lg_weights.values())
        partition = la.find_partition(
            g, partition_class,
            weights='weight',
            resolution_parameter=resolution,
            seed=seed,
            n_iterations=n_iterations
        )

        # share[node][c] = interaction strength of `node` that lies in edge-community c
        share = defaultdict(lambda: defaultdict(float))
        for c, comm in enumerate(partition):
            for k in comm:
                u, v, w = edges[k]
                share[u][c] += w
                share[v][c] += w

        # Membership: strongest community always, others if their share >= alpha
        members = defaultdict(set)
        for n, per_c in share.items():
            best = max(per_c, key=per_c.get)
            for c, s in per_c.items():
                if c == best or s / strength[n] >= alpha:
                    members[c].add(n)

        communities = {frozenset(m) for m in members.values() if m}

        # Features with no surviving interactions become singletons (e.g. noise features)
        covered = set().union(*communities) if communities else set()
        communities |= {frozenset([n]) for n in nodes if n not in covered}

        return [list(c) for c in communities]

    def get_memberships(self):
        """Returns {feature: [community indices]} for the current communities."""
        memberships = defaultdict(list)
        for i, comm in enumerate(self.communities):
            for node in comm:
                memberships[node].append(i)
        return dict(memberships)

    def get_overlapping_features(self):
        """Returns the features that belong to more than one community."""
        return sorted(n for n, cs in self.get_memberships().items() if len(cs) > 1)

    def plot_graph(self, title="Feature Interaction Network"):
        """
        Visualizes the graph with:
        - Node colors by community
        - Edge thickness and transparency based on interaction strength (H^2)
        """
        if not self.communities:
            print("Please run detect_communities() first.")
            return
             
        fig, ax = plt.subplots(figsize=(12, 10))

        # 1. Map nodes to community IDs for coloring.
        # With overlapping communities a node can have several; it is drawn as a pie.
        memberships = self.get_memberships()
        cmap = plt.cm.tab10

        # 2. Position nodes using Spring Layout
        # weight='weight' tells the algorithm to pull strongly connected nodes closer
        pos = nx.spring_layout(self.G, weight='weight', k=0.5, iterations=100, seed=42)
        
        # 3. Extract edge weights for visual mapping
        edges = self.G.edges(data=True)
        weights = np.array([d['weight'] for u, v, d in edges])
        
        # Normalize weights for thickness (e.g., scale between 0.5 and 8.0)
        if len(weights) > 0:
            # We use a linear scaling, but you could use log/exp if weights vary wildly
            max_w = weights.max()
            min_w = weights.min()
            widths = 1 + 7 * (weights - min_w) / (max_w - min_w + 1e-9)
            # Map alpha (opacity) between 0.1 and 0.8
            alphas = 0.1 + 0.7 * (weights - min_w) / (max_w - min_w + 1e-9)
        else:
            widths = 1.0
            alphas = 0.3

        # 4. Draw Edges first (individually to apply per-edge alpha) so nodes sit on top
        for i, (u, v, d) in enumerate(edges):
            nx.draw_networkx_edges(
                self.G, pos, ax=ax,
                edgelist=[(u, v)],
                width=widths[i] if np.ndim(widths) else widths,
                alpha=alphas[i] if np.ndim(alphas) else alphas,
                edge_color="gray"
            )

        # 5. Draw Nodes as pies: one wedge per community the feature belongs to
        from matplotlib.patches import Wedge, Patch
        radius = 0.06
        for node in self.G.nodes():
            comms = memberships.get(node, [])
            x, y = pos[node]
            if not comms:
                ax.add_patch(plt.Circle((x, y), radius, color="lightgray", zorder=10))
                continue
            step = 360.0 / len(comms)
            for k, c in enumerate(comms):
                ax.add_patch(Wedge((x, y), radius, 90 + k * step, 90 + (k + 1) * step,
                                   facecolor=cmap(c % 10), edgecolor="white",
                                   linewidth=1.5, alpha=1.0, zorder=10))
            if len(comms) > 1:  # Highlight shared features with a dark ring
                ax.add_patch(plt.Circle((x, y), radius, fill=False, edgecolor="black",
                                        linewidth=2, zorder=11))

        # 6. Draw Labels (offset below the node so they do not hide the pie)
        label_pos = {n: (p[0], p[1] - radius * 1.6) for n, p in pos.items()}
        nx.draw_networkx_labels(self.G, label_pos, ax=ax, font_size=9, font_weight="bold")

        # 7. Legend: one entry per multi-feature community
        handles = [Patch(color=cmap(i % 10), label=f"Grp {i}: " + ", ".join(comm))
                   for i, comm in enumerate(self.communities) if len(comm) > 1]
        if handles:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                      fontsize=8, frameon=False)

        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.margins(0.1)
        plt.title(title, fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

#---------------------------------------------------------
# Phase 3: The Validation (Synergy Scoring)
#---------------------------------------------------------

def community_shapley_values(wrapped_model, X_val_df, y_val, communities, metric_func,
                             n_repeats=5, max_exact_groups=10, n_samples=200, seed=None):
    """
    Shapley values over communities, treating each community as a player.

    The value of a coalition S of communities is the increase in loss when the UNION of
    their features is permuted jointly. Because the union is used, a feature shared by
    several communities is never permuted twice, and the values satisfy efficiency:
    sum_j phi_j = v(all communities). Percentages derived from them therefore stay
    meaningful when communities overlap.

    Exact enumeration (2^q coalitions) is used for q <= max_exact_groups, otherwise
    Monte Carlo sampling of n_samples random orderings.

    The same n_repeats row permutations are reused for every coalition (common random
    numbers), which strongly reduces the variance of the marginal contributions.
    """
    rng = np.random.default_rng(seed)
    q = len(communities)
    X_base = X_val_df.values
    col_idx = {c: i for i, c in enumerate(X_val_df.columns)}
    community_cols = [[col_idx[f] for f in comm] for comm in communities]
    perms = [rng.permutation(len(X_base)) for _ in range(n_repeats)]

    baseline = metric_func(y_val, wrapped_model.predict(X_base))
    cache = {frozenset(): 0.0}

    def value(coalition):
        coalition = frozenset(coalition)
        if coalition not in cache:
            cols = sorted({c for j in coalition for c in community_cols[j]})
            scores = []
            for p in perms:
                X_temp = X_base.copy()
                X_temp[:, cols] = X_base[p][:, cols]
                scores.append(metric_func(y_val, wrapped_model.predict(X_temp)))
            cache[coalition] = np.mean(scores) - baseline
        return cache[coalition]

    phi = np.zeros(q)
    players = list(range(q))

    if q <= max_exact_groups:
        fact = [math.factorial(k) for k in range(q + 1)]
        for i in tqdm(players, desc="Shapley (exact)"):
            others = [j for j in players if j != i]
            for size in range(q):
                w = fact[size] * fact[q - size - 1] / fact[q]
                for S in itertools.combinations(others, size):
                    phi[i] += w * (value(S + (i,)) - value(S))
    else:
        for _ in tqdm(range(n_samples), desc="Shapley (sampled)"):
            order = rng.permutation(q)
            S, v_prev = [], 0.0
            for i in order:
                S.append(i)
                v_new = value(S)
                phi[i] += v_new - v_prev
                v_prev = v_new
        phi /= n_samples

    return phi, value(players)


def calculate_synergy_score(wrapped_model, X_val_df, y_val, communities, metric_func, n_repeats=5,
                            shapley='auto', max_exact_groups=10, n_shapley_samples=200):
    """
    Evaluates Synergy and Importance using the ModelWrapper.

    Args:
        wrapped_model: Instance of ModelWrapper (SklearnWrapper or PyTorchWrapper)
        X_val_df (pd.DataFrame): Validation features
        y_val (np.array): Validation targets
        communities (list): List of feature groups (from Phase 2). May overlap.
        metric_func: Loss function (MSE or LogLoss)
        n_repeats: Number of permutations to average
        shapley: 'auto' (default) computes community Shapley values only when communities
                 overlap; True always computes them; False never.
                 When computed, 'Impact (%)' is based on Shapley values, which avoids
                 double-counting features shared between communities.
        max_exact_groups: Exact Shapley up to this many communities, sampled beyond it.
        n_shapley_samples: Number of sampled orderings when not exact.
    """
    # Use the wrapper to get consistent baseline predictions
    original_preds = wrapped_model.predict(X_val_df.values)
    baseline_score = metric_func(y_val, original_preds)
    
    results = []

    # How many communities each feature belongs to (>1 means overlapping)
    membership_count = defaultdict(int)
    for group in communities:
        for f in group:
            membership_count[f] += 1

    for i, group in enumerate(communities):
        # 1. Calculate Joint Impact (Shuffle all features in the group together)
        group_scores = []
        for _ in range(n_repeats):
            X_temp = X_val_df.copy()
            shuffled_idx = np.random.permutation(len(X_temp))
            # Shuffle the block of features
            X_temp.loc[:, group] = X_temp.loc[:, group].values[shuffled_idx]
            
            # Use wrapped_model here to handle PyTorch/Sklearn logic automatically
            new_preds = wrapped_model.predict(X_temp.values)
            group_scores.append(metric_func(y_val, new_preds))
        
        avg_group_score = np.mean(group_scores)
        joint_impact = avg_group_score - baseline_score
        
        # 2. Calculate Synergy (Interaction effect above the sum of individuals)
        synergy = 0.0
        sum_ind_impact = 0.0
        
        if len(group) > 1:
            for feature in group:
                feat_scores = []
                for _ in range(n_repeats):
                    X_temp_f = X_val_df.copy()
                    shuffled_idx_f = np.random.permutation(len(X_temp_f))
                    X_temp_f.loc[:, feature] = X_temp_f.loc[:, feature].values[shuffled_idx_f]
                    
                    feat_preds = wrapped_model.predict(X_temp_f.values)
                    feat_scores.append(metric_func(y_val, feat_preds))
                
                ind_impact = np.mean(feat_scores) - baseline_score
                sum_ind_impact += ind_impact
            
            synergy = joint_impact - sum_ind_impact
        else:
            sum_ind_impact = joint_impact
            synergy = 0.0

        shared = [f for f in group if membership_count[f] > 1]
        results.append({
            'Group ID': i,
            'Features': ", ".join(group),
            'Size': len(group),
            'Shared Features': ", ".join(shared),
            'Impact': joint_impact,
            'Synergy': synergy
        })

    df = pd.DataFrame(results)

    # --- 2b. OVERLAP-SAFE IMPACT SHARES (Shapley over communities) ---
    has_overlap = any(c > 1 for c in membership_count.values())
    use_shapley = has_overlap if shapley == 'auto' else bool(shapley)
    if not df.empty:
        if use_shapley:
            phi, total = community_shapley_values(
                wrapped_model, X_val_df, y_val, communities, metric_func,
                n_repeats=n_repeats, max_exact_groups=max_exact_groups,
                n_samples=n_shapley_samples
            )
            df['Shapley Impact'] = phi
            df['Impact (%)'] = 100 * phi / phi.sum() if phi.sum() != 0 else 0.0
        else:
            # Disjoint communities: the original normalisation is valid
            total = df['Impact'].sum()
            df['Impact (%)'] = 100 * df['Impact'] / total if total != 0 else 0.0

    # --- 3. DYNAMIC CLASSIFICATION ---
    if not df.empty:
        max_impact = df['Impact'].max()
        # Threshold for noise (anything less than 1% of max impact)
        noise_threshold = max(0.001, 0.01 * max_impact)
        
        def classify(row):
            if row['Impact'] < noise_threshold:
                return "Weak/Noise"
            if row['Size'] == 1:
                return "Additive (Solo)"
            # If synergy is more than 20% of the total impact, it's highly synergistic
            if abs(row['Synergy']) > (0.2 * row['Impact']):
                return "Synergistic (Complex)"
            return "Additive (Group)"

        df['Nature'] = df.apply(classify, axis=1)
        sort_col = 'Shapley Impact' if 'Shapley Impact' in df.columns else 'Impact'
        df = df.sort_values(sort_col, ascending=False)

    return df

# ---------------------------------------------------------
# Full SINC Pipeline Wrapper
# ---------------------------------------------------------

def run_sinc_pipeline(
    wrapped_model,
    X_train,
    X_val_df,
    y_val,
    feature_names,
    subsample_size=100,
    grid_resolution=50,
    batch_size=2048,
    tau=0.04,
    algo='leiden',
    method='cpm',
    resolution=1.0,
    n_repeats=5,
    random_seed=42,
    alpha=0.2,
    shapley='auto',
    interaction_measure='h',
    n_tii_samples=2000
):
    """
    Executes the SINC (Synergy Interaction Network Community) pipeline.

    Args:
        wrapped_model (ModelWrapper): An instance of SklearnWrapper or PyTorchWrapper. 
            Standardizes .predict() calls across different framework types.
        X_train (np.ndarray): Training data used to calculate feature distributions 
            and percentile-based grids for H-statistics.
        X_val_df (pd.DataFrame): Validation features used in Phase 3 to calculate 
            Out-of-Sample (OOS) Synergy and Impact scores.
        y_val (np.array): Ground-truth targets for the validation set. Used to 
            evaluate loss metrics during permutation tests.
        feature_names (list of str): The human-readable names of the features, 
            used for labeling dataframes and graph nodes.
        subsample_size (int, optional): The number of rows used as the background 
            reference set. A consistent set is used for both marginal and joint 
            PDs to ensure mathematical validity. Defaults to 100.
        grid_resolution (int, optional): The number of points in the 1D and 2D 
            Partial Dependence grids. Higher values capture more detail but 
            increase computation time. Defaults to 50.
        batch_size (int, optional): The number of samples sent to the model 
            per prediction call. Adjust based on available RAM/VRAM. Defaults to 2048.
        tau (float, optional): The H-squared interaction threshold. Pairs with 
            an interaction strength below this value are excluded from the 
            Network Map. Defaults to 0.04.
        algo (str, optional): The community detection algorithm ('leiden', 'louvain',
            'leiden_overlap' or 'cliques'). 'leiden_overlap' allows a feature to belong to
            several communities (Leiden on the line graph); 'cliques' returns all maximal
            cliques of the thresholded network. Defaults to 'leiden'.
        method (str, optional): The objective function for clustering ('cpm' or 'modularity'). 
            CPM is often preferred for feature interaction networks. Defaults to 'cpm'.
        resolution (float, optional): Controls the granularity of clusters. 
            Values >1.0 result in more/smaller groups; <1.0 result in fewer/larger groups. 
            Defaults to 1.0.
        n_repeats (int, optional): The number of permutations performed for each 
            group in Phase 3 to ensure stable Synergy/Impact scores. Defaults to 5.
        random_seed (int, optional): Seed for random number generators to ensure
            reproducibility of subsampling and clustering. Defaults to 42.
        alpha (float, optional): Only for algo='leiden_overlap'. Minimum share of a
            feature's interaction strength a community must hold for the feature to also
            join it. Lower = more overlap. Defaults to 0.2.
        shapley ('auto' | bool, optional): Compute Shapley values over communities for an
            overlap-safe 'Impact (%)'. 'auto' does so only when communities overlap.
        interaction_measure (str, optional): Edge weights of the interaction network.
            'h'   : Friedman's H^2 from partial dependence (default, as in the paper).
            'tii' : total interaction index, normalised by Var(f). Captures every
                    interaction order containing the pair, and is much cheaper.
            tau is applied on the chosen scale, so it must be tuned per measure.
        n_tii_samples (int, optional): Rows in each pick-freeze sample for 'tii'.

    Returns:
        dict: A dictionary containing:
            - 'results' (pd.DataFrame): Table of groups, their nature, impact, and synergy.
            - 'communities' (list): Raw lists of features belonging to each group.
            - 'weights' (pd.DataFrame): The D x D feature-level interaction matrix.
            - 'meta_weights' (pd.DataFrame): The G x G group-level interaction matrix.
            - 'graph_obj' (InteractionGraph): The object used for network visualization.
    """
    start_total = time.time()
    np.random.seed(random_seed)
    
    # --- PHASE 0: CONSISTENCY SETUP ---
    print("\n" + "="*50)
    print("PHASE 0: Background Initialization")
    print("="*50)
    X_bg = get_background_data(X_train, subsample_size=subsample_size)
    print(f" > Using a consistent reference set of {len(X_bg)} samples.")

    # --- PHASE 1: MAPPING ---
    print("\n" + "="*50)
    print("PHASE 1: Interaction Mapping")
    print("="*50)
    
    if interaction_measure.lower() == 'tii':
        print(f" > Computing Total Interaction Indices ({n_tii_samples} samples)...")
        scores = compute_total_interactions(
            wrapped_model,
            X_train,
            n_samples=n_tii_samples,
            batch_size=batch_size,
            seed=random_seed
        )
        square = False
    elif interaction_measure.lower() == 'h':
        print(f" > Computing Marginal PDs (Resolution: {grid_resolution})...")
        pds = compute_marginal_pds(
            wrapped_model,
            X_bg,
            grid_resolution=grid_resolution
        )

        print(f" > Computing Pairwise Interactions (Batch Size: {batch_size})...")
        scores = compute_pairwise_interactions(
            wrapped_model,
            X_bg,
            pds,
            batch_size=batch_size
        )
        square = True
    else:
        raise ValueError("interaction_measure must be 'h' or 'tii'")

    print(f" > Building Network Map (Threshold tau: {tau})...")
    W = build_interaction_network(scores, feature_names, tau=tau, square=square)

    # --- PHASE 2: DISCOVERY ---
    print("\n" + "="*50)
    print("PHASE 2: Community Discovery")
    print("="*50)
    
    ig_obj = InteractionGraph(W)
    communities = ig_obj.detect_communities(
        algorithm=algo, 
        method=method, 
        resolution=resolution,
        seed=random_seed,
        alpha=alpha
    )

    print(f" > Detected {len(communities)} interaction groups.")
    shared = ig_obj.get_overlapping_features()
    if shared:
        print(f" > Features shared between communities: {', '.join(map(str, shared))}")

    # --- PHASE 3: VALIDATION ---
    print("\n" + "="*50)
    print("PHASE 3: Synergy Validation")
    print("="*50)
    
    is_classification = False
    if hasattr(wrapped_model, 'model_type'):
        is_classification = wrapped_model.model_type == 'classification'
    elif hasattr(wrapped_model, 'is_classification'):
        is_classification = wrapped_model.is_classification
        
    metric = log_loss if is_classification else mean_squared_error
    
    results = calculate_synergy_score(
        wrapped_model, 
        X_val_df, 
        y_val, 
        communities, 
        metric,
        n_repeats=n_repeats,
        shapley=shapley
    )
    
    # --- PHASE 4: META-ANALYSIS ---
    meta_df = analyze_inter_group_interactions(W, communities)
    
    elapsed = time.time() - start_total
    print(f"\nSINC Complete in {elapsed:.2f} seconds.")
    
    return {
        'results': results,
        'communities': communities,
        'weights': W,
        'meta_weights': meta_df,
        'graph_obj': ig_obj
    }

# ---------------------------------------------------------
# Macro - Interaction
# ---------------------------------------------------------

def analyze_inter_group_interactions(adjacency_df, communities):
    """
    Collapses the feature-level adjacency matrix into a group-level matrix.
    """
    n_groups = len(communities)
    meta_matrix = np.zeros((n_groups, n_groups))
    
    print(f" Collapsing network into {n_groups}x{n_groups} Meta-Graph...")
    
    # Iterate through every pair of groups
    for i in range(n_groups):
        for j in range(n_groups):
            if i == j:
                # Diagonal: Internal cohesion (Sum of weights INSIDE the group)
                # We halve it because the matrix is symmetric and we don't want double counting
                group_features = communities[i]
                sub_matrix = adjacency_df.loc[group_features, group_features]
                # Sum upper triangle
                total_weight = sub_matrix.values[np.triu_indices(len(group_features), k=1)].sum()
                meta_matrix[i, j] = total_weight
            else:
                # Off-Diagonal: Interaction BETWEEN Group i and Group j.
                # With overlapping communities, shared features are excluded so that
                # interactions inside the overlap are not counted as between-group.
                shared = set(communities[i]) & set(communities[j])
                group_i_feats = [f for f in communities[i] if f not in shared]
                group_j_feats = [f for f in communities[j] if f not in shared]

                if not group_i_feats or not group_j_feats:
                    meta_matrix[i, j] = 0.0  # one group is contained in the other
                    continue

                # Extract the sub-rectangle connecting the two groups
                sub_matrix = adjacency_df.loc[group_i_feats, group_j_feats]
                meta_matrix[i, j] = sub_matrix.values.sum()

    # Convert to DataFrame for readability
    meta_df = pd.DataFrame(
        meta_matrix, 
        index=[f"Grp {i}" for i in range(n_groups)],
        columns=[f"Grp {i}" for i in range(n_groups)]
    )
    
    return meta_df

def plot_meta_heatmap(meta_df, results_df=None, title=None):
    """
    Plots the Meta-Interaction Map.
    
    Args:
        meta_df (pd.DataFrame): The group-level adjacency matrix.
        results_df (pd.DataFrame, optional): The results table from Phase 3. 
                                             If provided, filters out 'Weak/Noise' groups.
                                             If None, plots the full matrix.
        title (str, optional): Custom title for the plot.
    """
    matrix_to_plot = meta_df
    
    # --- Logic for Filtering ---
    if results_df is not None:
        # Filter: Keep only groups that are NOT 'Weak/Noise'
        active_df = results_df[results_df['Nature'] != 'Weak/Noise']
        
        if active_df.empty:
            print("Warning: No active groups found. Plotting full matrix instead.")
        else:
            # Get Group IDs and convert to labels (e.g., 0 -> "Grp 0")
            active_ids = sorted(active_df['Group ID'].values)
            keep_labels = [f"Grp {i}" for i in active_ids]
            
            # Intersection: ensure we only ask for labels that actually exist in the matrix
            valid_labels = [lbl for lbl in keep_labels if lbl in meta_df.index]
            
            # Apply filter
            matrix_to_plot = meta_df.loc[valid_labels, valid_labels]
            
            # Set default title for filtered view if user didn't provide one
            if title is None:
                title = "Meta-Interaction Map (Active Groups Only)"

    # --- Logic for Default Title ---
    if title is None:
        title = "Meta-Interaction Map (Full Network)"

    # --- Plotting ---
    # Dynamic size: smaller if few groups, standard size otherwise
    n_groups = len(matrix_to_plot)
    figsize = (8, 6) if n_groups < 10 else (10, 8)
    
    plt.figure(figsize=figsize)
    sns.heatmap(matrix_to_plot, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)
    plt.title(title)
    plt.show()