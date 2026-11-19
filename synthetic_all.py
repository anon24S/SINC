"""
Synthetic benchmark (F1-F10) for SINC, SHAP-IQ and Archipelago.

For every data-generating function, all three methods are run on two models:
  - "mlp":    a SmallNet MLP trained on the data (the black box in the paper)
  - "oracle": the true data-generating function itself, used directly as the model

The oracle shows what each method recovers when the model is exactly the ground
truth, so differences between oracle and MLP results come from what the MLP learned.
"""

import os
import sys
import ssl
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ssl._create_default_https_context = ssl._create_unverified_context

_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if _here not in sys.path:
    sys.path.insert(0, _here)

from sincv3 import (
    get_background_data,
    compute_marginal_pds,
    compute_pairwise_interactions,
    build_interaction_network,
    InteractionGraph,
    calculate_synergy_score,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATASETS = [f"f{i}" for i in range(11, 13)]   # f1, ..., f10
MODELS = ["mlp", "oracle"]

N_SAMPLES = 10000
TAU = 0.003            # interaction threshold used in the paper
GRID_RESOLUTION = 30   # R
SUBSAMPLE = 100        # M
N_REPEATS = 5          # permutation repeats in Stage 3
EPOCHS = 150           # MLP training epochs
N_EXPLAIN = 100        # validation instances explained by SHAP-IQ and Archipelago
SHAPIQ_MAX_ORDER = 5   # interactions up to this order (5 = largest term order in F1-F10)
SHAPIQ_INDICES = ["k-SII", "FSII", "STII", "Moebius"]  # all computed from the same global game
SHAPIQ_N_EVAL = 1000   # validation rows on which the global loss game is evaluated
SHAPIQ_N_BG = 300      # background rows used to marginalise absent features
# (100/100 was tested and gives noticeably noisier interactions)

RUN_SHAPIQ = True
RUN_ARCHIPELAGO = True

OUT_DIR = os.path.join(_here, "synth_results_all")

IMPORTANCE_THRESHOLD = 0.02  # keep items contributing >= 2% of the total importance

# ---------------------------------------------------------
# Data-generating functions (vectorised, numpy in -> numpy out)
# ---------------------------------------------------------
def f1(X):
    return (np.pi ** (X[:, 0] * X[:, 1])) * np.sqrt(2 * X[:, 2]) \
        - np.arcsin(X[:, 3]) \
        + np.log(X[:, 2] + X[:, 4]) \
        - (X[:, 8] / X[:, 9]) * np.sqrt(X[:, 6] / X[:, 7]) \
        - X[:, 1] * X[:, 6]

def f2(X):
    return (np.pi ** (X[:, 0] * X[:, 1])) * np.sqrt(2 * np.abs(X[:, 2])) \
        - np.arcsin(0.5 * X[:, 3]) \
        + np.log(np.abs(X[:, 2] + X[:, 4]) + 1) \
        + (X[:, 8] / (1 + np.abs(X[:, 9]))) * np.sqrt(np.abs(X[:, 6]) / (1 + np.abs(X[:, 7]))) \
        - X[:, 1] * X[:, 6]

def f3(X):
    return np.exp(np.abs(X[:, 0] - X[:, 1])) \
        + np.abs(X[:, 1] * X[:, 2]) \
        - np.abs(X[:, 2]) ** (2 * np.abs(X[:, 3])) \
        + np.log(X[:, 3]**2 + X[:, 4]**2 + X[:, 6]**2 + X[:, 7]**2) \
        + X[:, 8] \
        + 1 / (1 + X[:, 9]**2)

def f4(X):
    return np.exp(np.abs(X[:, 0] - X[:, 1])) \
        + np.abs(X[:, 1] * X[:, 2]) \
        - np.abs(X[:, 2]) ** (2 * np.abs(X[:, 3])) \
        + (X[:, 0] * X[:, 3])**2 \
        + np.log(X[:, 3]**2 + X[:, 4]**2 + X[:, 6]**2 + X[:, 7]**2) \
        + X[:, 8] \
        + 1 / (1 + X[:, 9]**2)

def f5(X):
    return 1 / (1 + X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2) \
        + np.sqrt(np.exp(X[:, 3] + X[:, 4])) \
        + np.abs(X[:, 5] + X[:, 6]) \
        + X[:, 7] * X[:, 8] * X[:, 9]

def f6(X):
    return np.exp(np.abs(X[:, 0] * X[:, 1]) + 1) \
        - np.exp(np.abs(X[:, 2] + X[:, 3]) + 1) \
        + np.cos(X[:, 4] + X[:, 5] - X[:, 7]) \
        + np.sqrt(X[:, 7]**2 + X[:, 8]**2 + X[:, 9]**2)

def f7(X):
    return (np.arctan(X[:, 0]) + np.arctan(X[:, 1]))**2 \
        + np.maximum(X[:, 2] * X[:, 3] + X[:, 5], 0) \
        - 1 / (1 + (X[:, 3] * X[:, 4] * X[:, 5] * X[:, 6] * X[:, 7])**2) \
        + (np.abs(X[:, 6]) / (1 + np.abs(X[:, 8])))**5 \
        + np.sum(X[:, 0:10], axis=1)

def f8(X):
    return X[:, 0] * X[:, 1] \
        + 2 ** (X[:, 2] + X[:, 4] + X[:, 5]) \
        + 2 ** (X[:, 2] + X[:, 3] + X[:, 4] + X[:, 6]) \
        + np.sin(X[:, 6] * np.sin(X[:, 7] + X[:, 8])) \
        + np.arccos(0.9 * X[:, 9])

def f9(X):
    return np.tanh(X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]) * np.sqrt(np.abs(X[:, 4])) \
        + np.exp(X[:, 4] + X[:, 5]) \
        + np.log((X[:, 5] * X[:, 6] * X[:, 7])**2 + 1) \
        + X[:, 8] * X[:, 9] \
        + 1 / (1 + np.abs(X[:, 9]))

def f10(X):
    return np.sinh(X[:, 0] + X[:, 1]) \
        + np.arccos(np.tanh(X[:, 2] + X[:, 4] + X[:, 6])) \
        + np.cos(X[:, 3] + X[:, 4]) \
        + 1 / np.cos(X[:, 6] * X[:, 8])

def f11(X):
    return X[:, 0]*X[:, 1] + X[:, 1]*X[:, 2] + X[:, 0]*X[:, 2]

def f12(X):
    return X[:, 0]*X[:, 1]*X[:, 2]

FUNCTIONS = {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5,
             "f6": f6, "f7": f7, "f8": f8, "f9": f9, "f10": f10, "f11": f11, "f12": f12}


def sample_X(dataset_name, n_samples):
    """Same sampling (and seed) as the original choose_dataset, so the data is unchanged."""
    np.random.seed(42)
    if dataset_name == "f1":
        X = np.zeros((n_samples, 10))
        standard_indices = [0, 1, 2, 5, 6, 8]   # uniform [0.0, 1.0]
        X[:, standard_indices] = np.random.uniform(0.0, 1.0, size=(n_samples, len(standard_indices)))
        offset_indices = [3, 4, 7, 9]           # uniform [0.6, 1.0]
        X[:, offset_indices] = np.random.uniform(0.6, 1.0, size=(n_samples, len(offset_indices)))
        return X
    if dataset_name in FUNCTIONS:
        return np.random.uniform(-1, 1, size=(n_samples, 3))
    raise ValueError(f"Unsupported dataset: {dataset_name}")


# ---------------------------------------------------------
# Ground truth for the recovery table: every term of the function, plus every
# uninformative feature as a singleton (1-based feature numbers)
# ---------------------------------------------------------
GROUND_TRUTH = {
    "f1":  [[1, 2, 3], [4], [3, 5], [7, 8, 9, 10], [2, 7], [6]],
    "f2":  [[1, 2, 3], [4], [3, 5], [7, 8, 9, 10], [2, 7], [6]],
    "f3":  [[1, 2], [2, 3], [3, 4], [4, 5, 7, 8], [9], [10], [6]],
    "f4":  [[1, 2], [2, 3], [3, 4], [1, 4], [4, 5, 7, 8], [9], [10], [6]],
    "f5":  [[1, 2, 3], [4, 5], [6, 7], [8, 9, 10]],
    "f6":  [[1, 2], [3, 4], [5, 6, 8], [8, 9, 10], [7]],
    "f7":  [[1, 2], [3, 4, 6], [4, 5, 6, 7, 8], [7, 9]] + [[i] for i in range(1, 11)],
    "f8":  [[1, 2], [3, 5, 6], [3, 4, 5, 7], [7, 8, 9], [10]],
    "f9":  [[1, 2, 3, 4, 5], [5, 6], [6, 7, 8], [9, 10], [10]],
    "f10": [[1, 2], [3, 5, 7], [4, 5], [7, 9], [6], [8], [10]],
    "f11": [[1, 2], [2, 3], [1, 3]],
    "f12": [[1, 2, 3]]
}


def ground_truth_sets(dataset_name):
    return {frozenset(f"x_{i}" for i in term) for term in GROUND_TRUTH[dataset_name]}


def parse_sets(names, sep):
    """'x_1 + x_2' (sep='+') or ['x_1', 'x_2'] lists -> frozensets of feature names."""
    out = set()
    for item in names:
        if isinstance(item, str):
            out.add(frozenset(t.strip() for t in item.split(sep)))
        else:
            out.add(frozenset(item))
    return out


# ---------------------------------------------------------
# MLP
# ---------------------------------------------------------
class SmallNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 16),         nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def train_nn(model, X_train_np, y_train_np, epochs=150, lr=3e-3, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    X_t = torch.from_numpy(X_train_np).float().to(device)
    y_t = torch.from_numpy(y_train_np).float().to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    opt, crit = torch.optim.Adam(model.parameters(), lr=lr), nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


# ---------------------------------------------------------
# Model-agnostic prediction functions
# Every method below only needs predict_fn: (n, 10) numpy array -> (n,) numpy array
# ---------------------------------------------------------
def make_mlp_predict(model):
    device = next(model.parameters()).device
    def predict(X):
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(X).to(device)).cpu().numpy()
        return out.reshape(-1)
    return predict


def make_oracle_predict(dataset_name):
    fn = FUNCTIONS[dataset_name]
    def predict(X):
        return np.asarray(fn(np.asarray(X, dtype=np.float64)), dtype=np.float64).reshape(-1)
    return predict


class FnWrapper:
    """SINC model wrapper around a plain prediction function."""
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn
    def predict(self, X):
        return self.predict_fn(X)


class TabularXformer:
    def __init__(self, instance, baseline):
        self.instance = np.array(instance)
        self.baseline = np.array(baseline)
        self.num_features = len(self.instance)  # Archipelago needs this

    def __call__(self, mask):
        x = np.copy(self.baseline)
        for i in np.argwhere(mask == True).flatten():
            x[i] = self.instance[i]
        return x


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def filter_by_share(df, value_col, threshold=IMPORTANCE_THRESHOLD):
    """Add a '<value_col>_Share_%' column and keep rows whose share is >= threshold."""
    df = df.copy()
    total = df[value_col].sum()
    share = df[value_col] / total if total > 0 else 0.0
    df[f"{value_col}_Share_%"] = (share * 100).round(2)
    kept = df[share >= threshold].sort_values(by=value_col, ascending=False).reset_index(drop=True)
    return kept, total, len(df)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def fmt_groups(groups):
    return " | ".join("{" + ", ".join(g) + "}" for g in groups)


# ---------------------------------------------------------
# Methods
# ---------------------------------------------------------
def run_sinc(label, predict_fn, X_bg, X_val_df, y_val, feature_names, tau):
    print(f"\n{'='*55}\n  SINC PIPELINE — {label}\n{'='*55}")
    wrapper = FnWrapper(predict_fn)

    print("\n--- Stage 1: Interaction Mapping ---")
    pds = compute_marginal_pds(model_wrapper=wrapper, X_background=X_bg,
                               grid_resolution=GRID_RESOLUTION)
    interactions = compute_pairwise_interactions(model_wrapper=wrapper, X_background=X_bg,
                                                 pds=pds, batch_size=512)
    W = build_interaction_network(interactions, feature_names, tau=tau)

    print("\n--- Stage 2: Community Detection (maximal cliques) ---")
    communities = InteractionGraph(W).detect_communities(algorithm="cliques", min_weight=0.0)
    print(f"Found {len(communities)} communities:")
    for i, c in enumerate(communities):
        print(f"  Group {i}: {c}")

    print("\n--- Stage 3: Impact Weighting ---")
    results = calculate_synergy_score(
        wrapped_model=wrapper,
        X_val_df=X_val_df,
        y_val=y_val,
        communities=communities,
        metric_func=mean_squared_error,
        n_repeats=N_REPEATS,
    )
    print("\n--- FINAL RESULTS ---")
    print(results.to_string(index=False))
    return results, communities


def make_global_loss_game(predict_fn, X_eval, y_eval, X_bg):
    """
    Global cooperative game on the model's loss (SAGE-style).

    v(S) = L(empty) - L(S), where L(S) is the MSE on (X_eval, y_eval) when only the features in S
    are known and the remaining features are marginalised over the background data:
        f_S(x) = mean_b f(x_S, b_{-S}).
    v(empty) = 0, and v(all features) is the total loss reduction achieved by the model.
    """
    n_eval, n_bg = len(X_eval), len(X_bg)
    X_rep = np.repeat(X_eval, n_bg, axis=0)   # each evaluation row paired with every background row
    B_rep = np.tile(X_bg, (n_eval, 1))

    def loss(mask):
        X_mix = np.where(mask[None, :], X_rep, B_rep)
        f_S = predict_fn(X_mix).reshape(n_eval, n_bg).mean(axis=1)
        return np.mean((y_eval - f_S) ** 2)

    loss_empty = loss(np.zeros(X_eval.shape[1], dtype=bool))

    def game(coalitions):
        coalitions = np.atleast_2d(np.asarray(coalitions, dtype=bool))
        return np.array([loss_empty - loss(c) for c in coalitions])

    return game


def run_shapiq(label, predict_fn, X_train, X_val, y_val, feature_names,
               indices=SHAPIQ_INDICES, max_order=SHAPIQ_MAX_ORDER):
    """
    Global SHAP-IQ: interaction indices of the global loss game, computed exactly.
    The game is evaluated once on all 2^n coalitions; every index is then derived from it.
    Returns {index: DataFrame of interactions of order 2..max_order with >= 2% of the total}.
    """
    print(f"\n{'='*55}\n  GLOBAL SHAP-IQ ({', '.join(indices)}; order <= {max_order}) — {label}\n{'='*55}")
    import shapiq

    n = X_val.shape[1]
    rng = np.random.default_rng(42)
    X_bg = X_train[rng.choice(len(X_train), size=min(SHAPIQ_N_BG, len(X_train)), replace=False)]
    game = make_global_loss_game(predict_fn, X_val[:SHAPIQ_N_EVAL], y_val[:SHAPIQ_N_EVAL], X_bg)

    print(f"Evaluating all 2^{n} = {2**n} coalitions of the global loss game...")
    try:
        exact = shapiq.ExactComputer(n_players=n, game=game)        # newer shapiq
    except TypeError:
        exact = shapiq.ExactComputer(n_players=n, game_fun=game)    # older shapiq

    results = {}
    for index in indices:
        order = n if index == "Moebius" else max_order   # Moebius is defined for every set
        iv = exact(index=index, order=order)
        if hasattr(iv, "dict_values"):
            items = iv.dict_values.items()
        else:
            items = ((key, iv.values[pos]) for key, pos in iv.interaction_lookup.items())

        # interactions only (order >= 2); the 2% share is computed among these
        rows = [{"Feature_Set": " + ".join(feature_names[j] for j in sorted(key)),
                 "Order": len(key),
                 "Value": float(val),
                 "SHAPIQ_Interaction_Strength": abs(float(val))}
                for key, val in items if 1 <= len(key) <= max_order]
        sets_df, total, n_all = filter_by_share(pd.DataFrame(rows), "SHAPIQ_Interaction_Strength")
        results[index] = sets_df

        print(f"\n[{index}] interactions (orders 1-{max_order}) >= {IMPORTANCE_THRESHOLD:.0%} of the total "
              f"|value| ({len(sets_df)} of {n_all} feature sets; total = {total:.4f}):")
        print(sets_df.to_string(index=False))
    return results


def run_archipelago(label, predict_fn, X_train, X_val_df, feature_names):
    print(f"\n{'='*55}\n  ARCHIPELAGO GLOBAL BASELINE — {label}\n{'='*55}")
    from src.explainer import Archipelago

    def archipelago_model(batch_data):
        return predict_fn(np.vstack(batch_data)).reshape(-1, 1)

    baseline = X_train.mean(axis=0)
    sample = X_val_df.sample(n=N_EXPLAIN, random_state=42)
    total_instances = len(sample)

    global_scores = defaultdict(float)
    interaction_counts = defaultdict(int)

    print(f"Running explanations for {total_instances} instances...")
    for i, (_, row) in enumerate(sample.iterrows()):
        apgo = Archipelago(archipelago_model,
                           data_xformer=TabularXformer(row.values, baseline),
                           output_indices=0)
        explanations = apgo.explain(top_k=5)
        if isinstance(explanations, dict):
            explanations = explanations.items()

        for feature_set, score in explanations:
            community = tuple(sorted(feature_set))
            global_scores[community] += abs(score)
            interaction_counts[community] += 1

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{total_instances} instances")

    rows = [{"Feature_Community": " + ".join(feature_names[idx] for idx in community),
             "Community_Size": len(community),
             "Global_Importance": total_score / total_instances,
             "Appearance_Count": interaction_counts[community]}
            for community, total_score in global_scores.items()]

    df_results, total, n_all = filter_by_share(pd.DataFrame(rows), "Global_Importance")
    print(f"\n=== ARCHIPELAGO COMMUNITIES >= {IMPORTANCE_THRESHOLD:.0%} OF TOTAL [{label}] ===")
    print(f"({len(df_results)} of {n_all} communities; total importance = {total:.4f})")
    print(df_results.to_string())
    return df_results


# ---------------------------------------------------------
# Recovery table (like Table 5): "MLP (oracle)" per method
# ---------------------------------------------------------
def print_recovery_table(rec_df):
    methods = [m for m in ["SINC", "Archipelago"] + [f"SHAP-IQ ({i})" for i in SHAPIQ_INDICES]
               if m in rec_df.columns]
    rows, tex_rows = [], []
    for ds in DATASETS:
        sub = rec_df[rec_df["dataset"] == ds].set_index("model")
        if sub.empty:
            continue
        total = int(sub["Total"].iloc[0])
        row = {"Function": ds.upper()}
        for m in methods:
            mlp = sub.at["mlp", m] if "mlp" in sub.index else float("nan")
            orc = sub.at["oracle", m] if "oracle" in sub.index else float("nan")
            row[m] = f"{mlp:.0f} ({orc:.0f})"
        row["Total"] = str(total)
        rows.append(row)

    # mean recovery: per-function rate, averaged over functions
    mean_row = {"Function": "Mean recovery (%)"}
    for m in methods:
        rates = {}
        for model in ["mlp", "oracle"]:
            sub = rec_df[rec_df["model"] == model]
            rates[model] = 100 * (sub[m] / sub["Total"]).mean()
        mean_row[m] = f"{rates['mlp']:.1f} ({rates['oracle']:.1f})"
    mean_row["Total"] = "100.0"
    rows.append(mean_row)

    table = pd.DataFrame(rows)
    print("\n" + "#" * 70 + "\n#  RECOVERED GROUND-TRUTH TERMS: MLP (oracle)\n" + "#" * 70)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(OUT_DIR, "recovery_table.csv"), index=False)
    rec_df.to_csv(os.path.join(OUT_DIR, "recovery_counts.csv"), index=False)

    # LaTeX version for the paper
    header = " & ".join(["Function"] + methods + ["Total"]) + r" \\"
    body = []
    for r in rows:
        name = r["Function"]
        if name.startswith("F") and name[1:].isdigit():
            name = f"$F_{{{name[1:]}}}$"
        else:
            body.append(r"\midrule")
            name = name.replace("%", r"\%")
        body.append(" & ".join([name] + [str(r[m]) for m in methods] + [str(r["Total"])]) + r" \\")
    tex = "\n".join([r"\begin{tabular}{@{}l" + "c" * (len(methods) + 1) + "@{}}", r"\toprule", header,
                     r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"])
    with open(os.path.join(OUT_DIR, "recovery_table.tex"), "w") as fh:
        fh.write(tex + "\n")
    print(f"\nSaved recovery_table.csv / .tex in {OUT_DIR}")


# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    recovery = []   # one row per dataset x model with the number of recovered terms per method

    for dataset_name in DATASETS:
        print("\n" + "#" * 70 + f"\n#  DATASET {dataset_name}\n" + "#" * 70)

        X = sample_X(dataset_name, N_SAMPLES)
        y = FUNCTIONS[dataset_name](X)
        feats = [f"x_{i + 1}" for i in range(X.shape[1])]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val_df = pd.DataFrame(X_val, columns=feats)
        X_bg = get_background_data(X_train, subsample_size=SUBSAMPLE)  # shared by both models

        for model_type in MODELS:
            np.random.seed(42)
            torch.manual_seed(42)

            if model_type == "mlp":
                print("\n" + "=" * 55 + f"\n  Training MLP ({EPOCHS} epochs) [{dataset_name}]\n" + "=" * 55)
                model = train_nn(SmallNet(X.shape[1]), X_train, y_train, epochs=EPOCHS)
                predict_fn = make_mlp_predict(model)
            else:
                predict_fn = make_oracle_predict(dataset_name)

            r2 = r2_score(y_val, predict_fn(X_val))
            label = f"{dataset_name} {model_type}"
            print(f"\n  [{label}] R²: {r2:.4f}")

            row = {"dataset": dataset_name, "model": model_type, "R2": r2, "tau": TAU}
            prefix = os.path.join(OUT_DIR, f"{dataset_name}_{model_type}")
            gt = ground_truth_sets(dataset_name)
            rec = {"dataset": dataset_name, "model": model_type, "Total": len(gt)}

            # Each method in its own try, so one failure does not stop the rest
            try:
                results, communities = run_sinc(label, predict_fn, X_bg, X_val_df, y_val, feats, TAU)
                results.to_csv(f"{prefix}_sinc.csv", index=False)
                row["sinc_communities"] = fmt_groups(communities)
                rec["SINC"] = len(gt & parse_sets(communities, ","))
            except Exception as e:
                print(f"[{label}] SINC failed: {e}")
                row["sinc_communities"] = f"FAILED: {e}"

            if RUN_SHAPIQ:
                try:
                    shapiq_results = run_shapiq(label, predict_fn, X_train, X_val, y_val, feats)
                    for index, df in shapiq_results.items():
                        df.to_csv(f"{prefix}_shapiq_{index}.csv", index=False)
                        row[f"shapiq_{index}"] = " | ".join(df["Feature_Set"])
                        rec[f"SHAP-IQ ({index})"] = len(gt & parse_sets(df["Feature_Set"], "+"))
                except Exception as e:
                    print(f"[{label}] SHAP-IQ failed: {e}")
                    row["shapiq"] = f"FAILED: {e}"

            if RUN_ARCHIPELAGO:
                try:
                    arch_df = run_archipelago(label, predict_fn, X_train, X_val_df, feats)
                    arch_df.to_csv(f"{prefix}_archipelago.csv", index=False)
                    row["archipelago_communities"] = " | ".join(arch_df["Feature_Community"])
                    rec["Archipelago"] = len(gt & parse_sets(arch_df["Feature_Community"], "+"))
                except Exception as e:
                    print(f"[{label}] Archipelago failed: {e}")
                    row["archipelago_communities"] = f"FAILED: {e}"

            summary.append(row)
            recovery.append(rec)
            # Save after every run so partial results survive a crash
            pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    summary_df = pd.DataFrame(summary)
    print("\n" + "#" * 70 + "\n#  SUMMARY\n" + "#" * 70)
    print(summary_df[["dataset", "model", "R2", "shapiq"]].to_string(index=False))

    print_recovery_table(pd.DataFrame(recovery))
    print(f"\nAll results saved in {OUT_DIR}")