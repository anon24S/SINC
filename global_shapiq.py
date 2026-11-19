
import numpy as np
import pandas as pd

IMPORTANCE_THRESHOLD = 0.02   # report feature sets with >= 2% of the total absolute value
EXACT_LIMIT = 16              # exact computation up to 2^16 coalitions; approximation above


def make_global_loss_game(predict_fn, X_eval, y_eval, X_bg, loss="mse", batch_rows=500_000):
    """Return game(coalitions) -> loss reduction v(S) for every row of a boolean coalition matrix."""
    X_eval, X_bg = np.asarray(X_eval, dtype=float), np.asarray(X_bg, dtype=float)
    y_eval = np.asarray(y_eval, dtype=float)
    n_eval, n_bg = len(X_eval), len(X_bg)
    X_rep = np.repeat(X_eval, n_bg, axis=0)   # every evaluation row paired with every background row
    B_rep = np.tile(X_bg, (n_eval, 1))

    def predict(X):
        return np.concatenate([np.asarray(predict_fn(X[s:s + batch_rows]), dtype=float).reshape(-1)
                               for s in range(0, len(X), batch_rows)])

    def loss_fn(f):
        if loss == "mse":
            return np.mean((y_eval - f) ** 2)
        if loss == "logloss":
            p = np.clip(f, 1e-7, 1 - 1e-7)
            return -np.mean(y_eval * np.log(p) + (1 - y_eval) * np.log(1 - p))
        raise ValueError("loss must be 'mse' or 'logloss'")

    def loss_of(mask):
        X_mix = np.where(mask[None, :], X_rep, B_rep)
        f_S = predict(X_mix).reshape(n_eval, n_bg).mean(axis=1)
        return loss_fn(f_S)

    loss_empty = loss_of(np.zeros(X_eval.shape[1], dtype=bool))

    def game(coalitions):
        coalitions = np.atleast_2d(np.asarray(coalitions, dtype=bool))
        return np.array([loss_empty - loss_of(c) for c in coalitions])

    return game


def _items(iv):
    if hasattr(iv, "dict_values"):
        return iv.dict_values.items()
    return ((key, iv.values[pos]) for key, pos in iv.interaction_lookup.items())


def run_global_shapiq(predict_fn, X_eval, y_eval, X_bg, feature_names, loss="mse",
                      max_order=5, index="k-SII", budget=2**13, random_state=42, label=""):
    """
    Global SHAP-IQ feature sets of order 1..max_order with >= 2% of the total absolute value.
    Exact for up to 16 features; for more features, KernelSHAP-IQ with the given budget
    (number of evaluated coalitions).
    """
    import shapiq

    n = len(feature_names)
    max_order = min(max_order, n)
    game = make_global_loss_game(predict_fn, X_eval, y_eval, X_bg, loss=loss)

    if n <= EXACT_LIMIT:
        print(f"[{label}] Global SHAP-IQ ({index}, order <= {max_order}): exact over 2^{n} = {2**n} coalitions")
        try:
            exact = shapiq.ExactComputer(n_players=n, game=game)
        except TypeError:   # older shapiq versions
            exact = shapiq.ExactComputer(n_players=n, game_fun=game)
        iv = exact(index=index, order=n if index == "Moebius" else max_order)
    else:
        print(f"[{label}] Global SHAP-IQ ({index}, order <= {max_order}): KernelSHAP-IQ with {budget} coalitions")
        approximator = shapiq.KernelSHAPIQ(n=n, max_order=max_order, index=index, random_state=random_state)
        iv = approximator.approximate(budget=budget, game=game)

    rows = [{"Feature_Set": " + ".join(feature_names[j] for j in sorted(key)),
             "Order": len(key),
             "Value": float(val),
             "Abs_Value": abs(float(val))}
            for key, val in _items(iv) if 1 <= len(key) <= max_order]
    df = pd.DataFrame(rows)
    total = df["Abs_Value"].sum()
    df["Share_%"] = 100 * df["Abs_Value"] / total if total > 0 else 0.0
    df = df[df["Share_%"] >= 100 * IMPORTANCE_THRESHOLD].sort_values("Abs_Value", ascending=False)
    df = df.reset_index(drop=True)

    print(f"\n[{label}] Global SHAP-IQ feature sets >= {IMPORTANCE_THRESHOLD:.0%} of the total "
          f"({len(df)} of {len(rows)}; total |value| = {total:.4f}):")
    print(df.to_string(index=False))
    return df