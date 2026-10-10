"""Task data for the original five tasks (copied from reckon.py so the runner needs no other modules)."""
import itertools
import numpy as np
from sklearn.datasets import load_iris, load_digits, make_circles


def task_data(name, seed):
    """Returns raw features in [0,1] (Xtr, ytr, Xte, yte) and class count."""
    if name == "iris":
        X, y = load_iris(return_X_y=True)
        perm = np.random.default_rng(0).permutation(150); tr, te = perm[:100], perm[100:]
    elif name == "circles":
        X, y = make_circles(300, noise=0.08, factor=0.5, random_state=seed)
        tr, te = np.arange(200), np.arange(200, 300)
    elif name.startswith("parity"):
        n = int(name[-1])
        X = np.array(list(itertools.product([0, 1], repeat=n)), float); y = (X.sum(1) % 2).astype(int)
        tr = te = np.arange(len(y))
    elif name == "digits":
        X, y = load_digits(return_X_y=True)
        perm = np.random.default_rng(seed).permutation(len(y)); cut = int(0.7 * len(y))
        tr, te = perm[:cut], perm[cut:]
    X = (X - X.min(0)) / np.maximum(X.max(0) - X.min(0), 1e-12)
    return X[tr], y[tr], X[te], y[te], int(y.max() + 1)
