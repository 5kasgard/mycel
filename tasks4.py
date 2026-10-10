"""Round four tasks: the original five plus six unseen ones (never used in development)."""
import itertools, numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, make_moons, make_classification
from data_orig import task_data as task_data_orig

ORIGINAL = ["iris", "circles", "parity3", "parity4", "digits"]
UNSEEN = ["wine", "cancer", "moons", "parity5", "logic8", "synth20"]

def _split(X, y, seed, frac=0.7):
    perm = np.random.default_rng(seed).permutation(len(y)); cut = int(frac * len(y))
    tr, te = perm[:cut], perm[cut:]
    lo, hi = X[tr].min(0), X[tr].max(0)                       # scale from training data only
    X = np.clip((X - lo) / np.maximum(hi - lo, 1e-12), -0.5, 1.5)
    return X[tr], y[tr], X[te], y[te], int(y.max() + 1)

def task_data(name, seed):
    if name in ORIGINAL:
        return task_data_orig(name, seed)
    if name == "wine":
        X, y = load_wine(return_X_y=True); return _split(X, y, seed)
    if name == "cancer":
        X, y = load_breast_cancer(return_X_y=True); return _split(X, y, seed)
    if name == "moons":
        X, y = make_moons(300, noise=0.2, random_state=seed); return _split(X, y, seed)
    if name == "parity5":
        X = np.array(list(itertools.product([0, 1], repeat=5)), float); y = (X.sum(1) % 2).astype(int)
        return X, y, X, y, 2
    if name == "logic8":
        X = np.array(list(itertools.product([0, 1], repeat=8)), float)
        r = np.random.default_rng(12345)                       # one fixed rule for every seed
        terms = [(r.choice(8, 3, replace=False), r.integers(0, 2, 3)) for _ in range(4)]
        y = np.zeros(len(X), int)
        for idx, val in terms:
            y |= np.all(X[:, idx] == val, 1).astype(int)
        return _split(X, y, seed)
    if name == "synth20":
        X, y = make_classification(600, 20, n_informative=5, n_redundant=5, n_classes=3, n_clusters_per_class=2,
                                   random_state=1000 + seed)
        return _split(X, y, seed)
    raise ValueError(name)
