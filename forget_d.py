"""Forgetting protocol (pre-registered): learn digits task A, then B, equal passes; test A again."""
import json, sys, copy, numpy as np
from sklearn.datasets import load_digits
PAIRS = [((0, 1), (7, 9)), ((3, 8), (4, 6))]

def split(X, y, digits, seed, dev=False):
    m = np.isin(y, digits)
    Xs, ys = X[m], (y[m] == digits[1]).astype(int)
    perm = np.random.default_rng(seed).permutation(len(ys)); cut = int(0.7 * len(ys))
    tr, te = perm[:cut], perm[cut:]
    if dev:                                   # development never sees the test part
        cut2 = int(0.7 * len(tr)); tr, te = tr[:cut2], tr[cut2:]
    return Xs[tr], ys[tr], Xs[te], ys[te]

def run(make, passes, seeds, dev=False, consolidate=None):
    X, y = load_digits(return_X_y=True); X = X / 16.0
    out = []
    for A, B in PAIRS:
        for sd in seeds:
            a_tr, ay_tr, a_te, ay_te = split(X, y, A, sd, dev)
            b_tr, by_tr, b_te, by_te = split(X, y, B, sd, dev)
            # one encoder for both tasks: fitted on what the system has seen first (task A)
            m = make(sd).fit(a_tr, ay_tr, passes)
            a1 = np.mean(m.predict(a_te) == ay_te)
            if consolidate is not None:
                consolidate(m, a_tr, ay_tr)
            m.fit(b_tr, by_tr, passes)
            out.append(dict(A=A, seed=sd, A_after_A=a1, B_after_B=float(np.mean(m.predict(b_te) == by_te)),
                            A_after_B=float(np.mean(m.predict(a_te) == ay_te))))
    return out

def summary(out):
    return {k: round(float(np.mean([o[k] for o in out])), 4) for k in ("A_after_A", "B_after_B", "A_after_B")}
