"""Round five streams. Each returns dict(X, y, K, changes, recur, tasks) where
changes  = indices where the rule changes
recur    = list of (first_start, return_start) pairs: a rule that comes back
tasks    = optional held-out sets per task: list of (Xtest, ytest), for retention at the end

DEV streams may be used for tuning. TEST streams are sealed until the design is frozen."""
import numpy as np
from sklearn.datasets import load_digits

DEV = ["sea", "hyperplane", "splitdigits"]
TEST = ["stagger", "rbfdrift", "permdigits", "agrawal", "seanoisy", "elec"]


def _flip(y, noise, r, K=2):
    if noise <= 0:
        return y
    m = r.random(len(y)) < noise
    y = y.copy(); y[m] = r.integers(0, K, m.sum()) if K > 2 else 1 - y[m]
    return y


def sea(seed, thresholds=(8.0, 9.0, 7.0, 9.5, 8.0), seg=5000, noise=0.05):
    r = np.random.default_rng(seed)
    X = r.uniform(0, 10, (seg * len(thresholds), 3))
    th = np.repeat(thresholds, seg)
    y = _flip((X[:, 0] + X[:, 1] <= th).astype(int), noise, r)
    changes = [seg * i for i in range(1, len(thresholds))]
    recur = [(seg * i, seg * j) for i in range(len(thresholds)) for j in range(i + 1, len(thresholds))
             if thresholds[i] == thresholds[j]]
    return dict(X=X / 10.0, y=y, K=2, changes=changes, recur=recur, tasks=None)


def hyperplane(seed, n=20000, d=10, mag=0.001, noise=0.05):
    r = np.random.default_rng(seed)
    w = r.uniform(0, 1, d); sigma = r.choice([-1, 1], d)
    X = r.uniform(0, 1, (n, d)); y = np.zeros(n, int)
    for i in range(n):
        y[i] = int(X[i] @ w >= 0.5 * w.sum())
        w += sigma * mag
        flip = r.random(d) < 0.001
        sigma[flip] *= -1
        w = np.clip(w, 0, None)
    return dict(X=X, y=_flip(y, noise, r), K=2, changes=[], recur=[], tasks=None)


def _digit_tasks(seed, pairs, passes, perms=None):
    X, y = load_digits(return_X_y=True); X = X / 16.0
    r = np.random.default_rng(seed)
    XS, YS, tests, changes, starts = [], [], [], [], []
    pos = 0
    for t, spec in enumerate(pairs):
        if perms is None:                                   # split digits: binary task on two digits
            m = np.isin(y, spec); Xt, yt = X[m], (y[m] == spec[1]).astype(int)
        else:                                               # permuted digits: all ten classes, permuted pixels
            Xt, yt = X[:, perms[spec]], y
        idx = r.permutation(len(yt)); cut = int(0.7 * len(yt))
        tr, te = idx[:cut], idx[cut:]
        tests.append((Xt[te], yt[te]))
        order = np.concatenate([r.permutation(tr) for _ in range(passes)])
        XS.append(Xt[order]); YS.append(yt[order])
        starts.append(pos); pos += len(order)
        if t > 0:
            changes.append(starts[-1])
    recur = [(starts[i], starts[j]) for i in range(len(pairs)) for j in range(i + 1, len(pairs)) if pairs[i] == pairs[j]]
    return dict(X=np.concatenate(XS), y=np.concatenate(YS), changes=changes, recur=recur, tasks=tests)


def splitdigits(seed):
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (0, 1)]
    s = _digit_tasks(seed, pairs, passes=4)
    s["K"] = 2; s["tasks"] = s["tasks"][:5]
    return s


# ------------------------------------------------------------------ sealed test streams
def stagger(seed, seg=5000, noise=0.0):
    r = np.random.default_rng(seed)
    n = seg * 4
    size, color, shape = r.integers(0, 3, n), r.integers(0, 3, n), r.integers(0, 3, n)
    concepts = [lambda s, c, h: (s == 0) & (c == 0),
                lambda s, c, h: (c == 1) | (h == 1),
                lambda s, c, h: (s == 1) | (s == 2),
                lambda s, c, h: (s == 0) & (c == 0)]
    y = np.concatenate([concepts[k](size[k*seg:(k+1)*seg], color[k*seg:(k+1)*seg], shape[k*seg:(k+1)*seg]) for k in range(4)]).astype(int)
    X = np.zeros((n, 9))
    X[np.arange(n), size] = 1; X[np.arange(n), 3 + color] = 1; X[np.arange(n), 6 + shape] = 1
    return dict(X=X, y=_flip(y, noise, r), K=2, changes=[seg, 2 * seg, 3 * seg], recur=[(0, 3 * seg)], tasks=None)


def rbfdrift(seed, n=20000, d=10, n_cent=50, speed=0.001, K=2):
    r = np.random.default_rng(seed)
    C = r.uniform(0, 1, (n_cent, d)); lab = r.integers(0, K, n_cent); wts = r.uniform(0, 1, n_cent)
    sd = r.uniform(0, 0.1, n_cent); vel = r.normal(0, 1, (n_cent, d)); vel /= np.linalg.norm(vel, axis=1, keepdims=True)
    X = np.zeros((n, d)); y = np.zeros(n, int)
    for i in range(n):
        k = r.choice(n_cent, p=wts / wts.sum())
        X[i] = C[k] + r.normal(0, 1, d) * sd[k]; y[i] = lab[k]
        C += vel * speed
        out = (C < 0) | (C > 1); vel[out] *= -1; C = np.clip(C, 0, 1)
    return dict(X=np.clip(X, -0.5, 1.5), y=y, K=K, changes=[], recur=[], tasks=None)


def permdigits(seed):
    r = np.random.default_rng(seed + 77)
    perms = [np.arange(64)] + [r.permutation(64) for _ in range(4)]
    s = _digit_tasks(seed, [0, 1, 2, 3, 4, 0], passes=2, perms=perms)
    s["K"] = 10; s["tasks"] = s["tasks"][:5]
    return s


def agrawal(seed, seg=5000, noise=0.05):
    r = np.random.default_rng(seed)
    n = seg * 4
    salary = r.uniform(20e3, 150e3, n); commission = np.where(salary >= 75e3, 0, r.uniform(10e3, 75e3, n))
    age = r.uniform(20, 80, n); elevel = r.integers(0, 5, n); car = r.integers(1, 21, n)
    zipc = r.integers(0, 9, n); hvalue = (9 - zipc) * 100e3 * r.uniform(0.5, 1.5, n)
    hyears = r.uniform(1, 30, n); loan = r.uniform(0, 500e3, n)
    f1 = (age < 40) | (age >= 60)
    f2 = ((age < 40) & (salary >= 50e3) & (salary <= 100e3)) | ((age >= 40) & (age < 60) & (salary >= 75e3) & (salary <= 125e3)) | \
         ((age >= 60) & (salary >= 25e3) & (salary <= 75e3))
    f3 = ((age < 40) & np.isin(elevel, [0, 1])) | ((age >= 40) & (age < 60) & np.isin(elevel, [1, 2, 3])) | ((age >= 60) & np.isin(elevel, [2, 3, 4]))
    seq = [f1, f2, f3, f1]
    y = np.concatenate([seq[k][k*seg:(k+1)*seg] for k in range(4)]).astype(int)
    X = np.column_stack([salary, commission, age, elevel, car, zipc, hvalue, hyears, loan])
    X = (X - X[:200].min(0)) / np.maximum(X[:200].max(0) - X[:200].min(0), 1e-12)   # scaled from the warm-up only
    return dict(X=np.clip(X, -0.5, 1.5), y=_flip(y, noise, r), K=2, changes=[seg, 2 * seg, 3 * seg], recur=[(0, 3 * seg)], tasks=None)


def seanoisy(seed):
    return sea(seed + 1000, thresholds=(7.5, 8.5, 9.5, 6.5, 7.5), noise=0.10)


def elec(seed):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=151, as_frame=False, parser="auto")
    X = d.data.astype(float); y = (d.target == np.unique(d.target)[1]).astype(int)
    X = (X - X[:200].min(0)) / np.maximum(X[:200].max(0) - X[:200].min(0), 1e-12)
    return dict(X=np.clip(X, -0.5, 1.5), y=y, K=2, changes=[], recur=[], tasks=None)


STREAMS = dict(sea=sea, hyperplane=hyperplane, splitdigits=splitdigits, stagger=stagger, rbfdrift=rbfdrift,
               permdigits=permdigits, agrawal=agrawal, seanoisy=seanoisy, elec=elec)


def get(name, seed):
    return STREAMS[name](seed)
