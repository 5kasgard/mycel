"""Confirmation pool for the market scan (registered 10 Oct 2026, before any of these tasks was run).
Tags keep the first scan's discovery/confirmation split untouched:
  c2_bin  = new two-class tasks with <= 100 features            (inside the profile)
  c2_out  = new tasks with more than two classes or > 100 features (outside the profile)
  c2_sig  = new synthetic signal-decomposition tasks (pull one source out of a mixed signal)
Profile under test (from the first scan): Mycel is a useful seller only on two-class tasks with <= 100 features."""
import numpy as np
import scan_tasks as T


def _mix(seed, draws, p_on, p_off, target, n=20000, win=8, drift=0.0, noise=20.0):
    r = np.random.default_rng(seed)
    draws = np.asarray(draws, float); k = len(draws); on = np.zeros(k, bool); agg = np.zeros(n); y = np.zeros(n, int)
    for i in range(n):
        if drift: draws = draws * (1 + drift * r.normal(0, 1, k))            # appliances age / swap
        flip = np.where(on, r.random(k) < p_off, r.random(k) < p_on); on ^= flip
        agg[i] = (draws * on * r.normal(1, 0.05, k)).sum() + r.normal(0, noise); y[i] = on[target]
    X = np.zeros((n, 2 * win))
    for j in range(win):
        X[j:, j] = agg[: n - j] if j else agg
    X[:, win:] = X[:, :win] - np.roll(X[:, :win], -1, axis=1)
    X = X / max(1.0, np.abs(X[:200]).max())
    return dict(X=X, y=y, K=2, changes=[], recur=[], tasks=None)


def sig_heater(seed):          # ten appliances, target a large long-running load
    return _mix(seed, [150, 2000, 800, 60, 1200, 300, 3000, 90, 450, 700],
                [.05, .003, .01, .02, .004, .008, .002, .03, .01, .006], [.05, .05, .03, .01, .08, .02, .01, .05, .04, .03], target=6)


def sig_small(seed):           # target a 60 W load hidden among big ones
    return _mix(seed, [150, 2000, 800, 60, 1200, 300.0], [.05, .003, .01, .02, .004, .008], [.05, .05, .03, .01, .08, .02], target=3, noise=10)


def sig_drift(seed):           # appliance draws wander over time
    return _mix(seed + 7, [150, 2000, 800, 60, 1200, 300.0], [.05, .003, .01, .02, .004, .008], [.05, .05, .03, .01, .08, .02],
                target=0, drift=0.0005)


def sig_spikes(seed, n=20000, win=16):
    """Two neurons on one electrode: does neuron A start a spike within one sample of the window's centre?"""
    r = np.random.default_rng(seed)
    tA = np.array([0, -.3, -1, -.6, .4, .7, .4, .1]); tB = np.array([0, .5, .9, .2, -.5, -.6, -.2, 0])
    sig = r.normal(0, 0.15, n + 64); a = r.random(n + 64) < 0.12; b = r.random(n + 64) < 0.12
    for i in np.flatnonzero(a)[:-1]: sig[i:i + 8] += tA[: len(sig[i:i + 8])]
    for i in np.flatnonzero(b)[:-1]: sig[i:i + 8] += tB[: len(sig[i:i + 8])] * 0.9
    X = np.stack([sig[i:i + win] for i in range(n)]); y = (a[win // 2 - 1: win // 2 - 1 + n] | a[win // 2: win // 2 + n] | a[win // 2 + 1: win // 2 + 1 + n]).astype(int)
    return dict(X=X, y=y, K=2, changes=[], recur=[], tasks=None)


BIN = {"pc4": 1049, "pc3": 1050, "kc2": 1063, "pc1": 1068, "qsar": 1494, "ozone": 1487, "wilt": 40983,
       "blood": 1464, "wdbc": 1510, "banknote": 1462, "ilpd": 1480, "churn": 40701, "mozilla4": 1046, "jm1": 1053,
       "steel-bin": 1504, "credit-approval": 29, "breast-w": 15, "hill-valley": 1479, "climate": 1467,
       "twonorm": 1507, "ringnorm": 1496, "banana": 1460, "numerai": 23517}
OUT = {"wall-robot": 1497, "first-order": 1475, "dna": 40670, "eucalyptus": 188, "texture": 40499, "cmc": 23,
       "yeast": 181, "mice": 40966, "gesture": 4538, "vowel": 307, "jungle": 41027, "semeion": 1501,
       "mfeat-factors": 12, "madelon": 1485, "internet-ads": 40978}

for nm, did in BIN.items():
    T._reg("c2_" + nm, T.openml(did), "c2_bin", False)
for nm, did in OUT.items():
    T._reg("c2_" + nm, T.openml(did), "c2_out", False)
for nm, fn in {"sig_heater": sig_heater, "sig_small": sig_small, "sig_drift": sig_drift, "sig_spikes": sig_spikes}.items():
    T._reg("c2_" + nm, fn, "c2_sig", True)
