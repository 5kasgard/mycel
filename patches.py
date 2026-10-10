"""Runtime fixes applied by grids.py (kept separate so large modules need not be re-uploaded).
1. river: 'import river' does not import its submodules; import tree and forest explicitly."""
import learners_s


def _river_fixed():
    try:
        import river.tree, river.forest  # noqa
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "river"], check=False)
        import river.tree, river.forest  # noqa
    import river
    return river


learners_s._river = _river_fixed


# 2. elec: fetch_openml's default parser needs pandas, which the runner lacks. Install it and encode the
#    one nominal column (day of week) as integer codes; min-max scaling on the first 200 rows follows as before.
import streams as _streams
import numpy as _np


def _elec_fixed(seed):
    try:
        import pandas  # noqa
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas"], check=False)
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=151, as_frame=True, parser="auto")
    df = d.data.copy()
    for c in df.columns:
        if str(df[c].dtype) in ("category", "object"):
            df[c] = df[c].astype("category").cat.codes
    X = df.to_numpy(dtype=float); t = _np.asarray(d.target).astype(str)
    y = (t == _np.unique(t)[1]).astype(int)
    X = (X - X[:200].min(0)) / _np.maximum(X[:200].max(0) - X[:200].min(0), 1e-12)
    return dict(X=_np.clip(X, -0.5, 1.5), y=y, K=2, changes=[], recur=[], tasks=None)


_streams.STREAMS["elec"] = _elec_fixed


# 3. market scan: river forests reported a NaN tree height for some members, which made every ARF answer
#    energy NaN (and silently dropped ARF from the frontier). Read size attributes defensively, scale the
#    measured members up to the whole forest, and record what was measured (diag) with each result.
import learners_v2 as _lv2


def _get(t, name):
    for o in (t, getattr(t, "model", None)):
        try:
            v = float(getattr(o, name))
            if _np.isfinite(v):
                return v
        except Exception:
            pass
    return None


def _river_tiers(self):
    ms = self.m.models if hasattr(self.m, "models") else [self.m]
    hs = [h for h in (_get(t, "height") for t in ms) if h is not None]
    ls = [v for v in (_get(t, "n_leaves") for t in ms) if v is not None]
    ns = [v for v in (_get(t, "n_nodes") for t in ms) if v is not None]
    if hs: self.depth = max(1.0, float(_np.mean(hs)))
    if ls: self.leaves = max(float(self.trees), float(_np.sum(ls)) * len(ms) / len(ls))
    if ns: self.nodes = max(float(self.trees), float(_np.sum(ns)) * len(ms) / len(ns))
    self.diag = dict(depth=self.depth, leaves=self.leaves, nodes=self.nodes, measured=len(hs), trees=len(ms))
    self.tn = _lv2.cache_read_pJ(self.nodes * 64)
    self.tsx = _lv2.cache_read_pJ(self.leaves * self.K * self.d * 3 * 32)


_lv2.River2._tiers = _river_tiers


# 4. market scan robustness check (added AFTER the first scan results; only makes Mycel's job harder):
#    an integer multiclass perceptron, the natural cheap competitor for a multiplier-light learner.
#    Inputs quantised to int8 (standardised x * 16), weights int16, sparse-aware; learns only on mistakes.
class Perceptron16(_lv2.Base):
    def __init__(self, K, seed=0):
        self.K = K; self.cfg = {}; self.answer_pJ = 0.0; self.update_pJ = 0.0

    def start(self, Xw, K):
        self.d = Xw.shape[1]; self.mu, self.sd = _lv2._scaler(Xw)
        self.W = _np.zeros((self.d + 1, K), _np.int32)
        self.tw = _lv2.cache_read_pJ((self.d + 1) * K * 16)
        self.binary = _np.all((Xw == 0) | (Xw == 1), axis=0)

    def answer(self, x):
        q = _np.clip(_np.round((x - self.mu) / self.sd * 16), -127, 127).astype(_np.int32)
        q = _np.append(q, 16); nz = _np.flatnonzero(q); nb = int((~_np.append(self.binary, True))[nz].sum())
        s = q[nz] @ self.W[nz]
        mul = nb * self.K * 0.2                                          # int8 x int16 multiplies (binary inputs need none)
        self.answer_pJ += self.tw * _lv2.words(len(nz) * self.K * 16) + mul + len(nz) * self.K * 0.05 + 10 * _lv2.words(self.d * 8)
        self.n_ans += 1; self._c = (q, nz); self._p = int(_np.argmax(s)); return self._p

    def learn(self, x, y):
        self.n_upd += 1; self.update_pJ += self.K * 0.05
        if self._p == y: return
        q, nz = self._c
        self.W[nz, y] = _np.clip(self.W[nz, y] + q[nz], -32767, 32767)
        self.W[nz, self._p] = _np.clip(self.W[nz, self._p] - q[nz], -32767, 32767)
        self.update_pJ += 2 * (self.tw * _lv2.words(len(nz) * 16) * 2 + len(nz) * 0.05)

    def mem_bits(self):
        return int((self.d + 1) * self.K * 16)


_lv2.VARIANTS["perc16"] = (Perceptron16, {})
