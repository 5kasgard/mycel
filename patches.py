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
