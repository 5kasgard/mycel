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
