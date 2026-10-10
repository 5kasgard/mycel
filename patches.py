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
