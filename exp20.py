"""Round four test (pre-registered). Usage: python3 exp20.py <task> <seed_from> <seed_to>
Mycel-D setting chosen automatically per seed by 3-fold CV inside the training split.
Networks: int8 and pruned int8 at 2-64 hidden units, each given its best of three training
settings judged on the TEST set (oracle tuning: harsh to Mycel). Trees as context."""
import json, sys, copy, itertools, numpy as np
from tasks4 import task_data
from dmycel import MycelD, MycelCo
from baselines_d import MLPD
from protect import flip_mycel, flip_mlp, SCHEMES
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from dmycel import cache_read_pJ

RATES = (1e-3, 1e-2)
KILLS = (0.1, 0.3, 0.5)

def grid():
    for B, s in itertools.product((2, 4, 8), (2.0, 8.0)):
        for C, w in itertools.product((4, 8, 16), (0, 1)):
            yield dict(tag="perclass", B=B, s=s, C=C, w=w, T=(2 * C if w else None))
        for P, Tm in itertools.product((16, 32, 64, 128, 256), (0.5, 1.0)):
            yield dict(tag="shared", B=B, s=s, P=P, T=max(2, int(Tm * P)))

def build(cfg, K, seed, ep):
    if cfg["tag"] == "shared":
        return MycelCo(K, P=cfg["P"], B=cfg["B"], s=cfg["s"], T=cfg["T"], epochs=ep, seed=seed)
    return MycelD(K, C=cfg["C"], B=cfg["B"], s=cfg["s"], T=cfg["T"], weighted=cfg["w"], epochs=ep, seed=seed)

def epochs_for(n):
    return int(np.clip(20000 // max(n, 1), 40, 300))     # fewer passes on bigger training sets

def select(X, y, K, seed, logic):
    """Automatic: 3-fold CV inside the training split; cheapest setting within 1 point of the best."""
    ep = epochs_for(len(y))
    idx = np.random.default_rng(seed + 555).permutation(len(y)); parts = np.array_split(idx, 3)
    res = []
    for cfg in grid():
        if logic and cfg["B"] != 2: continue             # binary inputs: thresholds are irrelevant
        accs, ens = [], []
        for f in range(3):
            te = parts[f]; tr = np.setdiff1d(idx, te)
            if logic and len(y) <= 32:                    # whole truth table: fit and score on it
                tr = te = idx
            m = build(cfg, K, seed, ep).fit(X[tr], y[tr])
            accs.append(np.mean(m.predict(X[te]) == y[te])); ens.append(m.energy())
            if logic and len(y) <= 32: break
        res.append((float(np.mean(accs)), float(np.mean(ens)), cfg))
    best = max(r[0] for r in res)
    ok = [r for r in res if r[0] >= best - 0.01]
    return min(ok, key=lambda r: r[1])[2], ep

def tree_energy(clf, X, K, d):
    """Per answer: each tree visited node is one word read + a compare; leaf: class scores."""
    trees = clf.estimators_ if hasattr(clf, "estimators_") else [clf]
    nodes = sum(t.tree_.node_count for t in trees); bits = nodes * 32 + (len(trees) > 1) * nodes * K * 8
    tier = cache_read_pJ(bits)
    depth = np.mean([t.decision_path(X).sum(1).mean() for t in trees]) * len(trees)
    leaf = len(trees) * (np.ceil(K * 8 / 64) if len(trees) > 1 else 0)
    return float(tier * (depth + leaf) + 0.1 * depth + 0.1 * K * len(trees) + 10 * np.ceil(d * 8 / 64)), int(bits)

def damage(m, kind, Xtr, ytr, Xte, yte, seed):
    acc = lambda z: float(np.mean(z.predict(Xte) == yte))
    out = dict(acc=acc(m), energy=float(m.energy()), bits=int(m.model_bits()), flip={}, kill={})
    flipper = flip_mycel if kind == "mycel" else flip_mlp
    for rate in RATES:
        best = None
        for mode, pol in SCHEMES:
            a = []
            for rep in range(3):
                d = copy.deepcopy(m); flipper(d, rate, seed * 100 + rep, mode, pol); a.append(acc(d))
            v = float(np.mean(a)); out["flip"][f"{rate}|{mode}|{pol}"] = v
            if best is None or v > best: best = v
        out["flip"][str(rate)] = best
    for frac in KILLS:
        d = copy.deepcopy(m); d.kill(frac, seed * 7 + int(frac * 100))
        hit = acc(d); d.retrain(Xtr, ytr, 20)
        out["kill"][str(frac)] = dict(after=hit, recovered=acc(d))
    return out

if __name__ == "__main__":
    task, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = []
    for seed in range(a, b):
        Xtr, ytr, Xte, yte, K = task_data(task, seed); d = Xtr.shape[1]
        logic = task.startswith("parity") or task == "logic8"
        cfg, ep = select(Xtr, ytr, K, seed, logic)
        rec = dict(task=task, seed=seed, cfg=cfg)
        m = build(cfg, K, seed, ep).fit(Xtr, ytr)
        rec["mycel"] = damage(m, "mycel", Xtr, ytr, Xte, yte, seed)
        big = len(ytr) > 1000
        for h in (2, 4, 8, 16, 32, 64):
            for pr in (0.0, 0.8):
                best = None
                for e_, lr in (((100, 0.01), (300, 0.01), (300, 0.03)) if big else ((300, 0.01), (1500, 0.01), (1500, 0.03))):
                    n = MLPD(d, K, h=h, bits=8, prune=pr, epochs=e_, lr=lr, seed=seed).fit(Xtr, ytr)
                    acc = np.mean(n.predict(Xte) == yte)
                    if best is None or acc > best[0]: best = (acc, n, e_, lr)
                r = damage(best[1], "mlp", Xtr, ytr, Xte, yte, seed); r["train"] = best[2:]
                rec[f"mlp_h{h}_p{pr}"] = r
        for depth in (2, 4, 6, 8, 10, 12):
            t = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(Xtr, ytr)
            e, bits = tree_energy(t, Xte, K, d); rec[f"tree_d{depth}"] = dict(acc=float(np.mean(t.predict(Xte) == yte)), energy=e, bits=bits)
        for depth in (4, 6, 8):
            t = RandomForestClassifier(10, max_depth=depth, random_state=seed).fit(Xtr, ytr)
            e, bits = tree_energy(t, Xte, K, d); rec[f"forest10_d{depth}"] = dict(acc=float(np.mean(t.predict(Xte) == yte)), energy=e, bits=bits)
        print(json.dumps(dict(task=task, seed=seed, cfg=cfg, mycel=(round(rec["mycel"]["acc"], 3), round(rec["mycel"]["energy"])),
                              h8=(round(rec["mlp_h8_p0.0"]["acc"], 3), round(rec["mlp_h8_p0.0"]["energy"])))), flush=True)
        out.append(rec)
        json.dump(out, open(f"results_r4_{task}_s{a}.json", "w"), indent=1)
