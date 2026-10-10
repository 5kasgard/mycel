"""Confirmation of the market-scan profile (written before any confirmation task was run).
Usage: python scan_confirm.py '<glob of results_scan_c2_*.json>' out.json
Competitors: all 15 non-Mycel variants (including the integer perceptron). Profit as in scan_analyze.py
(1.5x energy slack). A task is a USEFUL sale when profit > 0 AND Mycel's best accuracy is within 1 point
of the best accuracy any learner reached on that task.
Profile: K == 2 and d <= 100, measured on the loaded data (not taken from the tag).
C1: among tasks outside the profile, useful sales <= 10%.
C2: inside the profile, at least 3 useful sales, and the inside rate exceeds the outside rate
    (one-sided Fisher exact test, p < 0.10).
C3: Mycel is a useful seller on at least 2 of the 4 synthetic signal-decomposition tasks.
Tasks whose data failed to load, or that miss any learner, are excluded and listed."""
import glob, json, sys, collections, numpy as np
from math import comb

MY = ("mycelT", "mycelS", "mycelA", "mycelL")
ALL = ("mycelT", "mycelS", "mycelA", "mycelL", "mlp_adam128", "mlp_sgd32", "logistic", "gnb", "knn", "htree",
       "majority", "hat", "arf5", "arf20", "perc16")


def fisher_one_sided(a, b, c, d):
    """P(X >= a) for the table [[a, b], [c, d]] with fixed margins (hypergeometric)."""
    n1, n2, k = a + b, c + d, a + c
    tot = comb(n1 + n2, k)
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot


def main():
    recs = [r for f in sorted(glob.glob(sys.argv[1])) for r in json.load(open(f))]
    by = collections.defaultdict(list)
    for r in recs: by[(r["task"], r["variant"])].append(r)
    tasks = sorted({t for t, _ in by}); out, excluded = {}, []
    for t in tasks:
        rows = {v: dict(acc=np.mean([r["acc"] for r in by[(t, v)]]), E=np.mean([r["pJ_answer"] + r["pJ_update"] for r in by[(t, v)]]))
                for v in ALL if (t, v) in by}
        if len(rows) < len(ALL) or not all(np.isfinite(x["E"]) for x in rows.values()):
            excluded.append((t, sorted(set(ALL) - set(rows)))); continue
        r0 = by[(t, "majority")][0]
        base = {v: x for v, x in rows.items() if v not in MY}; mys = {v: x for v, x in rows.items() if v in MY}
        def F(E): return max([x["acc"] for x in base.values() if x["E"] <= E] or [0.0])
        prof = {v: x["acc"] - F(1.5 * x["E"]) for v, x in mys.items()}; vb = max(prof, key=prof.get)
        best = max(x["acc"] for x in rows.values())
        useful = bool(prof[vb] > 0 and mys[vb]["acc"] >= best - 0.01)
        inside = r0["K"] == 2 and r0["d"] <= 100
        out[t] = dict(tag=r0["tag"], K=r0["K"], d=r0["d"], inside=inside, profit=float(prof[vb]), best_mycel=vb,
                      mycel_acc=float(mys[vb]["acc"]), mycel_E=float(mys[vb]["E"]), best_acc=float(best),
                      best_learner=max(rows, key=lambda v: rows[v]["acc"]), useful=useful,
                      variants={v: dict(acc=round(float(x["acc"]), 4), E=round(float(x["E"]), 1)) for v, x in rows.items()})
    ins = [x for x in out.values() if x["inside"]]; outs = [x for x in out.values() if not x["inside"]]
    a = sum(x["useful"] for x in ins); b = len(ins) - a; c = sum(x["useful"] for x in outs); d = len(outs) - c
    p = fisher_one_sided(a, b, c, d) if ins and outs else 1.0
    sig = [x for t, x in out.items() if x["tag"] == "c2_sig"]
    res = dict(tasks=out, excluded=excluded,
               C1=dict(outside=len(outs), useful=c, rate=c / max(1, len(outs)), passed=bool(len(outs) and c / len(outs) <= 0.10)),
               C2=dict(inside=len(ins), useful=a, rate=a / max(1, len(ins)), fisher_p=p, passed=bool(a >= 3 and p < 0.10)),
               C3=dict(signal_tasks=len(sig), useful=sum(x["useful"] for x in sig), passed=bool(sum(x["useful"] for x in sig) >= 2)))
    json.dump(res, open(sys.argv[2], "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("C1", "C2", "C3")}), "excluded:", excluded)
    for t, x in sorted(out.items(), key=lambda kv: (-kv[1]["inside"], -kv[1]["profit"])):
        print(f"{t:22s} {'IN ' if x['inside'] else 'out'} K={x['K']:<3d} d={x['d']:<5d} profit {x['profit']:+.4f} "
              f"{'USEFUL' if x['useful'] else '      '} {x['best_mycel']:7s} {x['mycel_acc']:.3f}@{x['mycel_E']:.0f}  best {x['best_learner']} {x['best_acc']:.3f}")


if __name__ == "__main__":
    main()
