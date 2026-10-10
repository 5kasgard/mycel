"""Market scan analysis, written before any scan result existed (pre-registered).
Usage: python scan_analyze.py '<glob of results_scan_*.json>' out.json
Per task: mean over seeds of accuracy, energy per example (answer + update) and memory, per variant.
Baseline frontier F(E) = best baseline accuracy among baseline variants with energy <= E (majority included).
Because the energy model is an estimate, the primary measure gives baselines 1.5x slack:
Mycel profit = max over Mycel variants of [acc - F(1.5 * E_mycel)]; Mycel 'sells' on a task if profit > 0.
profit_strict (no slack) is reported alongside.
H1 (decomposition): on the confirmation half, mean profit of 'decomp' tasks exceeds 'other' tasks
    (one-sided permutation test, p < 0.05) AND the difference has the same sign on the discovery half.
H2 (symbolic input): same test with frac_binary >= 0.5 vs < 0.5.
H3 (trait model): on the discovery half, rank traits by Spearman correlation with profit; take the two
    strongest as a signed z-score sum; confirmed if Spearman(score, profit) on the confirmation half is > 0
    with one-sided p < 0.05 (permutation)."""
import glob, json, sys, collections, numpy as np

MY = ("mycelT", "mycelS", "mycelA", "mycelL")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 and np.std(ra) > 0 and np.std(rb) > 0 else 0.0


def perm_p(x, y, stat, n=20000, seed=0):
    r = np.random.default_rng(seed); obs = stat(x, y); z = np.concatenate([x, y]); k = len(x); c = 0
    for _ in range(n):
        r.shuffle(z); c += stat(z[:k], z[k:]) >= obs
    return obs, (c + 1) / (n + 1)


def main():
    recs = [r for f in sorted(glob.glob(sys.argv[1])) for r in json.load(open(f))]
    by = collections.defaultdict(list)
    for r in recs: by[(r["task"], r["variant"])].append(r)
    tasks = sorted({t for t, _ in by})
    rows = {}
    for (t, v), rs in by.items():
        rows[(t, v)] = dict(acc=np.mean([r["acc"] for r in rs]), E=np.mean([r["pJ_answer"] + r["pJ_update"] for r in rs]),
                            mem=np.mean([r["mem_bits"] for r in rs]), n=len(rs), sd=float(np.std([r["acc"] for r in rs])))
    out = {}
    for t in tasks:
        r0 = next(r for (tt, v), rs in by.items() if tt == t for r in rs)
        base = {v: x for (tt, v), x in rows.items() if tt == t and v not in MY}
        mys = {v: x for (tt, v), x in rows.items() if tt == t and v in MY}
        if not base or not mys: continue
        best_b = max(base.values(), key=lambda x: x["acc"])
        def F(E): return max([x["acc"] for x in base.values() if x["E"] <= E] or [0.0])
        prof = {v: float(x["acc"] - F(1.5 * x["E"])) for v, x in mys.items()}
        prof_strict = max(float(x["acc"] - F(x["E"])) for x in mys.values())
        vbest = max(prof, key=prof.get); m = mys[vbest]
        reach = [x["E"] for x in base.values() if x["acc"] >= m["acc"]]
        dom3 = [v for v, x in base.items() if x["acc"] >= m["acc"] and x["E"] <= m["E"] and x["mem"] <= m["mem"]]
        cheapest_b = min(base.items(), key=lambda kv: kv[1]["E"] if kv[0] != "majority" else 1e30)
        out[t] = dict(tag=r0["tag"], half=r0["half"], traits={k: r0[k] for k in ("n", "d", "K", "frac_binary", "density", "majority", "drift")},
                      profit=prof[vbest], profit_strict=prof_strict, best_mycel=vbest, mycel_acc=float(m["acc"]), mycel_E=float(m["E"]),
                      energy_ratio=float(m["E"] / min(reach)) if reach else None, best_baseline=[k for k, x in base.items() if x is best_b][0],
                      best_acc=float(best_b["acc"]), best_E=float(best_b["E"]), acc_gap_to_best=float(max(x["acc"] for x in mys.values()) - best_b["acc"]),
                      sells=bool(prof[vbest] > 0), sells_3d=not dom3, linear_gap=float(best_b["acc"] - base.get("logistic", best_b)["acc"]),
                      local_gap=float(best_b["acc"] - base.get("knn", best_b)["acc"]), n_baselines=len(base),
                      variants={v: dict(acc=round(float(x["acc"]), 4), E=round(float(x["E"]), 1), mem=int(x["mem"]), sd=round(x["sd"], 4)) for (tt, v), x in rows.items() if tt == t})
    res = dict(tasks=out)
    def test(group_fn, name):
        o = {}
        for h in ("discovery", "confirm"):
            a = np.array([x["profit"] for x in out.values() if x["half"] == h and group_fn(x)])
            b = np.array([x["profit"] for x in out.values() if x["half"] == h and not group_fn(x)])
            if len(a) and len(b):
                d, p = perm_p(a, b, lambda u, w: u.mean() - w.mean())
                o[h] = dict(n_in=len(a), n_out=len(b), mean_in=float(a.mean()), mean_out=float(b.mean()), diff=float(d), p=float(p))
        o["supported"] = bool(o.get("confirm", {}).get("p", 1) < 0.05 and o.get("discovery", {}).get("diff", -1) > 0 and o.get("confirm", {}).get("diff", -1) > 0)
        res[name] = o
    test(lambda x: x["tag"] == "decomp", "H1_decomposition")
    test(lambda x: x["traits"]["frac_binary"] >= 0.5, "H2_symbolic")
    def feats(x):
        tr = x["traits"]
        return dict(log_d=np.log(tr["d"]), log_K=np.log(tr["K"]), log_n=np.log(tr["n"]), frac_binary=tr["frac_binary"],
                    density=tr["density"], majority=tr["majority"], drift=tr["drift"], linear_gap=x["linear_gap"], local_gap=x["local_gap"])
    disc = [x for x in out.values() if x["half"] == "discovery"]; conf = [x for x in out.values() if x["half"] == "confirm"]
    if len(disc) > 4 and len(conf) > 4:
        names = list(feats(disc[0]))
        cors = {k: spearman([feats(x)[k] for x in disc], [x["profit"] for x in disc]) for k in names}
        top = sorted(names, key=lambda k: -abs(cors[k]))[:2]
        allv = {k: np.array([feats(x)[k] for x in disc]) for k in top}
        def score(x): return sum(np.sign(cors[k]) * (feats(x)[k] - allv[k].mean()) / (allv[k].std() + 1e-12) for k in top)
        sc = np.array([score(x) for x in conf]); pr = np.array([x["profit"] for x in conf]); obs = spearman(sc, pr)
        r = np.random.default_rng(1); c = sum(spearman(sc, r.permutation(pr)) >= obs for _ in range(5000))
        res["H3_trait_model"] = dict(discovery_correlations=cors, chosen=top, confirm_spearman=obs, p=(c + 1) / 5001, supported=bool((c + 1) / 5001 < 0.05 and obs > 0))
    res["market"] = sorted([(t, round(x["profit"], 4), x["best_mycel"], x["half"]) for t, x in out.items() if x["sells"]], key=lambda z: -z[1])
    res["summary"] = dict(tasks=len(out), sells=sum(x["sells"] for x in out.values()), sells_3d=sum(x["sells_3d"] for x in out.values()), sells_strict=sum(x["profit_strict"] > 0 for x in out.values()),
                          median_profit=float(np.median([x["profit"] for x in out.values()])) if out else None)
    json.dump(res, open(sys.argv[2], "w"), indent=1, default=float)
    print(json.dumps(res["summary"]), json.dumps({k: res[k].get("supported") for k in res if k.startswith("H")}))
    for t, x in sorted(out.items(), key=lambda kv: -kv[1]["profit"]):
        print(f"{t:22s} {x['tag']:6s} {x['half']:9s} profit {x['profit']:+.4f} {x['best_mycel']:7s} acc {x['mycel_acc']:.3f}@{x['mycel_E']:.0f}  best {x['best_baseline']} {x['best_acc']:.3f}@{x['best_E']:.0f}")


if __name__ == "__main__":
    main()
