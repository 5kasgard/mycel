"""Round five: summarise development (or test) results and apply the pre-registered selection/win rules.
Usage: python r5_summary.py <glob of result files> <out.json> [frozen.json]
Without frozen.json: development mode. Picks one setting per baseline family (best mean accuracy over
the streams) and the Mycel setting with the most wins under the win rule (ties: mean accuracy).
With frozen.json: scores the frozen settings on whatever streams the files contain."""
import json, glob, sys, collections, numpy as np


def key(c):
    return json.dumps(c, sort_keys=True)


def load(pattern):
    agg = collections.defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        for r in json.load(open(f)):
            agg[(r["stream"], r["family"], key(r["cfg"]))].append(r)
    rows = {}
    for (st, fam, k), rs in agg.items():
        ret = [np.mean(r["retention"]) for r in rs if r.get("retention")]
        rel = [np.mean(r["relearn"]) for r in rs if r.get("relearn")]
        rows[(st, fam, k)] = dict(acc=float(np.mean([r["acc"] for r in rs])), E=float(np.mean([r["pJ_answer"] + r["pJ_update"] for r in rs])),
                                  Ea=float(np.mean([r["pJ_answer"] for r in rs])), rec=float(np.mean([r["recovery"] for r in rs])) if rs[0]["recovery"] is not None else None,
                                  relearn=float(np.mean(rel)) if rel else None, retention=float(np.mean(ret)) if ret else None,
                                  mem=float(np.mean([r["mem_bits"] for r in rs])), n=len(rs))
    return rows


def wins(m, b):
    return (m["acc"] >= b["acc"] - 0.01 and m["E"] <= 0.5 * b["E"]) or (m["acc"] >= b["acc"] + 0.02 and m["E"] <= b["E"])


def dominated(m, others):
    """Sweep 3 rule (harsher): no selected baseline may be both at least as accurate and cheaper."""
    return any(o["acc"] >= m["acc"] and o["E"] < m["E"] for o in others)


def main():
    rows = load(sys.argv[1]); out_path = sys.argv[2]
    streams = sorted({k[0] for k in rows}); fams = sorted({k[1] for k in rows})
    frozen = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    if frozen is None:
        frozen = {}
        for fam in fams:
            cfgs = {k[2] for k in rows if k[1] == fam}
            full = [c for c in cfgs if all((st, fam, c) in rows for st in streams)]
            if fam == "mycel" or not full:
                continue
            frozen[fam] = max(full, key=lambda c: np.mean([rows[(st, fam, c)]["acc"] for st in streams]))
        best_b = {st: max((rows[(st, f, c)] | dict(fam=f) for f, c in frozen.items()), key=lambda r: r["acc"]) for st in streams}
        cands = [c for c in {k[2] for k in rows if k[1] == "mycel"} if all((st, "mycel", c) in rows for st in streams)]
        sel = {st: [rows[(st, f, c)] for f, c in frozen.items() if f != "majority" and (st, f, c) in rows] for st in streams}
        def score(c):
            w = sum(wins(rows[(st, "mycel", c)], best_b[st]) and not dominated(rows[(st, "mycel", c)], sel[st]) for st in streams)
            return (w, np.mean([rows[(st, "mycel", c)]["acc"] for st in streams]))
        ranked = sorted(cands, key=score, reverse=True)
        frozen["mycel"] = ranked[0]
        top = [dict(cfg=json.loads(c), wins=score(c)[0], acc=score(c)[1],
                    per_stream={st: dict(acc=round(rows[(st, 'mycel', c)]['acc'], 4), E=round(rows[(st, 'mycel', c)]['E'])) for st in streams})
               for c in ranked[:8]]
        most_acc = max(cands, key=lambda c: np.mean([rows[(st, "mycel", c)]["acc"] for st in streams]))
    else:
        top, most_acc = None, None
        frozen = {f: key(c) if isinstance(c, dict) else c for f, c in frozen.items()}
    table = {st: {f: dict(rows[(st, f, c)], cfg=json.loads(c)) for f, c in frozen.items() if (st, f, c) in rows} for st in streams}
    verdict = {}
    for st in streams:
        base = {f: r for f, r in table[st].items() if f not in ("mycel", "majority")}
        if "mycel" not in table[st] or not base: continue
        b = max(base.values(), key=lambda r: r["acc"]); m = table[st]["mycel"]
        dom = [f for f, r in base.items() if r["acc"] >= m["acc"] and r["E"] < m["E"]]
        anycfg = sum(1 for (s2, f, c), r in rows.items() if s2 == st and f not in ("mycel", "majority") and r["acc"] >= m["acc"] and r["E"] < m["E"])
        verdict[st] = dict(win=bool(wins(m, b)) and not dom, win_rule=bool(wins(m, b)), dominated_by=dom,
                           any_baseline_cfg_dominating=anycfg, best_baseline=[f for f, r in base.items() if r is b][0])
    res = dict(frozen={f: json.loads(c) for f, c in frozen.items()}, table=table, verdict=verdict, mycel_top=top,
               mycel_most_accurate=json.loads(most_acc) if most_acc else None, n_configs={f: len({k[2] for k in rows if k[1] == f}) for f in fams})
    json.dump(res, open(out_path, "w"), indent=1)
    for st in streams:
        print(st, {f: (round(r["acc"], 3), round(r["E"])) for f, r in table[st].items()}, verdict.get(st))


if __name__ == "__main__":
    main()
