"""Parts test (registered before the E runs): which part of Mycel sells?
Usage: python scan_parts.py '<glob>' out.json
Sale (stricter than before, per the audit): some variant of the product has profit >= 0.005 against every
other learner's frontier (1.5x energy slack) AND accuracy within 1 point of the best learner on the task.
Products compared: Mycel (any of T/S/A/L) vs thermometer-linear (thperc16/thlog: Mycel's ports + a linear learner).
P1: Mycel's sales with thermometer-linear learners on the menu (count; expected: only conjunction-type tasks).
P2: thermometer-linear's own sales against the original menu (does the port part sell by itself?)."""
import glob, json, sys, collections, numpy as np

MY = ("mycelT", "mycelS", "mycelA", "mycelL"); TH = ("thperc16", "thlog")


def sales(rows, prod, exclude=()):
    base = {v: x for v, x in rows.items() if v not in prod and v not in exclude}
    best = max(x["acc"] for v, x in rows.items() if v not in exclude)
    def F(E): return max([x["acc"] for x in base.values() if x["E"] <= E] or [0.0])
    return [v for v in prod if v in rows and rows[v]["acc"] - F(1.5 * rows[v]["E"]) >= 0.005 and rows[v]["acc"] >= best - 0.01]


def main():
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(sys.argv[1])):
        for r in json.load(open(f)): by[(r["task"], r["variant"])].append(r)
    tasks = sorted({t for t, _ in by}); out = {}
    for t in tasks:
        rows = {v: dict(acc=np.mean([r["acc"] for r in by[(t, v)]]), E=np.mean([r["pJ_answer"] + r["pJ_update"] for r in by[(t, v)]]))
                for (tt, v) in by if tt == t}
        if not all(np.isfinite(x["E"]) for x in rows.values()) or not all(v in rows for v in MY + TH + ("arf20", "perc16")):
            continue
        r0 = by[(t, "majority")][0]
        out[t] = dict(K=r0["K"], d=r0["d"], mycel_sales_full=sales(rows, MY), mycel_sales_noTH=sales(rows, MY, exclude=TH),
                      th_sales_vs_original=sales(rows, TH, exclude=MY),
                      acc={v: round(float(x["acc"]), 4) for v, x in rows.items()}, E={v: round(float(x["E"])) for v, x in rows.items()})
    res = dict(tasks=out, P1=sorted(t for t, x in out.items() if x["mycel_sales_full"]),
               mycel_without_TH=sorted(t for t, x in out.items() if x["mycel_sales_noTH"]),
               P2=sorted(t for t, x in out.items() if x["th_sales_vs_original"]), n=len(out))
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    print("tasks", len(out)); print("Mycel sales, original menu:", res["mycel_without_TH"]); print("P1 Mycel sales with thermo-linear on menu:", res["P1"])
    print("P2 thermo-linear sales vs original menu:", len(res["P2"]), res["P2"])


if __name__ == "__main__":
    main()
