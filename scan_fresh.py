"""Grow-on-demand test on fresh real data (written before any c3 task ran).
Usage: python scan_fresh.py '<glob of results_scan_c3_*.json>' out.json
Products: FULL = full Mycel (mycelT/S/A/L); GROW = grow-on-demand Mycel (mycelGc2/Gc3); PORTS = Mycel's
thermometer ports + linear readout (thperc16/thlog). A material sale for a product: some variant has
profit >= 0.005 against every learner outside that product (1.5x energy slack) AND accuracy within 1 point
of the best learner on the task.
KILL criterion: if FULL and GROW together make fewer than 3 material sales on the fresh pool, the Mycel
classifier line stops (the ports may continue on their own if PORTS sells).
Also reported: GROW vs PORTS accuracy and energy (does growing nuclei add anything over the ports?)."""
import glob, json, sys, collections, numpy as np

PRODUCTS = {"FULL": ("mycelT", "mycelS", "mycelA", "mycelL"), "GROW": ("mycelGc2", "mycelGc3"), "PORTS": ("thperc16", "thlog")}


def main():
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(sys.argv[1])):
        for r in json.load(open(f)): by[(r["task"], r["variant"])].append(r)
    need = set(sum(PRODUCTS.values(), ())) | {"arf20", "arf5", "hat", "perc16", "logistic", "knn", "mlp_adam128", "majority"}
    out, excluded = {}, []
    for t in sorted({t for t, _ in by}):
        rows = {v: dict(acc=float(np.mean([r["acc"] for r in by[(t, v)]])), E=float(np.mean([r["pJ_answer"] + r["pJ_update"] for r in by[(t, v)]])))
                for (tt, v) in by if tt == t}
        if not need <= set(rows) or not all(np.isfinite(x["E"]) for x in rows.values()):
            excluded.append((t, sorted(need - set(rows)))); continue
        best = max(x["acc"] for x in rows.values()); r0 = by[(t, "majority")][0]; o = dict(K=r0["K"], d=r0["d"], best=best)
        for name, vs in PRODUCTS.items():
            base = {v: x for v, x in rows.items() if v not in vs}
            def F(E): return max([x["acc"] for x in base.values() if x["E"] <= E] or [0.0])
            o[name] = [v for v in vs if rows[v]["acc"] - F(1.5 * rows[v]["E"]) >= 0.005 and rows[v]["acc"] >= best - 0.01]
        g = max(PRODUCTS["GROW"], key=lambda v: rows[v]["acc"]); p = max(PRODUCTS["PORTS"], key=lambda v: rows[v]["acc"])
        f = max(PRODUCTS["FULL"], key=lambda v: rows[v]["acc"])
        o["grow_minus_ports"] = rows[g]["acc"] - rows[p]["acc"]; o["full_minus_ports"] = rows[f]["acc"] - rows[p]["acc"]
        o["rows"] = {v: dict(acc=round(x["acc"], 4), E=round(x["E"])) for v, x in rows.items()}
        out[t] = o
    sales = {n: sorted(t for t, o in out.items() if o[n]) for n in PRODUCTS}
    mycel = sorted(set(sales["FULL"]) | set(sales["GROW"]))
    res = dict(tasks=out, excluded=excluded, sales=sales, mycel_sales=mycel, kill=len(mycel) < 3,
               median_grow_minus_ports=float(np.median([o["grow_minus_ports"] for o in out.values()])) if out else None,
               median_full_minus_ports=float(np.median([o["full_minus_ports"] for o in out.values()])) if out else None)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    print("tasks", len(out), "excluded", excluded)
    for n in PRODUCTS: print(n, len(sales[n]), sales[n])
    print("Mycel family sales:", len(mycel), "KILL" if res["kill"] else "CONTINUE")
    print("median acc gain over ports: grow %+.4f  full %+.4f" % (res["median_grow_minus_ports"], res["median_full_minus_ports"]))
    for t, o in sorted(out.items()):
        rr = o["rows"]; print(f"{t:20s} K={o['K']:<3d} d={o['d']:<5d} best {o['best']:.3f}  full {max(rr[v]['acc'] for v in PRODUCTS['FULL']):.3f} grow {max(rr[v]['acc'] for v in PRODUCTS['GROW']):.3f} ports {max(rr[v]['acc'] for v in PRODUCTS['PORTS']):.3f}  sales F{len(o['FULL'])} G{len(o['GROW'])} P{len(o['PORTS'])}")


if __name__ == "__main__":
    main()
