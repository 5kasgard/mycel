"""Round five runner: one stream, one learner family, a list of settings, a list of seeds.
Usage: python exp22.py <stream> <family> <settings-json-or-@file-or-grid:spec> <seed_from> <seed_to> [tag]
Writes results_r5_<tag>.json: one compact record per (setting, seed)."""
import json, sys, time, numpy as np
import streams
import patches  # noqa: runtime fixes (river submodules, elec loader)
from learners_s import OnlineMycel, OnlineMLP, WindowKNN, HTree, Majority, RiverLearner

FAMILIES = dict(mycel=OnlineMycel, mlp=OnlineMLP, knn=WindowKNN, htree=HTree, majority=Majority, river=RiverLearner)
WARM, WIN = 200, 500


def run(stream, family, cfg, seed):
    S = streams.get(stream, seed)
    X, y, K = S["X"], S["y"], S["K"]
    L = FAMILIES[family](K, seed=seed, **cfg)
    L.start(X[:WARM], K)
    correct = np.zeros(len(y), np.int8)
    bset = set(S["changes"]) if S["tasks"] is not None else set()
    t0 = time.time()
    for i in range(len(y)):
        if i in bset:
            L.boundary()
        correct[i] = L.answer(X[i]) == y[i]
        L.learn(X[i], y[i])
    rec = dict(stream=stream, family=family, cfg=cfg, seed=seed, n=len(y), acc=float(correct.mean()),
               pJ_answer=L.answer_pJ / len(y), pJ_update=L.update_pJ / len(y), mem_bits=L.mem_bits(),
               secs=round(time.time() - t0, 1))
    rec["recovery"] = float(np.mean([correct[c:c + WIN].mean() for c in S["changes"]])) if S["changes"] else None
    rec["relearn"] = [float(correct[b:b + WIN].mean() - correct[a:a + WIN].mean()) for a, b in S["recur"]]
    if S["tasks"] is not None:                                   # retention: every earlier task, at the end, no learning
        ret = []
        for Xt, yt in S["tasks"]:
            ret.append(float(np.mean([L.answer(x) == t for x, t in zip(Xt, yt)])))
        rec["retention"] = ret
    rec["curve"] = [round(float(v), 3) for v in correct.reshape(-1)[: len(y) // 250 * 250].reshape(-1, 250).mean(1)]
    return rec


if __name__ == "__main__":
    stream, family, cfgs, a, b = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    tag = sys.argv[6] if len(sys.argv) > 6 else f"{stream}_{family}_{a}"
    if cfgs.startswith("grid:"):
        import grids; cfgs = grids.chunk(cfgs)
    else:
        cfgs = json.load(open(cfgs[1:])) if cfgs.startswith("@") else json.loads(cfgs)
    if isinstance(cfgs, dict): cfgs = [cfgs]
    out = []
    for cfg in cfgs:
        for seed in range(a, b):
            try:
                r = run(stream, family, cfg, seed)
            except Exception as e:                               # a failing setting is recorded, not fatal
                print("FAILED", cfg, seed, repr(e), flush=True); continue
            out.append(r)
            print(json.dumps({k: r[k] for k in ("stream", "family", "cfg", "seed", "acc", "pJ_answer", "pJ_update", "secs")}), flush=True)
            json.dump(out, open(f"results_r5_{tag}.json", "w"), separators=(",", ":"))
