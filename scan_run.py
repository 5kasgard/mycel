"""Market scan runner. Usage: python scan_run.py <task> <variant,variant,...> <seed_from> <seed_to> <tag>
Prequential (answer, then learn) on one task; writes results_scan_<tag>.json."""
import json, sys, time, numpy as np
import patches  # noqa: runtime fixes (river submodules)
import scan_tasks as T
from learners_v2 import VARIANTS

WARM, WIN = 200, 500
GROUPS = {"A": "mycelT,mycelS,mycelA,mycelL,mlp_adam128,mlp_sgd32,logistic,gnb,knn,htree,majority",
          "B": "hat,arf5", "C": "arf20"}


def traits(S):
    X, y = S["X"], S["y"]
    binf = float(np.mean([np.all((X[:, j] == 0) | (X[:, j] == 1)) for j in range(X.shape[1])]))
    return dict(n=len(y), d=X.shape[1], K=S["K"], frac_binary=binf, density=float((X != 0).mean()),
                majority=float(np.bincount(y).max() / len(y)), drift=int(len(S["changes"]) > 0))


def run(task, S, name, seed):
    cls, kw = VARIANTS[name]
    X, y, K = S["X"], S["y"], S["K"]
    L = cls(K, seed=seed, **kw); L.start(X[:WARM], K)
    ok = np.zeros(len(y), np.int8); t0 = time.time()
    for i in range(len(y)):
        ok[i] = L.answer(X[i]) == y[i]; L.learn(X[i], y[i])
    rec = dict(task=task, tag=T.TAG[task], half=T.half(task), variant=name, seed=seed, acc=float(ok.mean()),
               acc_late=float(ok[len(y) // 2:].mean()), pJ_answer=L.answer_pJ / len(y), pJ_update=L.update_pJ / len(y),
               mem_bits=int(L.mem_bits()), secs=round(time.time() - t0, 1))
    if hasattr(L, "diag"): rec["diag"] = L.diag
    rec["recovery"] = float(np.mean([ok[c:c + WIN].mean() for c in S["changes"]])) if S["changes"] else None
    return rec


if __name__ == "__main__":
    task, names, a, b, tag = sys.argv[1], GROUPS.get(sys.argv[2], sys.argv[2]).split(","), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    out = []
    for seed in range(a, b):
        try:
            S = T.get(task, seed)
        except Exception as e:
            print("LOAD FAILED", task, seed, repr(e), flush=True); break
        tr = traits(S)
        for nm in names:
            try:
                r = run(task, S, nm, seed)
            except Exception as e:
                print("FAILED", task, nm, seed, repr(e), flush=True); continue
            r.update(tr); out.append(r)
            print(json.dumps({k: r[k] for k in ("task", "variant", "seed", "acc", "pJ_answer", "pJ_update", "secs")}), flush=True)
            json.dump(out, open(f"results_scan_{tag}.json", "w"), separators=(",", ":"))
