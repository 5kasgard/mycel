"""Round four forgetting test: frozen Mycel-D (shared colony 200, consolidation with double-strength
consolidated votes, 100 new nuclei for task B) vs plain and EWC networks. Seeds 0-4; three pairs,
the third (2 vs 5, then 6 vs 8) never used in development."""
import json, numpy as np
import forget_d
from forget_d import run, summary
from dmycel import MycelCo, grow_nuclei
from baselines_d import MLPD, ewc_anchor
forget_d.PAIRS = [((0, 1), (7, 9)), ((3, 8), (4, 6)), ((2, 5), (6, 8))]
PASSES, SEEDS = 50, (0, 1, 2, 3, 4)

def mycel_protect(m, X, y):
    m.consolidate(X, y, 1.0)
    k = m.alive == 2
    m.w8[:, k] = np.clip(m.w8[:, k].astype(int) * 2, -127, 127).astype(np.int8)
    grow_nuclei(m, 100)

res = {}
res["mycel_r4"] = run(lambda sd: MycelCo(2, P=200, B=2, s=5.0, T=100, seed=sd), PASSES, SEEDS, consolidate=mycel_protect)
res["mycel_r3"] = run(lambda sd: MycelCo(2, P=100, B=2, s=5.0, T=50, seed=sd), PASSES, SEEDS, consolidate=lambda m, X, y: m.consolidate(X, y, 0.5))
for h in (2, 4, 8, 16, 32):
    res[f"mlp_h{h}"] = run(lambda sd: MLPD(64, 2, h=h, bits=8, seed=sd), PASSES, SEEDS)
for h, lam in ((8, 1e3), (16, 1e5), (32, 1e5)):
    res[f"ewc_h{h}"] = run(lambda sd: MLPD(64, 2, h=h, bits=8, seed=sd), PASSES, SEEDS, consolidate=lambda m, X, y: ewc_anchor(m, X, y, lam))
for k, v in res.items():
    print(k, summary(v), "| unseen pair:", summary([o for o in v if tuple(o["A"]) == (2, 5)]), flush=True)
json.dump(res, open("results_r4_forgetting.json", "w"), indent=1)
