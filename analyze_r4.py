"""Score round four by the pre-registered rules."""
import json, glob, numpy as np
from tasks4 import ORIGINAL, UNSEEN

def load(task):
    recs = []
    for f in sorted(glob.glob(f"results_r4_{task}_s*.json")) + sorted(glob.glob(f"results/results_r4_{task}_s*.json")):
        recs += json.load(open(f))
    return recs

def mean(recs, key, f):
    return float(np.mean([f(r[key]) for r in recs]))

def score(task):
    R = load(task)
    if not R: return None
    nets = [k for k in R[0] if k.startswith("mlp")]
    trees = [k for k in R[0] if k.startswith(("tree", "forest"))]
    A = lambda k: mean(R, k, lambda v: v["acc"]); E = lambda k: mean(R, k, lambda v: v["energy"])
    ma, me = A("mycel"), E("mycel")
    ok = [k for k in nets if A(k) >= ma - 0.005]
    ref = min(ok, key=E) if ok else max(nets, key=A)
    cheaper = [k for k in nets if E(k) <= me]
    gain = ma - max(A(k) for k in cheaper) if cheaper else None
    fl = lambda k, rate: A(k) - mean(R, k, lambda v: v["flip"][rate])
    kl = lambda k: A(k) - mean(R, k, lambda v: v["kill"]["0.3"]["after"])
    tok = [k for k in trees if A(k) >= ma - 0.005]
    tref = min(tok, key=E) if tok else max(trees, key=A)
    s = dict(task=task, n=len(R), acc=ma, E=me, ref=ref, ref_acc=A(ref), ref_E=E(ref), ratio=me / E(ref), gain=gain,
             flip2=(fl("mycel", "0.01"), fl(ref, "0.01")), flip3=(fl("mycel", "0.001"), fl(ref, "0.001")),
             kill=(kl("mycel"), kl(ref)), tree=tref, tree_acc=A(tref), tree_ratio=me / E(tref),
             best_net=max(A(k) for k in nets), cfgs=[r["cfg"]["tag"] + str(r["cfg"].get("P", r["cfg"].get("C"))) for r in R])
    s["W_energy"] = s["ratio"] <= 0.5
    s["W_acc"] = gain is not None and gain >= 0.01
    s["W_flip"] = (s["flip2"][1] - s["flip2"][0] >= 0.02) and (s["flip3"][0] <= s["flip3"][1] + 0.005)
    s["W_kill"] = s["kill"][1] - s["kill"][0] >= 0.02
    return s

if __name__ == "__main__":
    for group, tasks in (("ORIGINAL", ORIGINAL), ("UNSEEN", UNSEEN)):
        S = [s for s in map(score, tasks) if s]
        print("==", group)
        for s in S:
            print(f"{s['task']:8s} n={s['n']:2d} Mycel {100*s['acc']:.1f}% {s['E']:.0f}pJ | ref {s['ref']} {100*s['ref_acc']:.1f}% {s['ref_E']:.0f} | ratio {s['ratio']:.2f} "
                  f"| eqE {('%+.1f' % (100*s['gain'])) if s['gain'] is not None else '-'} | flip.01 {100*s['flip2'][0]:.1f}/{100*s['flip2'][1]:.1f} flip.001 {100*s['flip3'][0]:.1f}/{100*s['flip3'][1]:.1f} "
                  f"| kill30 {100*s['kill'][0]:.1f}/{100*s['kill'][1]:.1f} | tree {s['tree']} {100*s['tree_acc']:.1f}% ratio {s['tree_ratio']:.2f} | W E{int(s['W_energy'])} A{int(s['W_acc'])} F{int(s['W_flip'])} K{int(s['W_kill'])}")
        for c in ("W_energy", "W_acc", "W_flip", "W_kill"):
            print(f"  {c}: {sum(s[c] for s in S)} of {len(S)}")
