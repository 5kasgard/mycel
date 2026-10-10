"""Condense round-four result files into one small JSON (what scoring needs, nothing else).
Usage: python compact.py <glob> <out>"""
import json, glob, sys
out = []
for f in sorted(glob.glob(sys.argv[1])):
    for r in json.load(open(f)):
        rec = dict(task=r["task"], seed=r["seed"], cfg=r["cfg"], m={})
        for k, v in r.items():
            if isinstance(v, dict) and "acc" in v:
                row = [round(v["acc"], 4), round(v["energy"], 1)]
                if "flip" in v:
                    row += [round(v["flip"]["0.001"], 4), round(v["flip"]["0.01"], 4),
                            round(v["kill"]["0.3"]["after"], 4), round(v["kill"]["0.3"]["recovered"], 4)]
                rec["m"][k] = row
        out.append(rec)
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
print(len(out), "records")
