"""Fresh real-data pool for the grow-on-demand test (registered 11 Oct 2026 before any of these ran).
None of these datasets appeared in the market scan or its confirmation. Tag: c3_fresh."""
import scan_tasks as T

FRESH = {"car": 40975, "nursery": 26, "connect4": 40668, "balance": 11, "vote": 56, "breast-cancer": 13,
         "kr-vs-k": 184, "soybean": 42, "sick": 38, "shuttle": 40685, "led7": 40496, "australian": 40981,
         "sylvine": 41146, "ada": 41156, "pc2": 1069, "dresses": 23381, "click": 1220, "airlines": 1169,
         "hepatitis": 55, "musk": 1116, "nomao": 1486, "jasmine": 41143, "miniboone": 41150, "jannis": 41168,
         "albert": 41147, "helena": 41169, "default-credit": 42477, "mfeat-pixel": 40979}
for nm, did in FRESH.items():
    T._reg("c3_" + nm, T.openml(did), "c3_fresh", False)
