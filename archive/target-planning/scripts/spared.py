import json, sys, time
sys.path.insert(0, "/home/vilem/Projects/brats-upenn-gbm")
from plan import measure, planeval
out = "/home/vilem/.claude/jobs/63a02e34/tmp/spared.json"
rows = []
for first, later in measure.pairs(tissue=True):
    for region in ("ET", None):
        t = time.time()
        try:
            r = planeval.spared(first, later, region=region)
        except Exception as e:
            r = {"region": region or "all", "error": repr(e)}
        r["patient"] = first[-8:-3]
        r["seconds"] = round(time.time() - t)
        rows.append(r)
        print(json.dumps(r), flush=True)
        json.dump(rows, open(out, "w"), indent=1)
print("DONE", flush=True)
