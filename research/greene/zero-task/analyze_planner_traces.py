"""CPU-only: summarize ex3 closed-loop traces per task and arm (planner status, gripper, contacts, motion).

Reads evaluation/fresh/<arm>-seed<s>/<task>/clean/seed-*/{result.json,trace.jsonl} of the ex3 study.
No model or simulator is loaded. Output: JSON summary + printed table.
"""
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/home/zifanz4/openwam-runtime/rae-factorial-20261010/evaluation/fresh")


def episode(d):
    res = json.loads((d / "result.json").read_text())
    rows = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines()]
    calls = collections.Counter()
    fail_steps = {"left": 0, "right": 0}
    for r in rows:
        for c in r["planner_calls"]:
            st = c.get("status", "EXC:" + c.get("exception", "?")[:40])
            calls[(c["side"], st)] += 1
            if st != "Success":
                fail_steps[c["side"]] += 1
    a = np.array([r["action"] for r in rows])
    p = np.array([r["proprio"] for r in rows])
    # eef20 layout: left [0:3 pos, 3:9 rot6d, 9 grip], right [10:13, 13:19, 19 grip]
    lg, rg = a[:, 9], a[:, 19]
    contact_objs = collections.Counter()
    for r in rows:
        for c in r["gripper_contacts"]:
            for b in c["bodies"]:
                contact_objs[b] += 1
    travel_l = float(np.linalg.norm(np.diff(p[:, 0:3], axis=0), axis=1).sum())
    travel_r = float(np.linalg.norm(np.diff(p[:, 10:13], axis=0), axis=1).sum())
    return dict(success=bool(res.get("success")), steps=len(rows), calls=calls, fail_steps=fail_steps,
                left_closed_frac=float((lg < 0.5).mean()), right_closed_frac=float((rg < 0.5).mean()),
                left_ever_closed=bool((lg < 0.5).any()), right_ever_closed=bool((rg < 0.5).any()),
                contact_objs=contact_objs, travel_l=travel_l, travel_r=travel_r,
                ticks=rows[-1]["physics_ticks"] if rows else 0)


def main():
    out = {}
    for task in ["adjust_bottle", "handover_block", "place_object_basket"]:
        for armdir in sorted(R.glob("*-seed*")):
            arm = armdir.name.rsplit("-seed", 1)[0]
            for d in sorted((armdir / task / "clean").glob("seed-*")):
                if not (d / "trace.jsonl").exists():
                    continue
                e = episode(d)
                key = f"{task}|{arm}"
                g = out.setdefault(key, dict(n=0, succ=0, steps=[], left_fail_steps=0, right_fail_steps=0, steps_total=0,
                                            statuses=collections.Counter(), left_ever_closed=0, right_ever_closed=0,
                                            contacts=collections.Counter(), travel_l=[], travel_r=[]))
                g["n"] += 1; g["succ"] += e["success"]; g["steps"].append(e["steps"]); g["steps_total"] += e["steps"]
                g["left_fail_steps"] += e["fail_steps"]["left"]; g["right_fail_steps"] += e["fail_steps"]["right"]
                for (side, st), k in e["calls"].items():
                    g["statuses"][f"{side}:{st}"] += k
                g["left_ever_closed"] += e["left_ever_closed"]; g["right_ever_closed"] += e["right_ever_closed"]
                for b, k in e["contact_objs"].items():
                    g["contacts"][b] += k
                g["travel_l"].append(e["travel_l"]); g["travel_r"].append(e["travel_r"])
    summary = {}
    for key, g in out.items():
        summary[key] = dict(episodes=g["n"], successes=g["succ"], mean_steps=float(np.mean(g["steps"])),
                            left_planner_fail_rate=g["left_fail_steps"] / g["steps_total"], right_planner_fail_rate=g["right_fail_steps"] / g["steps_total"],
                            statuses=dict(g["statuses"].most_common(8)), left_ever_closed=g["left_ever_closed"], right_ever_closed=g["right_ever_closed"],
                            top_contacts=dict(g["contacts"].most_common(6)), mean_travel_left_m=float(np.mean(g["travel_l"])), mean_travel_right_m=float(np.mean(g["travel_r"])))
    json.dump(summary, open(sys.argv[1], "w"), indent=1)
    for key, s in summary.items():
        print(key, f"succ {s['successes']}/{s['episodes']} steps {s['mean_steps']:.0f} planFail L {100*s['left_planner_fail_rate']:.1f}% R {100*s['right_planner_fail_rate']:.1f}% closedEver L {s['left_ever_closed']} R {s['right_ever_closed']} travel L {s['mean_travel_left_m']:.2f} R {s['mean_travel_right_m']:.2f}")
        print("    statuses", s["statuses"]); print("    contacts", s["top_contacts"])


if __name__ == "__main__":
    main()
