#!/usr/bin/env python3
"""Summarise the frozen endtoend re-evaluation (one Slurm array) into the research log.

Reads only per-episode result.json files of the array's run dirs (never modifies a run), writes
  research/greene/records/endtoend-eval-progress.json
and rewrites the marked block in research/README.zh-CN.md:
  <!-- greene-endtoend:start --> ... <!-- greene-endtoend:end -->
With --commit, commits just those two paths and pushes if they changed.
Live counts only: no success-rate differences, CIs or conclusions before all 1620 are in and audited.

    python3 update_endtoend_progress.py --array <array job id> [--commit]
"""
import argparse
import datetime
import glob
import json
import os
import subprocess

REPO = "/scratch/zz4330/OpenWAM"
RUNS = "/scratch/zz4330/openwam-runtime/runs"
PROTOCOL = "research/greene/endtoend-eval/protocol.json"
RECORD = "research/greene/records/endtoend-eval-progress.json"
README = "research/README.zh-CN.md"
START, END = "<!-- greene-endtoend:start -->", "<!-- greene-endtoend:end -->"
REMOTE = "git@github.com:Exiam6/OpenWAM.git"
BRANCH = "research/progress-20260927"


def array_tasks(array):
    """{array index: (raw job id, state)} from sacct."""
    try:
        out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", array, "-o", "JobIDRaw,JobID,State"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    tasks = {}
    for line in out.splitlines():
        raw, jid, state = line.split("|")[:3]
        if "_" in jid and "[" not in jid:
            tasks[int(jid.split("_")[1])] = (raw, state.split()[0])
    return tasks


def summarise(array):
    with open(os.path.join(REPO, PROTOCOL)) as fh:
        protocol = json.load(fh)
    cells, per = protocol["cells"], protocol["evaluation"]["scenes_per_task"]
    tasks = array_tasks(array)
    routes = {}
    states = {}
    for i, (route, seed, task, cond) in enumerate(cells):
        raw, state = tasks.get(i, (None, "NOT_SUBMITTED"))
        states[state] = states.get(state, 0) + 1
        rs = []
        if raw:
            for p in sorted(glob.glob(os.path.join(RUNS, "endtoend-eval-%s-seed%d-%s" % (route, seed, raw),
                                                   task, cond, "seed-*", "result.json"))):
                try:
                    with open(p) as fh:
                        rs.append(json.load(fh))
                except (OSError, ValueError):
                    pass          # being written right now; picked up next time
        done = [r for r in rs if r.get("status") == "complete"]
        v = routes.setdefault(route, {}).setdefault(cond, {"planned": 0, "complete": 0, "successes": 0,
                                                          "technical_failures": 0})
        v["planned"] += per
        v["complete"] += len(done)
        v["successes"] += sum(1 for r in done if r.get("success"))
        v["technical_failures"] += len(rs) - len(done)
    flat = [v for r in routes.values() for v in r.values()]
    return {
        "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "slurm_array": array,
        "array_task_states": states,
        "protocol": PROTOCOL,
        "deadline": protocol["deadline"],
        "episodes_planned": sum(v["planned"] for v in flat),
        "episodes_complete": sum(v["complete"] for v in flat),
        "successes": sum(v["successes"] for v in flat),
        "technical_failures": sum(v["technical_failures"] for v in flat),
        "routes": routes,
        "note": "Live counts, not a final report. Pooled over 3 seeds x 3 tasks; no differences, "
                "CIs or conclusions until all 1620 are complete and audited. 295M three-task setup only.",
    }


def block(s):
    conds = ["clean", "gaussian_sigma_0.10", "head_camera_yaw_5deg"]
    cell = lambda v: "%d/%d · %d 成功 · %d 技术失败" % (v["complete"], v["planned"], v["successes"],
                                                        v["technical_failures"])
    rows = "\n".join("| %s | %s |" % (r, " | ".join(cell(s["routes"][r][c]) for c in conds))
                     for r in ["wan", "svae", "pca"] if r in s["routes"])
    st = "、".join("%s %d" % kv for kv in sorted(s["array_task_states"].items()))
    return (START + "\n"
            "### Greene endtoend 匹配表征重评（自动更新）\n\n"
            "更新于 {u}；Slurm 数组 {a}（{st}）；协议 `{pr}`（已冻结，截止 {d}）。\n\n"
            "| 路线 | clean | gaussian σ0.10 | head yaw 5° |\n| --- | --- | --- | --- |\n{rows}\n\n"
            "合计 **{c}/{p}** 完成，{sc} 成功，{tf} 技术失败。每格合并 3 seed × 3 任务。"
            "实时计数，不是最终报告；1620 回合全部完成并审计前不计算差值或区间、不下结论。"
            "仅限 295M 三任务设置。\n"
            .format(u=s["updated_at"], a=s["slurm_array"], st=st, pr=s["protocol"], d=s["deadline"],
                    rows=rows, c=s["episodes_complete"], p=s["episodes_planned"],
                    sc=s["successes"], tf=s["technical_failures"])
            + END)


def git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", required=True)
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    s = summarise(a.array)

    with open(os.path.join(REPO, RECORD), "w") as fh:
        json.dump(s, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    rp = os.path.join(REPO, README)
    with open(rp) as fh:
        text = fh.read()
    if START in text and END in text:
        pre, rest = text.split(START, 1)
        text = pre + block(s) + rest.split(END, 1)[1]
    else:
        anchor = "## 后续执行顺序"
        text = text.replace(anchor, block(s) + "\n\n" + anchor, 1)
    with open(rp, "w") as fh:
        fh.write(text)
    print("%s  %d/%d complete, %d success, %d technical  %s"
          % (s["updated_at"], s["episodes_complete"], s["episodes_planned"],
             s["successes"], s["technical_failures"], s["array_task_states"]))

    if a.commit:
        if not git("status", "--porcelain", "--", RECORD, README).stdout.strip():
            print("no change"); return
        git("add", "--", RECORD, README)
        msg = ("research/greene: endtoend re-evaluation progress %d/%d\n\n"
               "Automatic update by research/greene/scripts/update_endtoend_progress.py.\n\n"
               "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
               % (s["episodes_complete"], s["episodes_planned"]))
        r = git("commit", "-q", "-m", msg, "--", RECORD, README)
        if r.returncode:
            print("commit failed:", r.stderr.strip()); return
        r = git("pull", "-q", "--rebase", "--autostash", REMOTE, BRANCH)
        if r.returncode:
            git("rebase", "--abort")
            print("pull --rebase failed, not pushing:", r.stderr.strip()); return
        r = git("push", "-q", REMOTE, "HEAD:" + BRANCH)
        print("pushed" if r.returncode == 0 else "push failed: " + r.stderr.strip())


if __name__ == "__main__":
    main()
