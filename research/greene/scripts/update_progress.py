#!/usr/bin/env python3
"""Summarise the running Greene official baseline into the research log.

Reads only per-episode result.json files (never modifies the run), writes
  research/greene/records/official-baseline-progress.json
and rewrites the marked block in research/README.zh-CN.md:
  <!-- greene-progress:start --> ... <!-- greene-progress:end -->
With --commit, commits just those two paths and pushes if they changed.

    python3 update_progress.py --run /scratch/zz4330/openwam-runtime/runs/official-baseline-18771822 [--commit]
"""
import argparse
import datetime
import glob
import json
import os
import subprocess

REPO = "/scratch/zz4330/OpenWAM"
RECORD = "research/greene/records/official-baseline-progress.json"
README = "research/README.zh-CN.md"
START, END = "<!-- greene-progress:start -->", "<!-- greene-progress:end -->"
TASKS = ["adjust_bottle", "handover_block", "place_object_basket"]
PER_TASK = 20
REMOTE = "git@github.com:Exiam6/OpenWAM.git"
BRANCH = "research/progress-20260927"


def job_state(job):
    try:
        out = subprocess.run(["sacct", "-n", "-X", "-j", job, "-o", "State"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        return out[0] if out else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def summarise(run):
    tasks = {}
    for t in TASKS:
        rs = []
        for p in sorted(glob.glob(os.path.join(run, t, "clean", "seed-*", "result.json"))):
            try:
                with open(p) as fh:
                    rs.append(json.load(fh))
            except (OSError, ValueError):
                pass          # being written right now; picked up next time
        done = [r for r in rs if r.get("status") == "complete"]
        tech = [r for r in rs if r.get("status") != "complete"]
        tasks[t] = {
            "planned": PER_TASK,
            "complete": len(done),
            "successes": sum(1 for r in done if r.get("success")),
            "technical_failures": len(tech),
            "mean_seconds": round(sum(r.get("seconds", 0) for r in done) / len(done), 1) if done else None,
        }
    comp = sum(v["complete"] for v in tasks.values())
    succ = sum(v["successes"] for v in tasks.values())
    job = os.path.basename(run.rstrip("/")).rsplit("-", 1)[-1]
    exit_rec = None
    if os.path.exists(os.path.join(run, "exit.json")):
        with open(os.path.join(run, "exit.json")) as fh:
            exit_rec = json.load(fh)
    return {
        "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": run,
        "slurm_job": job,
        "job_state": job_state(job),
        "exit": exit_rec,
        "protocol": "research/greene/baseline-official/protocol.json",
        "episodes_planned": PER_TASK * len(TASKS),
        "episodes_complete": comp,
        "successes": succ,
        "technical_failures": sum(v["technical_failures"] for v in tasks.values()),
        "tasks": tasks,
        "note": "Live progress, not a final report. Official released checkpoint on the "
                "custom 3-task x 20-scene clean cohort; not the official 50-task benchmark.",
    }


def block(s):
    rows = "\n".join(
        "| {t} | {c}/{p} | {s} | {f} |".format(t=t, c=v["complete"], p=v["planned"],
                                               s=v["successes"], f=v["technical_failures"])
        for t, v in s["tasks"].items())
    return (START + "\n"
            "### Greene 官方检查点参考基线（自动更新）\n\n"
            "更新于 {u}；Slurm {j}（{st}）；协议 `{pr}`。\n\n"
            "| 任务 | 完成 | 成功 | 技术失败 |\n| --- | --- | --- | --- |\n{rows}\n"
            "| **合计** | **{c}/{p}** | **{sc}** | **{tf}** |\n\n"
            "实时进度，不是最终报告；成功率在 60 回合完成并审计前不下结论。"
            "发布检查点 + 自定义三任务 clean 场景，不是官方 50 任务基准。\n"
            .format(u=s["updated_at"], j=s["slurm_job"], st=s["job_state"], pr=s["protocol"],
                    rows=rows, c=s["episodes_complete"], p=s["episodes_planned"],
                    sc=s["successes"], tf=s["technical_failures"])
            + END)


def git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()
    s = summarise(a.run)

    rec = os.path.join(REPO, RECORD)
    os.makedirs(os.path.dirname(rec), exist_ok=True)
    with open(rec, "w") as fh:
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
    print("%s  %d/%d complete, %d success, %d technical"
          % (s["updated_at"], s["episodes_complete"], s["episodes_planned"],
             s["successes"], s["technical_failures"]))

    if a.commit:
        if not git("status", "--porcelain", "--", RECORD, README).stdout.strip():
            print("no change"); return
        git("add", "--", RECORD, README)
        msg = ("research/greene: official baseline progress %d/%d (%d success)\n\n"
               "Automatic update by research/greene/scripts/update_progress.py.\n\n"
               "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
               % (s["episodes_complete"], s["episodes_planned"], s["successes"]))
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
