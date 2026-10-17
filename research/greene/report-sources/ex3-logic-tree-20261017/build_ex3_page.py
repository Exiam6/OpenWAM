"""Generate the ex3 logic-tree progress page FROM RUN FILES (TOP_PRIORITY/LOGIC_TREE.md rule 8).

Reads: the study runtime (protocol, claims, run status/metrics/results, scene manifests, evaluation
results, success-summary.json), the registered PLAN, and the research-branch records for the earlier
nodes (ex1/ex2 ledger audit, official baseline, renderer A/B, shift-weight audit). Emits spec.json and
builds page/index.html with research_lab/templates/logic_tree_page/build_logic_tree.py. Videos are
assembled from the frames each evaluation episode logged (make_videos.py). Never edit the page by hand.

Usage: build_ex3_page.py [--no-videos]
"""
import datetime as dt
import html
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
R = Path("/home/zifanz4/openwam-runtime/rae-factorial-20261010")
G = HERE.parent.parent  # research/greene
REC = G / "records"
PAGE = HERE / "page"
MEDIA = PAGE / "media"
BUILDER = Path("/home/zifanz4/research_lab/templates/logic_tree_page/build_logic_tree.py")
TZ = ZoneInfo("America/Chicago")
TASKS = ["adjust_bottle", "handover_block", "place_object_basket"]
ARMS = ["raw768_native", "raw768_native_shift16", "raw768_wide", "raw768_wide_shift16", "svae48_native"]
ARM_LABEL = {
    "raw768_native": "raw768 native (baseline)",
    "raw768_native_shift16": "raw768 native, shift 16.43 (ex3-a)",
    "raw768_wide": "raw768 wide head (ex2-b)",
    "raw768_wide_shift16": "raw768 wide head, shift 16.43 (ex3-b)",
    "svae48_native": "S-VAE 48 (ex2-c)",
}
SEEDS = [42, 43, 44]
SOURCES = []  # every file a number came from


def load(path, default=None):
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default
    SOURCES.append(str(path))
    return data


def esc(s):
    return html.escape(str(s), quote=False)


def local(iso):
    if not iso:
        return None
    t = dt.datetime.fromisoformat(iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=TZ)
    return t.astimezone(TZ).replace(tzinfo=None)


def fmt(t):
    return t.strftime("%m-%d %H:%M") if t else "—"


def pct(a, b):
    return f"{100 * a / b:.1f} %" if b else "—"


def wilson(k, n, z=1.96):
    if not n:
        return "—"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return f"{100 * (c - h):.0f}–{100 * (c + h):.0f}"


def table(head, rows):
    return ('<div class="tw"><table><thead><tr>' + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>")


def dl(src, name):
    return f'<button type="button" class="dl" data-src="{esc(src)}" data-name="{esc(name)}">Download</button>'


def five_parts(background, question, variable, criterion, outcomes):
    return ("<h3>Background</h3><p>" + background + "</p><h3>Question</h3><p>" + question + "</p><h3>Single variable, everything else locked</h3><p>"
            + variable + "</p><h3>Criterion (written before the run)</h3><p>" + criterion + "</p><h3>What each outcome changes</h3><p>" + outcomes + "</p>")


def sources_html(paths):
    return "<h3>Files the numbers come from</h3><ul>" + "".join(f"<li><code>{esc(p)}</code></li>" for p in paths) + "</ul>"


# ----------------------------------------------------------------------------- live study state
def claims(folder):
    out = {}
    for f in sorted((R / folder).glob("*.json")):
        out[f.stem] = load(f, {})
    return out


def run_state(arm, seed):
    d = R / "runs" / f"{arm}-seed{seed}"
    result = load(d / "result.json")
    status = load(d / "status.json")
    return result, status


def training_rows(train_claims):
    rows, lanes = [], []
    for arm, group in [("raw768_native_shift16", "ex3-a training"), ("raw768_wide_shift16", "ex3-b training")]:
        for seed in SEEDS:
            claim = train_claims.get(f"train-{arm}-seed{seed}", {})
            result, status = run_state(arm, seed)
            if result and result.get("complete"):
                state, text = "done", f"done, {result['training_seconds'] / 3600:.2f} h, dev action loss {result['final_dev_action_loss']:.4f}"
            elif claim.get("status") == "failed":
                state, text = "wait", f"FAILED: {claim.get('error', claim.get('exit_code'))}"
            elif status:
                state, text = "run", f"step {status['step']} / {status['steps']}, loss {status['loss']:.3f}"
            elif claim:
                state, text = "run", "claimed, loading"
            else:
                state, text = "wait", "queued"
            rows.append([esc(ARM_LABEL[arm]), seed, claim.get("gpu", "—"), fmt(local(claim.get("started"))), fmt(local(claim.get("finished"))), esc(text)])
            start = local(claim.get("started"))
            end = local(claim.get("finished"))
            projected = False
            if start and not end and status and status["step"] > 0:
                rate = status["seconds"] / status["step"]
                end = start + dt.timedelta(seconds=rate * status["steps"] + 240)
                projected = True
            if start:
                lanes.append(dict(lane=group, label=f"seed {seed}", start=start.isoformat(), end=end.isoformat() if end else None, state=state, href="h1" if "native" in arm else "h3", projected=projected))
    return rows, lanes


def scene_state():
    rows = []
    for task in TASKS:
        m = load(R / "scenes" / task / "manifest.json")
        if m:
            rows.append([task, len(m.get("accepted", [])), len(m.get("attempted", m.get("candidates", []))) if isinstance(m.get("attempted", m.get("candidates", [])), list) else m.get("attempted", "—"), "complete" if m.get("complete") else "in progress"])
        else:
            rows.append([task, 0, 0, "not started"])
    return rows


def episode_records():
    recs = {}
    for f in (R / "evaluation" / "fresh").glob("*/*/clean/seed-*/result.json"):
        r = load(f)
        if r:
            recs[(r["route"], r["training_seed"], r["task"], r["scene_seed"])] = (r, f.parent)
    return recs


def counts(recs):
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for task in TASKS + ["all"]:
            sel = [r for (a, s, t, c), (r, _) in recs.items() if a == arm and (task == "all" or t == task)]
            out[arm][task] = dict(
                successes=sum(1 for r in sel if r["status"] == "complete" and r.get("success")),
                attempted=len(sel),
                failures=sum(1 for r in sel if r["status"] != "complete"),
                by_seed={s: (sum(1 for r in sel if r["training_seed"] == s and r["status"] == "complete" and r.get("success")), sum(1 for r in sel if r["training_seed"] == s)) for s in SEEDS},
            )
    return out


# ----------------------------------------------------------------------------- videos
def make_videos(recs, enabled):
    """success / median / failure per arm x task x seed 42 (then other seeds if 42 lacks one), from logged frames."""
    media = {}
    cards = {}
    if not enabled:
        return media, cards
    spec = importlib.util.spec_from_file_location("mv", HERE / "make_videos.py")
    mv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mv)
    MEDIA.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for task in TASKS:
            eps = [(k, v) for k, v in recs.items() if k[0] == arm and k[2] == task and v[0]["status"] == "complete"]
            if not eps:
                continue
            eps.sort(key=lambda kv: (kv[0][1], kv[0][3]))
            succ = [e for e in eps if e[1][0].get("success")]
            fail = [e for e in eps if not e[1][0].get("success")]
            steps = sorted(eps, key=lambda e: e[1][0].get("steps", 0))
            picks = {}
            if succ:
                picks["success"] = succ[0]
            if fail:
                picks["failure"] = fail[-1]
            picks["median"] = steps[len(steps) // 2]
            for kind, ((a, s, t, c), (r, d)) in picks.items():
                name = f"{arm}__{task}__{kind}__s{s}_scene{c}.mp4"
                target = MEDIA / name
                if not target.exists() and (d / "frames").exists():
                    try:
                        mv.assemble(d, target)
                    except Exception as exc:  # keep building the page; say what failed
                        print("video failed", name, exc, file=sys.stderr)
                        continue
                if target.exists():
                    media[f"media/{name}"] = str(target)
                    cards.setdefault((arm, task), []).append((kind, s, c, r, f"media/{name}"))
    return media, cards


def video_grid(cards, arm, task=None):
    out = []
    for (a, t), items in sorted(cards.items()):
        if a != arm or (task and t != task):
            continue
        for kind, s, c, r, src in items:
            label = f"{t} · {kind} · seed {s} · scene {c} · {'success' if r.get('success') else 'failure'} in {r.get('steps')} actions"
            out.append(f'<div class="card"><video controls preload="metadata" src="{src}"></video><p class="muted">{esc(label)}</p>{dl(src, Path(src).name)}</div>')
    return '<div class="grid">' + "".join(out) + "</div>" if out else '<p class="muted">No evaluation episodes for this arm yet; videos appear here when the first cell finishes.</p>'


# ----------------------------------------------------------------------------- earlier nodes (records)
def ex2_block():
    a = load(REC / "expanded-20261006-audit-20261009.json")
    succ = a["by_set"]["combined"]["success"]
    rows = []
    for arm in ["svae48_native", "raw768_native", "raw768_wide"]:
        r = [esc(ARM_LABEL[arm])]
        for task in TASKS:
            s = succ[arm][task]
            r.append(f"{s['successes']} / {s['valid']} ({pct(s['successes'], s['valid'])}; per seed " + " / ".join(str(s['by_seed'][str(k)][0]) for k in SEEDS) + ")")
        rows.append(r)
    comp = a["comparisons"]["combined"]
    crow = [[esc(c["arm"]), esc(c["base"]), c["valid_pairs"], c["arm_only"], c["base_only"], f"{c['two_sided_exact_sign_p']:.3f}"] for c in comp]
    return succ, table(["arm", "adjust_bottle", "handover_block", "place_object_basket"], rows) + "<p>Paired sign test on the 177 valid scene×seed pairs (arm-only successes vs base-only successes):</p>" + table(["arm", "base", "valid pairs", "arm only", "base only", "two-sided exact p"], crow)


def official_block():
    o = load(REC / "official-baseline-20261002-result.json")
    f = o["full_60"]
    rows = [[t, f"{f[t]['successes']} / {f[t]['episodes']}"] for t in TASKS] + [["total", f"{f['total']['successes']} / {f['total']['episodes']}"]]
    return o, table(["task", "official checkpoint, 60 old scenes"], rows)


def shift_block():
    c = load(REC / "rae-shift-weight-audit-20261010/corrected-curves.json")
    rows = []
    for p42, p43 in zip(c["paired"]["42"], c["paired"]["43"]):
        rows.append([f"{p42['target_sigma']:.2f}", f"{p42['weighted_loss_ratio']:.3f}", f"{p42['weight_ratio']:.3f}", f"{p42['unweighted_mse_ratio']:.4f}", f"{p43['unweighted_mse_ratio']:.4f}"])
    ratios = [p["unweighted_mse_ratio"] for s in ("42", "43") for p in c["paired"][s]]
    return (min(ratios), max(ratios)), table(["target σ", "old weighted loss ratio (s42)", "training-weight ratio", "unweighted MSE ratio s42", "s43"], rows)


def render_block():
    o = load(REC / "oidn-ab-20261003.json")
    ref = o["shenlong_reference"]["all"]
    return ref


def ex1_numbers():
    text = (G / "EXPERIMENT_PLAN.md").read_text()
    SOURCES.append(str(G / "EXPERIMENT_PLAN.md"))
    rows = []
    for arm in ["raw768_native", "raw768_wide", "svae48_native"]:
        m = re.search(r"\| `" + arm + r"` \|.*?\*{0,2}(\d+/42)\*{0,2} \|", text)
        rows.append([esc(ARM_LABEL[arm]), m.group(1) if m else "?"])
    return rows


# ----------------------------------------------------------------------------- main
def main():
    no_videos = "--no-videos" in sys.argv
    now = dt.datetime.now(TZ).replace(tzinfo=None)
    protocol = load(R / "protocol.json")
    amendment = load(R / "resource-amendment-gpu01.json")
    plan_text = (HERE.parent.parent / "factorial" / "PLAN_ex3_20261017.md").read_text()
    SOURCES.append(str(HERE.parent.parent / "factorial" / "PLAN_ex3_20261017.md"))
    train_claims = claims("claims")
    post_claims = claims("post-claims")
    preflights = {v: load(R / "preflight" / f"{v}-seed42" / "result.json") for v in protocol["variants"]}
    trows, lanes = training_rows(train_claims)
    srows = scene_state()
    recs = episode_records()
    cnt = counts(recs)
    summary = load(R / "success-summary.json")
    media, cards = make_videos(recs, not no_videos)
    deadline = local(protocol["deadline"])
    t0 = dt.datetime(2026, 10, 17, 1, 50)

    # ---- lanes for screening and evaluation (measured from post-claims, else projected)
    train_end = max((local(c.get("finished")) or local(c.get("started")) for c in train_claims.values() if c.get("stage") == "train" and c.get("started")), default=None)
    proj_train_end = max((dt.datetime.fromisoformat(l["end"]) for l in lanes if l.get("end")), default=None)
    for task in TASKS:
        c = post_claims.get(f"collect-{task}")
        if c and c.get("started"):
            lanes.append(dict(lane="scene screening", label=task, start=local(c["started"]).isoformat(), end=(local(c.get("finished")) or now).isoformat(), state="done" if c.get("status") == "complete" else "run", href="base", projected=False))
        elif proj_train_end:
            lanes.append(dict(lane="scene screening", label=task + " (projected)", start=proj_train_end.isoformat(), end=(proj_train_end + dt.timedelta(hours=1)).isoformat(), state="wait", href="base", projected=True))
    eval_measured = False
    for arm in ARMS:
        for seed in SEEDS:
            c = post_claims.get(f"evaluate-{arm}-seed{seed}")
            if c and c.get("started"):
                eval_measured = True
                lanes.append(dict(lane=f"evaluation {arm}", label=f"seed {seed}", start=local(c["started"]).isoformat(), end=(local(c.get("finished")) or now).isoformat(), state="done" if c.get("status") == "complete" else "run", href="h1", projected=False))
    if not eval_measured and proj_train_end:
        s = proj_train_end + dt.timedelta(hours=1)
        lanes.append(dict(lane="evaluation (15 cells, projected)", label="15 cells × ~2.4 GPU-h on 8 GPUs", start=s.isoformat(), end=(s + dt.timedelta(hours=5)).isoformat(), state="wait", href="h1", projected=True))
    milestones = [dict(at=t0.isoformat(), label="authorization"), dict(at=deadline.isoformat(), label="deadline 12:00")]
    pf_done = [local(c.get("finished")) for k, c in train_claims.items() if k.startswith("preflight") and c.get("finished")]
    if pf_done:
        milestones.append(dict(at=max(pf_done).isoformat(), label="L1 preflight passed"))

    # ---- node states
    def arm_state(arm):
        n = cnt[arm]["all"]
        planned = 180
        if summary and summary.get("complete"):
            return "done", "final: " + " / ".join(f"{cnt[arm][t]['successes']}/{cnt[arm][t]['attempted']}" for t in TASKS)
        if n["attempted"]:
            return "run", f"evaluating: {n['successes']}/{n['attempted']} of {planned}"
        if arm in protocol["variants"]:
            runs = [run_state(arm, s) for s in SEEDS]
            if all(r and r.get("complete") for r, _ in runs):
                return "run", "trained; evaluation queued"
            steps = [st["step"] for _, st in runs if st]
            return ("run", f"training: steps {', '.join(map(str, steps))} / 20000") if steps else ("wait", "queued")
        return "wait", "checkpoints reused; evaluation queued"

    def contrast_text(cid, fallback):
        if summary and summary.get("complete") and cid in summary.get("contrasts", {}):
            c = summary["contrasts"][cid]
            lo, hi = c["scene_cluster_paired_95pct_interval"]
            verdict = "supported" if c["supported"] else ("refuted" if c["refuted"] else "undecided")
            return "done", f"Δ {100 * c['task_balanced_success_difference']:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] → {verdict}"
        return fallback

    h1_state = contrast_text("H1_shift16", arm_state("raw768_native_shift16"))
    h2_state = contrast_text("H2_widehead", arm_state("raw768_wide"))
    h3a_state = contrast_text("H3a_widehead_at_shift16", arm_state("raw768_wide_shift16"))
    h3b_state = contrast_text("H3b_shift16_at_widehead", arm_state("raw768_wide_shift16"))
    ex2_succ, ex2_tables = ex2_block()
    official, official_table = official_block()
    (rmin, rmax), shift_table = shift_block()
    render_ref = render_block()
    ex1_rows = ex1_numbers()
    base_state = ("done", "20 scenes × 3 tasks frozen") if all(r[3] == "complete" for r in srows) else (("run", "screening candidates") if any(r[3] == "in progress" for r in srows) else ("wait", "screening after training"))

    tree = [
        dict(id="Q", parent=None, title="Can RAE ideas raise RoboTwin success?", text="research question (user, 2026-10-02)", state="root", href="plan"),
        dict(id="Q0", parent="Q", title="Is the benchmark doable at all?", text="answered: yes (official 58/60)", state="root", href=None),
        dict(id="OFF", parent="Q0", title="Official checkpoint (Torch)", text=f"{official['full_60']['total']['successes']}/{official['full_60']['total']['episodes']} on the 60 old scenes", state="done", href="official"),
        dict(id="REN", parent="OFF", title="Renderer divergence (OIDN)", text=f"{render_ref['min']:.2f}–{render_ref['max']:.2f} uint8 vs reference PNGs", state="done", href="render"),
        dict(id="Q1", parent="Q", title="Does the latent width matter?", text="768-dim vs 48-dim", state="root", href=None),
        dict(id="EX1", parent="Q1", title="ex1: 66 demos, 4k steps", text=" · ".join(r[1] for r in ex1_rows) + " (native · wide · S-VAE)", state="done", href="ex1"),
        dict(id="EX2", parent="EX1", title="ex2: 11× data, 20k steps", text=f"adjust_bottle {ex2_succ['raw768_native']['adjust_bottle']['successes']} · {ex2_succ['raw768_wide']['adjust_bottle']['successes']} · {ex2_succ['svae48_native']['adjust_bottle']['successes']} of 60 (native · wide · S-VAE); 2 tasks at 0", state="done", href="ex2"),
        dict(id="B0", parent="EX2", title="Fresh-scene evaluation baseline", text=base_state[1], state=base_state[0], href="base"),
        dict(id="Q2", parent="Q", title="Does RAE's noise shift help?", text="shift 5 → 16.43", state="root", href=None),
        dict(id="EX0", parent="Q2", title="ex0-shift: MSE audit (corrected)", text=f"unweighted MSE ratio {rmin:.3f}–{rmax:.3f}; old 1.7× reading retracted", state="done", href="shift"),
        dict(id="H1", parent="EX0", title="H1 ex3-a: shift16, native head", text=h1_state[1], state=h1_state[0], href="h1"),
        dict(id="Q3", parent="Q", title="Does the RAE wide head help?", text="native head → 3072-wide head", state="root", href=None),
        dict(id="H2", parent="Q3", title="H2 ex2-b: wide head on fresh scenes", text=h2_state[1], state=h2_state[0], href="h2"),
        dict(id="H3A", parent="H2", title="H3a ex3-b: wide vs native at shift16", text=h3a_state[1], state=h3a_state[0], href="h3"),
        dict(id="H3B", parent="H1", title="H3b ex3-b: shift16 vs 5 under wide head", text=h3b_state[1], state=h3b_state[0], href="h3"),
        dict(id="Q4", parent="Q", title="Missing control: Wan-VAE latent arm", text="not started; needs user decision", state="wait", href="vae"),
    ]

    # ---- views
    plan_html = "<pre style='white-space:pre-wrap;font-size:.85rem'>" + esc(plan_text) + "</pre>"
    views = []
    views.append(dict(id="plan", title="Registered plan (PLAN_ex3_20261017.md, committed before launch)", html=plan_html + dl("media/PLAN_ex3_20261017.md", "PLAN_ex3_20261017.md")))

    pf_rows = []
    for v, r in preflights.items():
        if r:
            g = r["gradients"]
            pf_rows.append([v, "pass" if r.get("complete") else "fail", f"{g['video_backbone.dit.patch_embedding.weight']:.4f}", f"{g['action']['norm']:.4f}", f"{g.get('video_backbone.dit.rae_head.raw_projection.weight', float('nan')):.4f}" if "rae_head" in str(g) else "n/a", "yes" if r.get("serialization_exact") else "no", f"{r['peak_gib']:.1f}"])
    status_html = ("<h3>Status</h3>" + table(["arm", "seed", "GPU", "started (CDT)", "finished", "state"], trows)
                   + "<h3>L1 gate on this host (3-step preflight)</h3>" + table(["arm", "result", "trunk input grad", "action grad", "wide-head grad", "serialization exact", "peak GiB"], pf_rows))

    def result_table(arm):
        rows = []
        for task in TASKS:
            c = cnt[arm][task]
            rows.append([task, f"{c['successes']} / {c['attempted']}" if c["attempted"] else "pending", pct(c["successes"], c["attempted"]) if c["attempted"] else "—", wilson(c["successes"], c["attempted"]) if c["attempted"] else "—",
                         " / ".join(f"{a}/{b}" for a, b in c["by_seed"].values()) if c["attempted"] else "—", c["failures"] if c["attempted"] else "—"])
        return table(["task", "successes / attempted (3 seeds × 20 fresh scenes)", "rate", "Wilson 95 %", "per seed 42 / 43 / 44", "technical failures"], rows)

    def contrast_table(cid):
        if not (summary and summary.get("complete") and cid in summary.get("contrasts", {})):
            return "<p class='muted'>Contrast pending: the cohort is scored only when all 900 attempts are terminal (score.py).</p>"
        c = summary["contrasts"][cid]
        lo, hi = c["scene_cluster_paired_95pct_interval"]
        rows = [[esc(c["candidate"]), esc(c["baseline"]), f"{100 * c['task_balanced_success_difference']:+.1f} pp", f"[{100 * lo:+.1f}, {100 * hi:+.1f}]",
                 " / ".join(f"{100 * v:+.1f}" for v in c["by_training_seed"].values()), "supported" if c["supported"] else ("refuted" if c["refuted"] else "undecided")]]
        return table(["candidate", "baseline", "task-balanced paired Δ", "scene-cluster bootstrap 95 %", "per training seed (pp)", "verdict"], rows) + dl("media/success-summary.json", "success-summary.json")

    common_lock = ("Same 765/76 demonstrations (cache.pt sha256 2f8419ff…), common per-seed initialization, paired batch schedule and noise streams, 20,000 steps, final checkpoint only; "
                   "same 60 fresh scenes, native success rule, 600 s per episode, one host and renderer for every arm.")
    views.append(dict(id="h1", title="H1 — ex3-a: video shift 5 → 16.4317 with the native head", html=five_parts(
        "RAE (Zheng et al. 2025) prescribes a dimension-dependent noise shift α = √(m/4096); for our 1,105,920-dim raw768 latent that is 16.43, while OpenWAM uses 5. The corrected 2026-10-10 audit found no fit difference (unweighted MSE ratio %.3f–%.3f), so success is the only remaining test." % (rmin, rmax),
        "Does training raw768_native with shift_video 16.4317 instead of 5 change paired fresh-scene success?",
        "<code>shift_video: 5.0 → 16.4317</code> in the training config; action shift stays 5. " + common_lock,
        "Supported if Δ ≥ +5 pp and the 95 % interval excludes 0; refuted if Δ ≤ −5 pp and the interval excludes 0; otherwise undecided. Prediction: no detectable change (|Δ| < 5 pp).",
        "Supported → the shift prescription transfers to actions and becomes the default for the raw768 route. Refuted → 5 stays and the mechanism is marked as not transferring. Undecided → the raw768 route is not pursued further through the shift; the next decision is the missing Wan-VAE control.")
        + status_html + "<h3>Result: raw768_native_shift16 (candidate)</h3>" + result_table("raw768_native_shift16") + "<h3>Result: raw768_native (baseline, re-evaluated on the same fresh scenes)</h3>" + result_table("raw768_native")
        + "<h3>Contrast H1</h3>" + contrast_table("H1_shift16") + "<h3>Videos (policy input: head | left wrist | right wrist, one frame per policy call)</h3>" + video_grid(cards, "raw768_native_shift16") + "<h3>Baseline videos</h3>" + video_grid(cards, "raw768_native")))
    views.append(dict(id="h2", title="H2 — ex2-b: RAE wide head at shift 5, re-evaluated on fresh scenes", html=five_parts(
        "ex1 showed the wide head cuts held-out video loss 3.6× without changing success; ex2 (60 old scenes) read 24/60 vs 32/60 on adjust_bottle (paired sign test p ≈ 0.10). Those scenes were used repeatedly, so ex3 re-measures on fresh scenes.",
        "Does the 3072-wide shallow head, with shift 5, change paired fresh-scene success?",
        "the four <code>rae_head_*</code> config lines (width 3072, 24 heads, depth 1, MLP 2). " + common_lock + " Checkpoints are the ex2 ones (sha256 verified before each cell).",
        "Same ±5 pp / interval criterion. Prediction: zero or negative (−5 … 0 pp).",
        "Refuted → the wide head alone is closed. Undecided → stays a non-result. Supported → contradicts ex2 and would need a third independent cohort before any claim.")
        + "<h3>Result: raw768_wide</h3>" + result_table("raw768_wide") + "<h3>Contrast H2</h3>" + contrast_table("H2_widehead") + "<h3>Videos</h3>" + video_grid(cards, "raw768_wide")))
    views.append(dict(id="h3", title="H3 — ex3-b: wide head and shift 16.4317 together (two single-variable contrasts)", html=five_parts(
        "The RAE paper uses the wide head together with the shift; either might need the other. ex3-b is the fourth cell of the 2×2 design.",
        "H3a: at shift 16.4317, does the wide head change success (vs raw768_native_shift16)? H3b: under the wide head, does shift 16.4317 change success (vs raw768_wide)?",
        "H3a changes only the head; H3b changes only the shift. " + common_lock,
        "Same ±5 pp / interval criterion for each contrast. The two-variable diagonal (wide+shift16 vs native) is reported but labelled confounded. Prediction: undecided.",
        "If H3a or H3b is supported while H1 and H2 are not, the effect is an interaction and needs a replication cohort before being used. If none is supported, the RAE transfer to this policy is closed at this scale.")
        + "<h3>Result: raw768_wide_shift16</h3>" + result_table("raw768_wide_shift16") + "<h3>Contrast H3a</h3>" + contrast_table("H3a_widehead_at_shift16") + "<h3>Contrast H3b</h3>" + contrast_table("H3b_shift16_at_widehead")
        + "<h3>Confounded diagonal (two variables)</h3>" + contrast_table("C_combination_two_variables") + "<h3>Videos</h3>" + video_grid(cards, "raw768_wide_shift16")))
    views.append(dict(id="base", title="Fresh-scene evaluation baseline (frozen before any policy is scored)", html=
        "<p>20 expert-feasible scenes per task, taken in order from 40 fixed candidate seeds (6200000+, 6201000+, 6202000+). The expert planner, not any learned policy, decides feasibility; infeasible candidates are retained in the manifest. Overlap with every known training/evaluation seed list was checked (<code>seed-exclusion.json</code>).</p>"
        + table(["task", "accepted", "attempted candidates", "state"], srows)
        + "<h3>Success rule</h3><p>RoboTwin's native per-task success check at any step within the native step limit (400 / 800 / 700 actions); an episode is a technical failure if the initial render deviates more than 1 uint8 level from the frozen reference, if the 600 s wall limit is hit, or if the simulator errors; technical failures count as unsuccessful in every contrast and are reported separately.</p>"
        + "<h3>Known gaps</h3><ul><li>handover_block and place_object_basket were at zero for every arm in ex2; if they stay at zero here they carry no information and the contrasts rest on adjust_bottle alone.</li><li>Rendering on this host uses OIDN 2.3.3 (the only denoiser that runs on Blackwell); the policies were trained on data collected with the same denoiser on gpu-02.</li><li>Three training seeds; generalization over training randomness is not established.</li></ul>"
        + "<h3>S-VAE 48 arm (secondary)</h3>" + result_table("svae48_native") + contrast_table("S_representation") + video_grid(cards, "svae48_native")))
    views.append(dict(id="ex2", title="ex2 — 11× data and 5× steps, same recipe (rae-policy-expanded-20261006, done 2026-10-08)", html=five_parts(
        "ex1 could not separate arms: 13 successes in 126 cells. The question was whether data volume was the bottleneck.",
        "With 765 train demos and 20,000 steps, do the two zero tasks move, and do the three representations differ?",
        "Demo count and steps were raised together (confounded by design; both marked). Arms, seeds, evaluation scenes unchanged.",
        "Per-task success on 60 scenes × 3 seeds; paired sign test between arms.",
        "Observed: both zero tasks stayed at zero; 768-dim vs 48-dim indistinguishable; wide head lower (p ≈ 0.10, not below the registered threshold). Decision taken: stop scaling data; test the RAE mechanisms directly (ex3).")
        + ex2_tables + sources_html([str(REC / "expanded-20261006-audit-20261009.json")])))
    views.append(dict(id="ex1", title="ex1 — three representations, 66 demos, 4,000 steps (rae-policy-20261001)", html=
        "<p>First controlled comparison of raw768 native, raw768 wide head and S-VAE 48 (3 seeds, 15 scenes × 3 tasks). Numbers below are copied from the 2026-10-02 report table in EXPERIMENT_PLAN.md §1 (the study's ledger lives on gpu-02).</p>"
        + table(["arm", "closed-loop successes / 42 valid cells"], ex1_rows)
        + "<p>Reading: all 13 successes came from adjust_bottle; the wide head lowered dev video loss 3.6× without changing success; cheap proxies (dev losses) did not predict the endpoint.</p>"))
    views.append(dict(id="official", title="Official checkpoint on the same 60 scenes (Torch cluster, 2026-09-29)", html=
        "<p>Question: are the two zero tasks solvable at all under our evaluation? The released OpenWAM-Alpha-Sim-RoboTwin-Full checkpoint (118,655 steps, Wan-VAE latent, 30-layer DiT) was run on byte-identical scenes with the same clean condition.</p>" + official_table
        + f"<p>{esc(official['answer'])}</p>" + sources_html([str(REC / "official-baseline-20261002-result.json")])))
    views.append(dict(id="render", title="Renderer divergence: OIDN 2.3.3 vs the 2.0.1 reference renders", html=
        f"<p>Same physics state, same camera: renders on gpu-02 differ from the reference PNGs by {render_ref['min']:.2f}–{render_ref['max']:.2f} uint8 levels (median {render_ref['median']:.2f}). Torch reproduced the deviation to 3–4 decimals by preloading OIDN 2.3.3 on an L40S, so the cause is the denoiser version, not the GPU or driver. OIDN 2.0.1 cannot run on Blackwell (no sm_120 kernels). Consequence: every arm in ex2/ex3 is evaluated with the same 2.3.3 renderer it was trained with; the official checkpoint still reached 58/60 under it.</p>"
        + sources_html([str(REC / "oidn-ab-20261003.json"), str(G / "OIDN_AB_RESULT_20261003.md")])))
    views.append(dict(id="shift", title="ex0-shift — the 2026-10-03 shift experiment and its 2026-10-10 correction", html=
        "<p>Three seeds were trained at shift 16.4317 (4,000 steps, 66 demos) and read as 1.7× worse video loss at every σ. The 2026-10-10 audit showed the old evaluator kept each arm's training loss weights, which differ by shift; dividing them out leaves the unweighted MSE ratios below. The 'shift path is closed' conclusion was retracted; closed-loop success (H1) is the remaining test.</p>"
        + shift_table + "<p>Limits recorded with the audit: " + "; ".join(esc(x) for x in load(REC / "rae-shift-weight-audit-20261010/corrected-curves.json")["limits"]) + "</p>"
        + sources_html([str(REC / "rae-shift-weight-audit-20261010/corrected-curves.json"), str(G / "RAE_SHIFT_RESULT.md")])))
    views.append(dict(id="vae", title="Missing control: Wan-VAE 48-dim latent arm (not started)", html=
        "<p>RAE's claim is 'representation autoencoder instead of VAE'. Every arm so far is DINOv3-based; the VAE-latent arm with the same backbone, steps, data and seeds has not been run. Proposed on 2026-10-02 (stage 2) and 2026-10-07 (official-standard recipe); waiting for the user's decision. Estimated 3 trainings × ~1 h plus 3 evaluation cells.</p>"))

    # ---- home view
    comp_rows = []
    for arm in ARMS:
        st = arm_state(arm)
        href = {"raw768_native": "h1", "raw768_native_shift16": "h1", "raw768_wide": "h2", "raw768_wide_shift16": "h3", "svae48_native": "base"}[arm]
        c = cnt[arm]
        res = " / ".join(f"{c[t]['successes']}/{c[t]['attempted']}" for t in TASKS) if c["all"]["attempted"] else "pending"
        ex2v = ex2_succ.get(arm)
        caveat = (f"ex2 (old scenes): " + " / ".join(f"{ex2v[t]['successes']}/{ex2v[t]['valid']}" for t in TASKS)) if ex2v else "new in ex3"
        comp_rows.append([f'<a href="#{href}">{esc(ARM_LABEL[arm])}</a>', esc(st[1]), res, caveat])
    done_train = sum(1 for k, c in train_claims.items() if k.startswith("train-") and c.get("status") == "complete")
    summary_html = (
        "<p><b>Success rule:</b> RoboTwin's own per-task check within the native step limit, on 20 fresh expert-feasible scenes per task, 3 training seeds, paired across arms; technical failures count as failures.</p>"
        + table(["arm", "state", "fresh scenes: adjust_bottle / handover_block / place_object_basket", "earlier reading"], comp_rows)
        + f"<p><b>Now:</b> {done_train}/6 ex3 trainings complete; {len(recs)}/900 evaluation attempts recorded; "
        + ("final scoring done." if summary and summary.get("complete") else "final scoring pending.") + f" Deadline 2026-10-18 12:00 CDT. Host shenlong-gpu-01, 8 GPUs. Page generated {now.strftime('%Y-%m-%d %H:%M')} CDT from run files.</p>"
        + "<p><b>Known gaps:</b> two tasks were at 0 for every arm in ex2; no Wan-VAE control arm yet; 3 seeds only. <a href='#plan'>Registered plan</a> · <a href='#base'>evaluation baseline</a>.</p>")

    # ---- data files for download
    MEDIA.mkdir(parents=True, exist_ok=True)
    (MEDIA / "PLAN_ex3_20261017.md").write_text(plan_text)
    media["media/PLAN_ex3_20261017.md"] = str(MEDIA / "PLAN_ex3_20261017.md")
    if summary:
        (MEDIA / "success-summary.json").write_text(json.dumps(summary, indent=2))
        media["media/success-summary.json"] = str(MEDIA / "success-summary.json")

    spec = dict(title="OpenWAM RAE Logic Tree", t0=t0.isoformat(), now=now.isoformat(), summary_html=summary_html, tree=tree, lanes=lanes, milestones=milestones, views=views, media=media)
    (HERE / "spec.json").write_text(json.dumps(spec, indent=1))
    (HERE / "sources.json").write_text(json.dumps(sorted(set(SOURCES)), indent=1))
    bspec = importlib.util.spec_from_file_location("builder", BUILDER)
    builder = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(builder)
    builder.build(spec, PAGE)
    # incremental publish list: files not yet published or changed since
    published = load(HERE / "published.json", {}) or {}
    pending = {}
    for pub, src in media.items():
        st = os.stat(src)
        key = f"{st.st_size}:{int(st.st_mtime)}"
        if published.get(pub) != key:
            pending[pub] = src
    (HERE / "pending-files.json").write_text(json.dumps(pending, indent=1))
    print(f"media total {len(media)}, pending publish {len(pending)}, sources {len(set(SOURCES))}")


if __name__ == "__main__":
    main()
