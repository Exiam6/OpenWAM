"""Generic builder for a logic-tree progress page (TOP_PRIORITY/LOGIC_TREE.md), extracted 2026-10-16 from the G1 metric-fridge page.

Input: a JSON spec (see example_spec.json) that a project-specific script generates FROM RUN FILES (PLAN.md, training.jsonl, eval rows,
launch logs, video folders) - never by hand. Output: one self-contained index.html (hash-routed views, SVG logic tree and timeline with
2-D pan / wheel zoom / double-click reset / slider, download buttons via the artifact 'downloads' capability) plus files.json listing the
media to publish next to it. Usage: python3 build_logic_tree.py spec.json out_dir

Spec keys: title; summary_html (home view, keep <= 1000 CJK characters of reading); tree: [{id, parent, title, text, state(root|done|run|wait),
href(view id or null)}]; lanes: [{lane, label, start, end (ISO datetime or null), state, href, projected(bool)}]; milestones: [{at, label}];
t0 (ISO datetime of the axis origin); now (ISO); views: [{id, title, html}]; media: {published_path: source_path}; plan_url (optional).
Media paths referenced inside view html must be the published paths (e.g. media/x.mp4); a download button is `<button class="dl" data-src=".." data-name="..">Download</button>`.
"""
import html, json, re, sys, datetime as dt
from pathlib import Path

NW, NH, GX, GY = 236, 76, 22, 60      # node width/height, horizontal/vertical gaps of the tree
PX_H, LH, LEFT = 12.0, 46, 190        # timeline: pixels per hour, lane height, label column


def esc(s): return html.escape(str(s), quote=False)


def wrap(t, n): t = str(t); return [t[i:i+n] for i in range(0, len(t), n)][:3]


def tree_svg(nodes):
    kids, width, pos, depth = {}, {}, {}, {}
    for n in nodes: kids.setdefault(n['parent'], []).append(n['id'])
    roots = kids.get(None, [])
    def sw(nid):
        cs = kids.get(nid, []); w = max(NW, sum(sw(c) for c in cs)+GX*(len(cs)-1)) if cs else NW; width[nid] = w; return w
    def place(nid, x0, d):
        cs = kids.get(nid, []); total = sum(width[c] for c in cs)+GX*(len(cs)-1) if cs else 0
        pos[nid] = (x0+width[nid]/2, 60+d*(NH+GY)); depth[nid] = d; x = x0+(width[nid]-total)/2
        for c in cs: place(c, x, d+1); x += width[c]+GX
    x = 10
    for r in roots: sw(r); place(r, x, 0); x += width[r]+GX
    TW = int(x+10); TH = int(60+(max(depth.values())+1)*(NH+GY))
    by = {n['id']: n for n in nodes}
    out = [f'<svg id="lt" viewBox="0 0 {TW} {TH}" width="100%" height="{min(TH, 560)}" class="ltree panzoom" data-w="{TW}" data-h="{TH}" preserveAspectRatio="xMidYMin meet" role="img" aria-label="logic tree">',
           f'<rect x="0" y="0" width="{TW}" height="{TH}" fill="var(--card)"/>']
    for n in nodes:
        if n['parent']:
            px, py = pos[n['parent']]; cx, cy = pos[n['id']]
            out.append(f'<path d="M{px:.0f},{py+NH/2:.0f} C{px:.0f},{py+NH/2+GY/2:.0f} {cx:.0f},{cy-NH/2-GY/2:.0f} {cx:.0f},{cy-NH/2:.0f}" fill="none" stroke="var(--line)" stroke-width="2"/>')
    for n in nodes:
        cx, cy = pos[n['id']]; x0, y0 = cx-NW/2, cy-NH/2; cls = n.get('state', 'wait')
        body = (f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{NW}" height="{NH}" rx="8" class="nd {cls}"/><rect x="{x0:.0f}" y="{y0:.0f}" width="6" height="{NH}" rx="3" class="ndbar {cls}"/>'
                f'<text x="{x0+14:.0f}" y="{y0+20:.0f}" class="nt">{esc(n["title"])}</text>' + ''.join(f'<text x="{x0+14:.0f}" y="{y0+37+i*15:.0f}" class="ns">{esc(l)}</text>' for i, l in enumerate(wrap(n.get('text', ''), 35))))
        out.append(f'<a href="#{n["href"]}">{body}</a>' if n.get('href') else body)
    out.append('</svg>')
    return '<div class="tlwrap">' + ''.join(out) + '</div><p class="muted legend">drag to pan, wheel to zoom, double-click to reset; click a node for its experiment page.</p>'


def timeline_svg(lanes, milestones, t0, now):
    P = lambda s: dt.datetime.fromisoformat(s) if s else None
    T0, NOW = P(t0), P(now); segs = [dict(s, start=P(s.get('start')), end=P(s.get('end'))) for s in lanes]
    names = []
    for s in segs:
        if s['lane'] not in names: names.append(s['lane'])
    W = int(PX_H*144)+LEFT; HT = 40+LH*len(names)+30
    X = lambda t: LEFT+(t-T0).total_seconds()/3600*PX_H
    out = [f'<svg id="tl" viewBox="0 0 {W} {HT}" width="100%" height="{HT}" class="timeline panzoom" data-w="{W}" data-h="{HT}" data-cx="{X(NOW):.0f}" preserveAspectRatio="xMinYMin meet" role="img" aria-label="timeline">',
           f'<rect x="0" y="0" width="{W}" height="{HT}" fill="var(--card)"/>']
    for h in range(0, 145, 12):
        x = LEFT+h*PX_H; t = T0+dt.timedelta(hours=h)
        out.append(f'<line x1="{x:.0f}" y1="22" x2="{x:.0f}" y2="{HT-22}" stroke="var(--line)"/><text x="{x+2:.0f}" y="16" class="ct">{t.strftime("%m-%d %H:%M")}</text>')
    for i, name in enumerate(names):
        y = 40+i*LH; out.append(f'<text x="8" y="{y+LH/2+4:.0f}" class="lane">{esc(name)}</text><line x1="{LEFT}" y1="{y+LH:.0f}" x2="{W}" y2="{y+LH:.0f}" stroke="var(--line)"/>')
    for s in segs:
        if not s['start']: continue
        i = names.index(s['lane']); y = 40+i*LH+8; x1 = X(s['start']); x2 = X(s['end'] or NOW); cls = f"bar {s.get('state', 'wait')}{' proj' if s.get('projected') else ''}"
        bar = f'<rect x="{x1:.0f}" y="{y:.0f}" width="{max(4, x2-x1):.0f}" height="{LH-16}" rx="4" class="{cls}"/><text x="{x1+6:.0f}" y="{y+(LH-16)/2:.0f}" class="bl">{esc(s["label"])}</text>'
        out.append(f'<a href="#{s["href"]}">{bar}</a>' if s.get('href') else bar)
    for m in milestones or []:
        x = X(P(m['at'])); out.append(f'<line x1="{x:.0f}" y1="26" x2="{x:.0f}" y2="{HT-26}" stroke="var(--accent)" stroke-dasharray="4 3"/><text x="{x+3:.0f}" y="{HT-8}" class="ct">{esc(m["label"])}</text>')
    xn = X(NOW); out.append(f'<line x1="{xn:.0f}" y1="22" x2="{xn:.0f}" y2="{HT-22}" stroke="#d04a3a" stroke-width="2"/><text x="{xn+4:.0f}" y="{HT-16}" class="ct" fill="#d04a3a">now {NOW.strftime("%H:%M")}</text></svg>')
    return ('<div class="tlwrap" id="tlwrap">' + ''.join(out) + '</div>' + f'<input type="range" id="tl-slider" class="tlslider" min="0" max="{W}" value="0" step="1" aria-label="timeline position">'
            '<p class="muted legend">horizontal axis = time; drag, wheel-zoom, double-click to reset; solid = measured, dashed = projected; red = now.</p>')


CSS = """:root{--bg:#f5f4ef;--fg:#1e2622;--muted:#60706a;--accent:#1d6a8f;--line:#d6ddd8;--card:#ffffff;--code:#e9eeeb;--done:#2e7d4f;--run:#2b5f8f;--wait:#8a8a80}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#13181a;--fg:#e5ebe8;--muted:#9db0a8;--accent:#6fb6d9;--line:#2c3634;--card:#1a2224;--code:#222c2e;--done:#7fd3a3;--run:#8fc3ea;--wait:#9a9a90;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#13181a;--fg:#e5ebe8;--muted:#9db0a8;--accent:#6fb6d9;--line:#2c3634;--card:#1a2224;--code:#222c2e;--done:#7fd3a3;--run:#8fc3ea;--wait:#9a9a90;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,"PingFang SC","Noto Sans CJK SC",system-ui,sans-serif;line-height:1.6;padding-inline:16px;padding-block:18px 48px}
main{max-width:980px;margin:0 auto}h1{font-size:1.4rem;margin:0 0 8px;text-wrap:balance}h2{font-size:1.2rem;margin:0 0 10px}h3{font-size:1rem;margin:16px 0 6px;color:var(--accent)}p{margin:0 0 10px}.muted{color:var(--muted)}a{color:var(--accent)}
.tw{overflow-x:auto;margin:0 0 10px}table{border-collapse:collapse;font-size:.9rem;min-width:100%}th,td{border:1px solid var(--line);padding:5px 8px;vertical-align:top;text-align:left}th{background:var(--code)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}.card{border:1px solid var(--line);border-radius:8px;padding:8px;background:var(--card);min-width:0}
video,img{width:100%;max-width:100%;border-radius:4px;background:#000}button.dl{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:4px 10px;font-size:.85rem;cursor:pointer}button[disabled]{opacity:.5}button.dl[hidden]{display:none}
.legend{font-size:.85rem}.legend span{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:middle}.tlslider{width:100%;margin:6px 0 2px;accent-color:var(--accent)}
.tlwrap{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--card);touch-action:none;cursor:grab;user-select:none}.tlwrap:active{cursor:grabbing}.ct{font-size:11px;fill:var(--muted)}
.timeline .lane{font-size:13px;font-weight:600;fill:var(--fg)}.timeline .bl{font-size:12px;fill:#fff;dominant-baseline:middle;pointer-events:none}.timeline .bar.done{fill:var(--done)}.timeline .bar.run{fill:var(--run)}.timeline .bar.wait{fill:var(--wait)}
.timeline .bar.proj{stroke:var(--fg);stroke-dasharray:4 3;stroke-width:1.5;fill-opacity:.55}.timeline a:hover .bar{filter:brightness(1.12)}
.ltree .nd{fill:var(--bg);stroke:var(--line);stroke-width:1.5}.ltree .nd.root{stroke:var(--accent)}.ltree .ndbar.done{fill:var(--done)}.ltree .ndbar.run{fill:var(--run)}.ltree .ndbar.wait{fill:var(--wait)}.ltree .ndbar.root{fill:var(--accent)}
.ltree .nt{font-size:13px;font-weight:600;fill:var(--fg)}.ltree .ns{font-size:11px;fill:var(--muted)}.ltree a:hover .nd{stroke:var(--accent);stroke-width:2}.back{margin:0 0 8px}"""

JS = """(function(){
  const views = Array.from(document.querySelectorAll('section.view'));
  function route(){ const h = (location.hash || '#home').slice(1); const t = document.getElementById('v-' + h) ? h : 'home';
    views.forEach(v => { v.hidden = (v.id !== 'v-' + t); }); if (t !== 'home') window.scrollTo(0, 0); }
  window.addEventListener('hashchange', route); route();
})();
(function(){   // 2-D pan + wheel zoom around the cursor + double-click reset on every svg.panzoom; a drag < 6 px still counts as a click
  document.querySelectorAll('svg.panzoom').forEach(svg => {
    const W = +svg.dataset.w, HT = +svg.dataset.h, CX = svg.dataset.cx ? +svg.dataset.cx : null;
    let vw = Math.min(W, 1400), vh = HT * vw / W, vx = CX === null ? 0 : Math.max(0, Math.min(W - vw, CX - vw * 0.6)), vy = 0;
    if (svg.id === 'lt') { vw = W; vh = HT; vx = 0; vy = 0; }
    const home = [vx, vy, vw, vh]; const slider = svg.id === 'tl' ? document.getElementById('tl-slider') : null;
    function apply(){ svg.setAttribute('viewBox', vx + ' ' + vy + ' ' + vw + ' ' + vh); if (slider) { slider.max = Math.max(0, W - vw); slider.value = Math.max(0, Math.min(W - vw, vx)); } }
    if (slider) slider.addEventListener('input', () => { vx = +slider.value; apply(); });
    apply(); let drag = null, suppress = false;
    svg.addEventListener('pointerdown', e => { if (e.button !== 0) return; drag = {x: e.clientX, y: e.clientY, vx, vy, moved: false}; });
    window.addEventListener('pointermove', e => { if (!drag) return; const r = svg.getBoundingClientRect(); const dx = (e.clientX - drag.x) * vw / r.width, dy = (e.clientY - drag.y) * vh / r.height;
      if (Math.abs(e.clientX - drag.x) > 6 || Math.abs(e.clientY - drag.y) > 6) drag.moved = true; if (drag.moved) { vx = drag.vx - dx; vy = drag.vy - dy; apply(); } });
    window.addEventListener('pointerup', () => { if (drag) { suppress = drag.moved; drag = null; setTimeout(() => { suppress = false; }, 50); } });
    svg.addEventListener('click', e => { if (suppress) { e.preventDefault(); e.stopPropagation(); } }, true);
    svg.addEventListener('wheel', e => { e.preventDefault(); const r = svg.getBoundingClientRect(); const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height; const f = e.deltaY > 0 ? 1.15 : 1/1.15;
      const nvw = Math.max(200, Math.min(W * 3, vw * f)), nvh = nvw * vh / vw; vx += fx * (vw - nvw); vy += fy * (vh - nvh); vw = nvw; vh = nvh; apply(); }, {passive: false});
    svg.addEventListener('dblclick', () => { [vx, vy, vw, vh] = home; apply(); });
  });
})();
(async () => {   // download buttons through the artifact 'downloads' capability (publish with capabilities: {downloads: true})
  const downloads = await (window.claude && window.claude.use ? window.claude.use('downloads') : Promise.resolve(null));
  const btns = document.querySelectorAll('button.dl'); if (!downloads) { btns.forEach(b => b.hidden = true); return; }
  btns.forEach(b => b.addEventListener('click', async () => { b.disabled = true;
    try { const r = await fetch(b.dataset.src); const blob = await r.blob(); await downloads.save({filename: b.dataset.name, data: blob}); } catch (e) { }
    b.disabled = false; }));
})();"""


def build(spec, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    views = ''.join(f'<section class="view" id="v-{v["id"]}" hidden><p class="back"><a href="#top">&larr; home</a></p><h2>{esc(v["title"])}</h2>{v["html"]}</section>' for v in spec.get('views', []))
    tl = timeline_svg(spec.get('lanes', []), spec.get('milestones', []), spec['t0'], spec['now']) if spec.get('lanes') else ''
    page = (f'<title>{esc(spec["title"])}</title>\n<style>{CSS}</style>\n<main>\n<h1 id="top">{esc(spec["title"])}</h1>\n<section class="view" id="v-home">\n{spec.get("summary_html", "")}\n'
            + (f'<h2>Timeline (click a bar for the experiment page)</h2>{tl}' if tl else '')
            + '<h2>Logic tree (question &rarr; hypothesis &rarr; next step)</h2><p class="legend muted"><span style="background:var(--done)"></span>done <span style="background:var(--run)"></span>running <span style="background:var(--wait)"></span>waiting</p>'
            + tree_svg(spec['tree']) + f'</section>\n{views}\n</main>\n<script>\n{JS}\n</script>')
    (out/'index.html').write_text(page); (out/'files.json').write_text(json.dumps(spec.get('media', {}), indent=1))
    home = re.search(r'<section class="view" id="v-home">(.*?)</section>', page, re.S).group(1); txt = re.sub(r'<[^>]+>', '', home)
    print(f'wrote {out/"index.html"} ({len(page)} bytes); media files {len(spec.get("media", {}))}; CJK characters in the home view: {len(re.findall(chr(0x4e00) + "-" + chr(0x9fff), txt)) if False else len(re.findall("[一-鿿]", txt))}')


if __name__ == '__main__':
    build(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])
