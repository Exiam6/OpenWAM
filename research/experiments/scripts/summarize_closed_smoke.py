#!/usr/bin/env python3
import hashlib,json,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/closed-smoke-20260921'
assert (RUN/'exit-code.txt').read_text().strip()=='0'
log=ROOT/'logs/closed-smoke-client.log';text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text())
rows=[dict(successes=int(a),episodes=int(b),seed=int(c)) for a,b,c in re.findall(r'Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)',text)]
assert len(rows)==5 and [r['episodes'] for r in rows]==list(range(1,6)),rows
assert all(0<=r['successes']<=r['episodes'] for r in rows)
files=list((RUN/'runtime/eval_result').rglob('_result.txt'));assert len(files)==1,files
native=files[0].read_text();raw_rate=float(native.strip().splitlines()[-1]);last=rows[-1]
summary={**last,'success_rate':last['successes']/last['episodes'],'progress':rows,
         'scope':'Published DINO/S-VAE policy only; setup smoke, not auxiliary-loss or full-paper evaluation',
         'task':'pick_dual_bottles','mode':'demo_clean','seed_argument':0,'instruction_type':'unseen','step_limit':400,
         'checkpoint_revision':'af1595c8ee54955116abbc0b0ffa17fa48deb691','robotwin_commit':'0aeea2d669c0f8516f4d5785f0aa33ba812c14b4',
         'compile_enabled':False,'denoise_steps':10,'dit_cache_enabled':True,'planner_fallback':False,
         'original_result_file_rate':raw_rate,'original_result_file_text':native,
         'denominator_bug_observed':abs(raw_rate-last['successes']/100)<1e-12 and last['episodes']!=100,
         'client_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
         'setup_exceptions_in_log':text.count('[eval_policy_wrapper] exception in'),
         'exit_code':0,'summarized_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
(RUN/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
