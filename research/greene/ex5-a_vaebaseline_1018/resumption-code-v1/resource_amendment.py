"""Explicit resource-only continuation; original scientific evidence is immutable."""
import datetime as dt
import json
import math
from pathlib import Path

from build_cache import sha

ROOT = Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
AUTH = ROOT / 'resumption-20261019-v1/authorization.json'
MANIFEST = ROOT / 'resumption-code-v1-manifest.json'


def validate_authorization(a, now=None):
    assert a['status'] == 'authorized_resource_resumption'
    assert a['seeds'] == [42, 43] and a['gpus'] == [1, 2]
    assert a['evaluation_limits_seconds'] == {'42': 10920, '43': 10920, '44': 10800}
    assert a['study_limit_seconds'] == 73800 and a['fixed_remaining_episodes_per_seed'] == 45
    assert a['checkpoint_steps'] == 20000 and a['no_retraining'] and a['no_automatic_retry']
    start, end = (dt.datetime.fromisoformat(a[k]) for k in ['start', 'deadline'])
    assert start.tzinfo is not None and end.tzinfo is not None
    assert (end - start).total_seconds() == a['window_seconds'] == 21600
    assert dt.datetime.fromisoformat(a['allocation_end']) == end - dt.timedelta(seconds=60)
    if now is not None:
        assert start <= now < end, 'Outside authorized resumption window'
    return a


def limit(p, seed):
    values = p.get('evaluation_limits_seconds')
    if values is None:
        return p['evaluation_seconds_per_seed']
    assert values == {'42': 10920, '43': 10920, '44': 10800}
    return values[str(seed)]


def check_remaining(cap, prior, remaining, projected=None):
    assert all(math.isfinite(x) for x in [cap, prior, remaining])
    assert 0 < prior <= 3600 and remaining > 60
    assert math.isclose(prior + remaining, cap, abs_tol=1e-6)
    if projected is not None:
        assert math.isfinite(projected) and 0 < projected < remaining - 60


def apply(p, source, require_current_window=True):
    a = validate_authorization(json.loads(AUTH.read_text()),
                               dt.datetime.now(dt.timezone.utc) if require_current_window else None)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['authorization_sha256'] == sha(AUTH)
    assert Path(manifest['runtime']) == source == ROOT / 'resumption-code-v1'
    for path, digest in {**a['evidence_sha256'], **manifest['files']}.items():
        assert sha(path) == digest, path
    assert str(source / 'resource_amendment.py') in manifest['files']
    assert p['evaluation_seconds_per_seed'] == 10800 and p['max_occupied_seconds'] == 73800
    assert p['deadline'] == '2026-10-18T17:00:00+00:00'
    prior = json.loads((ROOT / 'corrected-full-training-release-v1.json').read_text())['prior_occupied_gpu_seconds']
    for phase, seeds in [('training', [42, 43, 44]), ('pilot_evaluation', [42, 43, 44]), ('full_evaluation', [44])]:
        for seed in seeds:
            path = ROOT / 'phase-claims' / f'{phase}-seed{seed}.json'
            assert str(path) in a['evidence_sha256']
            claim = json.loads(path.read_text())
            assert claim['complete'] and claim['exit_code'] == 0
            assert sha(claim['result_path']) == claim['result_sha256']
            assert math.isfinite(claim['occupied_seconds']) and claim['occupied_seconds'] > 0
            prior += claim['occupied_seconds']
    reserved = 0
    for seed in a['seeds']:
        pilot = json.loads((ROOT / 'phase-claims' / f'pilot_evaluation-seed{seed}.json').read_text())
        reserved += a['evaluation_limits_seconds'][str(seed)] - pilot['occupied_seconds']
    assert prior + reserved <= a['study_limit_seconds'], 'Global cap including both reserved evaluations exceeded'
    p.update(deadline=a['deadline'], evaluation_limits_seconds=a['evaluation_limits_seconds'],
             resource_authorization=str(AUTH), resource_authorization_sha256=sha(AUTH),
             authorized_gpus=a['gpus'], authorized_full_seeds=a['seeds'])
    return p, manifest['files']
