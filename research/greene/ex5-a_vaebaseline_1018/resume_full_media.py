"""Bounded CPU continuation of the expired media queue, preserving its evidence."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from build_cache import sha
from full_media import ROOT, available, specs, stop, write
from prepare_media_resume import verify_reuse

SOURCE = Path(__file__).parent
INVENTORY = ROOT / 'resumption-preparation-20261019-v1/verified-media-reuse-inventory.json'


def initial_records(p, inventory):
    for path, digest in inventory['source_sha256'].items():
        assert sha(path) == digest, path
    original = json.loads((ROOT / 'full-media/queue.json').read_text())
    assert original['status'] == 'failed' and not original['active']
    assert not Path(f"/proc/{original['pid']}").exists()
    keys = {r['key'] for r in inventory['records']}
    current = [s for s in specs(p, available()) if s['key'] in keys]
    verified = verify_reuse(inventory['records'], current)
    assert len(verified) == original['completed'] == inventory['verified_pairs'] == 180
    return [json.loads(Path(r['metadata']).read_text()) for r in verified.values()]


def pending_specs(candidates, seen):
    assert len({s['key'] for s in candidates}) == len(candidates)
    pending = []
    for spec in candidates:
        if spec['key'] in seen:
            continue
        output = Path(spec['output'])
        assert not any(output.with_suffix(ext).exists() for ext in ['.json', '.mp4', '.jpg']), 'Preserve existing/partial output'
        pending.append(spec)
    return pending


def main():
    sys.path.insert(0, str(ROOT / 'resumption-code-v1'))
    from corrected_protocol import load_registration
    p = load_registration()
    manifest_path = ROOT / 'resumption-evidence-code-v1-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert Path(manifest['runtime']) == SOURCE
    assert manifest['authorization_sha256'] == p['resource_authorization_sha256']
    for path, digest in manifest['files'].items():
        assert sha(path) == digest, path
    inventory = json.loads(INVENTORY.read_text())
    assert sha(INVENTORY) == manifest['reuse_inventory_sha256']
    records = initial_records(p, inventory)
    root = ROOT / 'full-media-continuation-v1'
    root.mkdir(exist_ok=False)
    assert not (ROOT / 'full-media/manifest.json').exists()
    state = dict(pid=os.getpid(), status='starting', planned=360, reused=180,
                 created=dt.datetime.now(dt.timezone.utc).isoformat(),
                 inventory_sha256=sha(INVENTORY), authorization_sha256=p['resource_authorization_sha256'])
    active, seen = {}, {r['key'] for r in records}
    def save():
        state.update(time=dt.datetime.now(dt.timezone.utc).isoformat(), completed=len(records),
                     active={k:dict(pid=v[0].pid, started_monotonic=v[1]) for k,v in active.items()})
        write(root / 'queue.json', state)
    def terminated(*_):
        raise RuntimeError('Media continuation terminated')
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    save()
    try:
        while len(records) < 360:
            assert dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(p['deadline'])
            assert not any(Path(x).exists() for x in p['pause_paths'])
            for key, (child, start, spec) in list(active.items()):
                code = child.poll()
                if code is None:
                    assert time.monotonic() - start < 180
                    continue
                assert code == 0, (key, code)
                row = json.loads(Path(spec['output']).read_text())
                assert row['key'] == key and row['full_decode_passed']
                records.append(row)
                del active[key]
            candidates = specs(p, available())
            state['available_pairs'] = len(candidates)
            for spec in pending_specs(candidates, seen)[:max(0, 2-len(active))]:
                key = spec['key']
                spec_path = root / 'specs' / f'{key}.json'
                spec_path.parent.mkdir(parents=True, exist_ok=True)
                with spec_path.open('x') as handle:
                    json.dump(spec, handle, indent=2)
                log = root / 'logs' / f'{key}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                with log.open('x') as handle:
                    child = subprocess.Popen([sys.executable, '-u', str(SOURCE / 'paired_video.py'), '--spec', str(spec_path)],
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, env=env)
                active[key] = child, time.monotonic(), spec
                seen.add(key)
            state['status'] = 'encoding' if active else 'waiting_audited_results'
            save()
            if len(records) < 360:
                time.sleep(1 if active else 15)
        assert len({r['key'] for r in records}) == 360
        result = dict(complete=True, paired_videos=360, records=records,
                      continuation_authorization_sha256=p['resource_authorization_sha256'],
                      original_queue_sha256=sha(ROOT / 'full-media/queue.json'),
                      reuse_inventory_sha256=sha(INVENTORY))
        with (ROOT / 'full-media/manifest.json').open('x') as handle:
            json.dump(result, handle, indent=2)
        state['status'] = 'complete'
        save()
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        for child, _, _ in active.values():
            stop(child)
        state['finished'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
