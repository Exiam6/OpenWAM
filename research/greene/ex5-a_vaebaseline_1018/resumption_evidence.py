"""Archive explicit continuation lineage without relabeling the expired attempt."""
import json
import math
from pathlib import Path

from build_cache import sha


def evaluation_cap(root, seed):
    path = Path(root) / 'resumption-20261019-v1/authorization.json'
    if not path.exists():
        return 10800
    auth = json.loads(path.read_text())
    assert auth['status'] == 'authorized_resource_resumption'
    assert auth['evaluation_limits_seconds'] == {'42':10920, '43':10920, '44':10800}
    assert auth['study_limit_seconds'] == 73800
    return auth['evaluation_limits_seconds'][str(seed)]


def evidence(root):
    root = Path(root)
    run = root / 'resumption-20261019-v1'
    if not (run / 'authorization.json').exists():
        return []
    paths = set()
    def include(path, digest=None):
        path = Path(path)
        assert path.is_file(), path
        if digest is not None: assert sha(path) == digest, path
        paths.add(path)
        return json.loads(path.read_text()) if path.suffix == '.json' else None
    auth = include(run / 'authorization.json')
    assert auth['status'] == 'authorized_resource_resumption'
    assert auth['seeds'] == [42,43] and auth['gpus'] == [1,2]
    assert auth['no_retraining'] and auth['no_automatic_retry']
    for path, digest in auth['evidence_sha256'].items(): include(path, digest)
    for name in ['resumption-code-v1-manifest.json', 'resumption-evidence-code-v1-manifest.json']:
        manifest = include(root / name)
        assert manifest['authorization_sha256'] == sha(run / 'authorization.json')
        for path, digest in manifest['files'].items(): include(path, digest)
    for name in ['frozen-tests.json', 'evidence-tests.json']:
        tests = include(run / name)
        assert tests['passed'] and tests['tests'] > 0
    for name in ['batch-process.json', 'media-process.json', 'batch.log', 'media.log']:
        include(run / name)
    batch = include(root / 'batches/full_evaluation-42-43/result.json')
    assert batch['complete'] and batch['status'] == 'complete'
    assert batch['seeds'] == [42,43] and batch['gpus'] == [1,2]
    assert len(batch['children']) == 2 and all(c['complete'] and c['exit_code']==0 for c in batch['children'])
    for seed in [42,43]:
        release = include(root / f'evaluation-release-seed{seed}.json')
        assert release['resource_authorization_sha256'] == sha(run / 'authorization.json')
        assert release['evaluation_limit_seconds'] == evaluation_cap(root, seed)
        pilot = include(root / f'phase-claims/pilot_evaluation-seed{seed}.json', release['parent_claim_sha256'])
        assert math.isclose(pilot['occupied_seconds'] + release['remaining_seconds'], evaluation_cap(root,seed), abs_tol=1e-6)
        assert release['cost']['projected_remaining_seconds'] < release['remaining_seconds'] - 60
        for name in ['execution.json', 'run.log']: include(root / f'batches/full_evaluation-42-43/seed{seed}/{name}')
    old = include(root / 'full-media/queue.json')
    assert old['status'] == 'failed' and old['completed'] == 180 and not old['active']
    include(root / 'full-media-v8.log')
    queue = include(root / 'full-media-continuation-v1/queue.json')
    assert queue['status'] == 'complete' and queue['completed'] == 360 and queue['reused'] == 180 and not queue['active']
    assert queue['authorization_sha256'] == sha(run / 'authorization.json')
    media = include(root / 'full-media/manifest.json')
    assert media['complete'] and media['paired_videos'] == 360
    assert media['continuation_authorization_sha256'] == sha(run / 'authorization.json')
    assert media['original_queue_sha256'] == sha(root / 'full-media/queue.json')
    inventory = root / 'resumption-preparation-20261019-v1/verified-media-reuse-inventory.json'
    data = include(inventory, media['reuse_inventory_sha256'])
    assert data['verified_pairs'] == data['missing_pairs'] == 180
    assert queue['inventory_sha256'] == sha(inventory)
    for path, digest in data['source_sha256'].items(): include(path, digest)
    return sorted(paths)
