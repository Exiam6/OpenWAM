"""Read-only media reuse inventory. This module cannot authorize or launch work."""
import argparse
import datetime as dt
import json
from pathlib import Path

from build_cache import sha
from paired_video import CAMERAS, checked_pair


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_reuse(entries, specs):
    """Recheck archived bytes and their correspondence to current audited pairs."""
    by_key = {s['key']: s for s in specs}
    require(len(by_key) == len(specs), 'Duplicate available pair')
    records = {}
    for entry in entries:
        key = entry['key']
        require(key in by_key and key not in records, 'Unknown or duplicate reused pair')
        spec = by_key[key]
        require(Path(entry['metadata']) == Path(spec['output']), 'Unexpected metadata path')
        for field in ['metadata', 'video', 'preview']:
            require(sha(entry[field]) == entry[field + '_sha256'], f'Changed {field}: {key}')
        row = json.loads(Path(entry['metadata']).read_text())
        for field in ['key', 'kind', 'comparison', 'seed', 'task', 'scene_seed', 'labels', 'left', 'right']:
            require(row[field] == spec[field], f'Changed pairing field {field}: {key}')
        for field in ['video', 'preview']:
            require(row[field] == entry[field] and row[field + '_sha256'] == entry[field + '_sha256'],
                    f'Changed artifact reference: {key}')
        results = checked_pair(spec)
        counts = [r['steps'] + 1 for r in results]
        count = max(counts)
        samples = sorted(set([0, count // 5, 2 * count // 5, 3 * count // 5, 4 * count // 5, count - 1]))
        require(row['full_decode_passed'] is True, 'Missing original decode attestation')
        require(row['side_frames'] == counts and row['frames'] == count, 'Frame count mismatch')
        require(row['preview_indices'] == samples and row['camera_order'] == CAMERAS,
                'Preview or camera mismatch')
        require(row['left_success'] == results[0]['success'] and row['right_success'] == results[1]['success'],
                'Outcome mismatch')
        records[key] = dict(entry)
    require(set(records) == set(by_key), 'Available pairs are not fully accounted for')
    return records


def prepare(root, audit_path, audit_sha256):
    import full_media

    require(root == full_media.ROOT, 'Use the registered study root')
    require(sha(audit_path) == audit_sha256, 'Changed inventory audit')
    audit = json.loads(audit_path.read_text())
    require(audit['available_evidence_verification_passed'] is True, 'Unverified inventory')
    queue_path = root / 'full-media/queue.json'
    queue = json.loads(queue_path.read_text())
    require(queue['status'] == 'failed' and not queue['active'], 'Original queue is not terminal')
    require(not Path(f"/proc/{queue['pid']}").exists(), 'Original queue PID still exists')
    registration_path = root / 'gpu-registration.json'
    registration = json.loads(registration_path.read_text())
    expected = {
        f'{comparison}/seed{seed}/{task}/scene-{scene}'
        for comparison in full_media.COMPARISONS
        for seed in registration['seeds']
        for task, scenes in registration['scenes'].items()
        for scene in scenes
    }
    require(len(expected) == 360, 'Unexpected registered endpoint')
    specs = full_media.specs(registration, full_media.available())
    records = verify_reuse(audit['paired_media'], specs)
    require(set(records) <= expected, 'Pair outside registered endpoint')
    require(len(records) == queue['completed'] == audit['available_pairs'], 'Inventory count mismatch')
    return dict(
        time=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='verified_reuse_inventory_only', launch_authorized=False,
        scope='No encoding, evaluation, new decode, or deadline amendment performed. Revalidate before reuse.',
        complete_study=False, planned_pairs=360, verified_pairs=len(records),
        missing_pairs=len(expected - records.keys()), missing_keys=sorted(expected - records.keys()),
        records=[records[k] for k in sorted(records)],
        source_sha256={str(p): sha(p) for p in [audit_path, queue_path, registration_path]},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--audit-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing inventory')
    result = prepare(args.root, args.audit, args.audit_sha256)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ['status', 'verified_pairs', 'missing_pairs', 'launch_authorized']}))


if __name__ == '__main__':
    main()
