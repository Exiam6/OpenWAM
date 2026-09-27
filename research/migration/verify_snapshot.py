"""Verify archived source and experiment bytes without executing experiments."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / 'research/migration/snapshot.json').read_text())
failures = []
for relative, expected in manifest['files_sha256'].items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(relative)
print(json.dumps({'checked': len(manifest['files_sha256']), 'failures': failures}, indent=2))
raise SystemExit(bool(failures))
