"""CPU-only combined patch review; no model fitting or simulation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/zifanz4/openwam-experiments')
OUT = Path('/data02/zifanz4/openwam-experiments/results/contribution-review-20260922')
BASE = '7c5861e45cfe1339a0323f0e0b03a3316c37971c'
PYTHON = '/data02/zifanz4/openwam-experiments/env/bin/python'
TOOLS = '/tmp/openwam-review-tools'
SITE = Path('/home/zifanz4/research-reports/dist/assets')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started = time.monotonic()
    combined = OUT / 'combined-source'
    combined.mkdir(exist_ok=False)
    sources = [SITE / (n + '-20260921') / 'summary.json' for n in ['night', 'control', 'groups', 'robustness', 'norm']]
    sources += list((ROOT / 'contribution').glob('*.patch'))
    original = {str(p): sha(p) for p in sources}
    patches = []
    for name in ['svae', 'robustness', 'trace']:
        tree = Path('/home/zifanz4/openwam-' + name + '-contribution')
        assert not subprocess.check_output(['git', '-C', str(tree), 'status', '--porcelain'])
        commit = subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip()
        diff = subprocess.check_output(['git', '-C', str(tree), 'diff', '--binary', BASE, commit])
        p = OUT / (name + '.patch')
        p.write_bytes(diff)
        if not patches:
            archive = subprocess.check_output(['git', '-C', str(tree), 'archive', BASE])
            subprocess.run(['tar', '-xf', '-', '-C', str(combined)], input=archive, check=True)
        subprocess.run(['git', 'apply', '--check', str(p)], cwd=combined, check=True)
        subprocess.run(['git', 'apply', str(p)], cwd=combined, check=True)
        files = subprocess.check_output(['git', '-C', str(tree), 'diff', '--name-only', BASE, commit], text=True).splitlines()
        for f in files:
            assert (combined / f).read_bytes() == (tree / f).read_bytes(), f
        patches.append({'name': name, 'commit': commit, 'base': BASE, 'sha256': sha(p), 'files': files,
                        'diffstat': subprocess.check_output(['git', '-C', str(tree), 'diff', '--shortstat', BASE, commit], text=True).strip()})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONPATH=str(combined) + ':' + TOOLS, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    tests = ['tests/test_svae_evaluation.py', 'tests/test_robotwin_scene_seeded_language.py', 'tests/test_policy_trace.py']
    checks = []
    commands = [
        ('cpu-tests', [PYTHON, '-m', 'pytest', '-q', '-c', '/dev/null', '--rootdir=' + str(combined), '--junitxml=' + str(OUT / 'tests.xml'), *tests]),
        ('ruff', [TOOLS + '/bin/ruff', 'check', *sorted({f for p in patches for f in p['files'] if f.endswith('.py')})]),
        ('environment', [PYTHON, '-c', 'import sys,torch,pytest,numpy;print(sys.version);print("torch",torch.__version__,"pytest",pytest.__version__,"numpy",numpy.__version__);print("cuda_available",torch.cuda.is_available())']),
    ]
    for name, command in commands:
        with (OUT / (name + '.log')).open('w') as f:
            result = subprocess.run(command, cwd=combined, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=120)
        checks.append({'name': name, 'command': command, 'exit_code': result.returncode, 'log': name + '.log'})
    unchanged = all(sha(Path(p)) == h for p, h in original.items())
    result = {'checked_at': datetime.datetime.now().astimezone().isoformat(), 'base': BASE, 'patches': patches, 'checks': checks,
              'elapsed_seconds': time.monotonic() - started, 'source_artifacts_unchanged': unchanged, 'source_hashes': original,
              'passed': unchanged and all(c['exit_code'] == 0 for c in checks),
              'limits': ['Targeted combined-source CPU checks only; full upstream dependency/GPU suite not run.',
                         'S-VAE CLI package bootstrap in complete upstream environment remains unchecked.',
                         'Existing scientific summaries reused; no new experiments or independent samples.']}
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['source_hashes', 'patches']}, indent=2))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
