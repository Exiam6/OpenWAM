"""Freeze the registered calibration pilot without starting GPU work."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

SOURCE = Path(__file__).resolve().parent
ROOT = Path('/home/zifanz4/openwam-runtime/ex4-f_nativecalibration_1017')
BASE = ROOT.parent / 'rae-factorial-20261010'
EXPERT = ROOT.parent / 'ex4-e_plannerpath_1017'
VARIANTS = ['raw768_native', 'svae48_native']
TASKS = ['adjust_bottle', 'handover_block', 'place_object_basket']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')

def main():
    assert not ROOT.exists(), 'No preparation overwrite'
    assert json.loads((EXPERT/'L1-gate.json').read_text())['passed']
    assert json.loads((EXPERT/'branch-audit.json').read_text())['passed']
    ROOT.mkdir()
    shutil.copytree(SOURCE, ROOT/'code', ignore=shutil.ignore_patterns('__pycache__'))
    baseline_freeze = json.loads((BASE/'eval-freeze.json').read_text())['files']
    checked = {}
    scripts = ['evaluate_policy.py','serve_policy.py','simulation_smoke.py','migration_render.py','collect_scene.py','execution_trace.py']
    configs = ['scene-protocol.json','eval-protocol.json','seed-exclusion.json']
    for name in scripts+configs:
        path = BASE/name
        assert sha(path) == baseline_freeze[str(path)], path
        checked[str(path)] = sha(path)
    # Model/deployment code is shared read-only, and checked again per cell.
    model = Path('/home/zifanz4/openwam-rae-research')
    for directory in ['openwam', 'configs']:
        for path in (model/directory).rglob('*'):
            if path.suffix in ('.py','.yaml','.json') and '__pycache__' not in path.parts:
                checked[str(path)] = sha(path)
    selected = {}
    for task in TASKS:
        manifest = json.loads((BASE/'scenes'/task/'manifest.json').read_text())
        assert manifest['complete'] and len(manifest['accepted']) == 20
        selected[task] = [row['seed'] for row in manifest['accepted'][:5]]
        for name, expected in json.loads((BASE/'scenes'/task/'freeze.json').read_text()).items():
            path = BASE/name
            assert sha(path) == expected, path
            checked[str(path)] = expected
    checkpoints = {}
    baseline = []
    for variant in VARIANTS:
        for seed in [42,43,44]:
            key = f'{variant}-seed{seed}'
            path = BASE/'runs'/key/'checkpoint_step_20000.safetensors'
            row = json.loads((path.parent/'result.json').read_text())
            assert row['complete'] and row['strict_reload'] and sha(path) == row['checkpoint_sha256']
            checkpoints[key] = dict(path=str(path), sha256=row['checkpoint_sha256'])
            for config in path.parent.iterdir():
                if config.is_file() and config.suffix in ('.json','.yaml','.npy'):
                    checked[str(config)] = sha(config)
            for task in TASKS:
                result = BASE/'evaluation/fresh'/key/task/'clean/summary.json'
                summary = json.loads(result.read_text())
                assert summary['complete']
                rows = [x for x in summary['records'] if x['scene_seed'] in selected[task]]
                assert len(rows) == 5 and all(x['status']=='complete' for x in rows)
                baseline.extend(rows)
                checked[str(result)] = sha(result)
    config_root = ROOT/'robot-configs'
    config_root.mkdir()
    for side in ('left','right'):
        path = ROOT.parent/'migration-20260927/robot-configs'/f'curobo_{side}.yml'
        shutil.copy2(path,config_root/path.name)
        checked[str(path)] = checked[str(config_root/path.name)] = sha(path)
    for condition in ['old','fixed']:
        dest = ROOT/condition
        dest.mkdir()
        for name in scripts+configs:
            shutil.copy2(BASE/name,dest/name)
        if condition == 'fixed':
            shutil.copy2(SOURCE.parent/'zero-task/migration_render.py',dest/'migration_render.py')
        (dest/'scenes').symlink_to(BASE/'scenes',target_is_directory=True)
        for name in scripts+configs:
            checked[str(dest/name)] = sha(dest/name)
        assert sha(dest/'evaluate_policy.py') == sha(BASE/'evaluate_policy.py')
        assert sha(dest/'serve_policy.py') == sha(BASE/'serve_policy.py')
    for path in (ROOT/'code').glob('*'):
        if path.is_file():checked[str(path)] = sha(path)
    for path in [EXPERT/'L1-gate.json',EXPERT/'branch-audit.json']:
        checked[str(path)] = sha(path)
    write(ROOT/'baseline-selected.json',baseline)
    checked[str(ROOT/'baseline-selected.json')] = sha(ROOT/'baseline-selected.json')
    protocol = json.loads((BASE/'protocol.json').read_text())
    write(ROOT/'registration.json',dict(name=ROOT.name,
        commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        model_commit=subprocess.check_output(['git','-C',str(model),'rev-parse','HEAD'],text=True).strip(),
        selected=selected,variants=VARIANTS,seeds=[42,43,44],checkpoints=checkpoints,files=checked,
        pause_paths=protocol['pause_paths']+[str(ROOT/'paused.json')],deadline=protocol['deadline'],
        lane_active_seconds=10800,environment_source=str(EXPERT/'environment.json')))
    print(json.dumps(dict(prepared=str(ROOT),hashes=len(checked),selected=selected)))

if __name__ == '__main__':main()
