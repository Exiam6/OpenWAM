"""One-time CPU registration after committed amendment; never starts GPU work."""
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess

from build_cache import sha

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
BASE=ROOT.parent/'rae-factorial-20261010';G=ROOT.parent/'ex4-g_calibratedfactorial_1017';H=ROOT.parent/'ex4-h_officialcalibration_1017'
SOURCE=Path(__file__).resolve().parent


def main():
    destination=ROOT/'gpu-registration.json';code=ROOT/'code'
    assert not destination.exists() and not code.exists(),'Preserve prior registration/snapshot'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True),'Commit and review source first'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    assert (SOURCE/'GPU_ADMISSION.md').exists()
    g=json.loads((G/'registration.json').read_text());files=dict(g['files'])
    for name,digest in files.items():assert sha(name)==digest,name
    h=json.loads((H/'final-bundle.json').read_text());assert h['complete'] and h['verdict']=='predicted_calibration_transfer_supported'
    assert sha(h['path'])==h['sha256']
    assert json.loads((G/'L3-release.json').read_text())['passed']
    input_path=ROOT/'attempt-data-restored/input-audit.json';inputs=json.loads(input_path.read_text());assert inputs['passed']
    assert json.loads((ROOT/'model-cpu-audit.json').read_text())['passed']
    manifest_path=ROOT/'attempt-data-restored/window-manifest.json';assert sha(manifest_path)==inputs['window_manifest_sha256']
    manifest=json.loads(manifest_path.read_text());assert len(manifest['windows'])==6424 and len(manifest['sources'])==841
    assets={str(BASE/'cache.pt'):inputs['cache_sha256']}
    for row in manifest['sources']:assets[row['path']]=row['sha256']
    for row in inputs['schedules']:
        assets[row['initialization_path']]=row['initialization_sha256'];files[row['batches_path']]=row['batches_sha256']
    extraction=json.loads((ROOT/'vae-extraction.json').read_text());assert extraction['passed']
    assets[extraction['destination']]=extraction['sha256']
    protocol=json.loads((BASE/'protocol.json').read_text());assets[protocol['source_checkpoint']]=sha(protocol['source_checkpoint'])
    for name,digest in assets.items():assert sha(name)==digest,name
    audit=json.loads((ROOT/'evaluation-template-audit.json').read_text());assert audit['passed'] and not audit['scientific_evaluator_modified']
    for name,digest in audit['files'].items():assert sha(name)==digest,name
    files.update(audit['files'])
    template=json.loads((ROOT/'template-audit.json').read_text());assert template['passed']
    for name,digest in template['files'].items():
        path=ROOT/'templates/vae48_native'/name;assert sha(path)==digest;files[str(path)]=digest
    for directory in [ROOT/'robot-configs']:
        for path in directory.rglob('*'):
            if path.is_file():files[str(path)]=sha(path)
    for path in [BASE/'common.py',BASE/'protocol.json',input_path,manifest_path,ROOT/'base-environment.json',ROOT/'historical-outcomes.json',ROOT/'model-cpu-audit.json',ROOT/'vae-extraction.json']:
        files[str(path)]=sha(path)
    predecessor=[G/'registration.json',G/'L3-release.json',G/'frozen-reuse.json',G/'pilot-evidence-v2.zip',H/'final-bundle.json',H/'success-summary.json',Path(h['path'])]
    predecessor.extend(H/f'visual-review-{task}.json' for task in g['tasks'])
    predecessors={str(path):sha(path) for path in predecessor}
    monitor=ROOT.parent/'ex4-b_actiondiagnostic_1017/resource-latest-v7.json';m=json.loads(monitor.read_text())
    assert Path('/proc',str(m['pid'])).exists()
    assert b'resource_monitor_v7.py' in Path('/proc',str(m['pid']),'cmdline').read_bytes()
    now=dt.datetime.now(dt.timezone.utc);assert (now-dt.datetime.fromisoformat(m['time'])).total_seconds()<1900
    assert now<dt.datetime(2026,10,18,6,13,tzinfo=dt.timezone.utc),'Full allocation envelope no longer fits initial admission'
    code.mkdir()
    for path in sorted(SOURCE.iterdir()):
        if path.is_file() and path.suffix in ['.py','.md','.json']:shutil.copy2(path,code/path.name)
    for path in code.iterdir():files[str(path)]=sha(path)
    registration=dict(name='ex5-a_vaebaseline_1018',time=now.isoformat(),source_commit=commit,model_commit=g['model_commit'],
        gpu_launch_permitted=True,preflight_launch_permitted=True,released_stages=['cache','training_smoke','deployment_contract','tiny_fit','full_training','evaluation'],
        release_semantics='Stages are conditionally registered; actual preflight, full-training and per-seed video/cost releases are independently required.',
        cache_seconds_limit=7200,tiny_total_seconds=1800,training_seconds_per_seed=10800,evaluation_seconds_per_seed=10800,evaluation_pilot_seconds=3600,
        max_occupied_seconds=73800,deadline='2026-10-18T17:00:00+00:00',extra_window_start='2026-10-18T06:00:00+00:00',extra_window_end='2026-10-18T15:00:00+00:00',
        gpus=g['gpus'],pause_paths=g['pause_paths']+[str(ROOT/'paused.json')],seeds=[42,43,44],scenes=g['scenes'],tasks=g['tasks'],
        planned_evaluations=180,checkpoint_steps=20000,bootstrap_seed=921018,bootstrap_draws=10000,
        asset_files=assets,predecessor_files=predecessors,files=files,
        source_snapshot=str(code),monitor=dict(path=str(monitor),pid=m['pid']),gpu_execution_performed=False)
    # One shared small-file inventory is deliberately rechecked at every stage.
    for group in ['launcher_files','cache_files','training_files','deployment_files','tiny_files','batch_files','evaluation_files']:
        registration[group]=files
    with destination.open('x') as f:json.dump(registration,f,indent=2);f.write('\n')
    result=dict(time=now.isoformat(),passed=True,registration=str(destination),registration_sha256=sha(destination),source_commit=commit,
        inherited_files=len(g['files']),frozen_small_files=len(files),asset_files=len(assets),asset_bytes=sum(Path(x).stat().st_size for x in assets),
        code_files=len(list(code.iterdir())),gpu_execution_performed=False,scope='Registration and current checksums only; no GPU launch or VAE numerical validation.')
    with (ROOT/'gpu-registration-audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
