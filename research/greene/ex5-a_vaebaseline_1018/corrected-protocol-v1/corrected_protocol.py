"""Explicit corrected-protocol lineage; retain the original failed registration."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/zifanz4/openwam-runtime/ex5-a_vaebaseline_1018')
SOURCE=Path(__file__).parent
ENGINE=ROOT/'corrected-engine-v1'
ORIGINAL_ENGINE=Path('/home/zifanz4/openwam-rae-research')
RELEASE=ROOT/'corrected-full-training-release-v1.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def load_registration():
    plan=read(SOURCE/'protocol.json')
    assert plan['status']=='registered_corrected_protocol'
    assert sha(ROOT/'gpu-registration.json')==plan['original_registration_sha256']
    for path,digest in plan['files'].items():assert sha(path)==digest,path
    p=read(ROOT/'gpu-registration.json')
    assert p['seeds']==[42,43,44] and p['checkpoint_steps']==20000 and p['planned_evaluations']==180
    assert p['max_occupied_seconds']==73800 and p['training_seconds_per_seed']==p['evaluation_seconds_per_seed']==10800
    assert p['deadline']=='2026-10-18T17:00:00+00:00'
    for pause in p['pause_paths']:assert not Path(pause).exists()
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(p['deadline'])
    manifest=read(ROOT/'corrected-engine-v1-manifest.json')
    assert manifest['source_commit']==p['model_commit']=='e983676b069150d0346fcd9e90fd8ebe8a840f13'
    assert manifest['registered_source_files_verified']==len(manifest['rows'])==234
    assert manifest['production_delta']==['openwam/model/video_backbone/wan/models/vae.py']
    for row in manifest['rows']:
        assert sha(ORIGINAL_ENGINE/row['relative_path'])==row['original_sha256']
        assert sha(ENGINE/row['relative_path'])==row['candidate_sha256']
        assert (row['original_sha256']!=row['candidate_sha256'])==(row['relative_path'] in manifest['production_delta'])
    # Preserve all original checks, and additionally freeze the corrected equivalents.
    for key,values in list(p.items()):
        if key!='files' and not key.endswith('_files'):continue
        if not isinstance(values,dict):continue
        augmented=dict(values)
        for path in values:
            path=Path(path)
            if path.is_relative_to(ROOT/'code'):
                candidate=SOURCE/path.relative_to(ROOT/'code')
            elif path.is_relative_to(ORIGINAL_ENGINE):
                candidate=ENGINE/path.relative_to(ORIGINAL_ENGINE)
            else:continue
            if candidate.exists():augmented[str(candidate)]=plan['files'][str(candidate)]
        augmented[str(SOURCE/'corrected_protocol.py')]=plan['files'][str(SOURCE/'corrected_protocol.py')]
        p[key]=augmented
    return p


def assert_runtime():
    from openwam.model.video_backbone.wan.models import vae
    assert Path(vae.__file__).resolve()==ENGINE/'openwam/model/video_backbone/wan/models/vae.py'
    row=next(x for x in read(ROOT/'corrected-engine-v1-manifest.json')['rows'] if x['changed'])
    assert sha(vae.__file__)==row['candidate_sha256']


def verify_training_release():
    gate=read(RELEASE)
    assert gate['passed'] and gate['native_deployment_contract_passed'] and gate['tiny_fit_passed'] and gate['budget_projection_passed']
    assert gate['corrected_protocol_sha256']==sha(SOURCE/'protocol.json')
    assert gate['runtime_manifest_sha256']==sha(ROOT/'corrected-engine-v1-manifest.json')
    assert gate['planned_training_seeds']==[42,43,44] and gate['fixed_training_steps']==20000
    assert gate['fresh_initialization_required'] and not gate['tiny_checkpoint_reused']
    assert gate['original_preflight_status']=='failed'
    for path,digest in gate['evidence_sha256'].items():assert sha(path)==digest,path
    assert gate['prior_occupied_gpu_seconds']+6*10800<=73800
    return gate


def verify_training_lineage(trained):
    verify_training_release()
    assert trained['complete'] and trained['steps']==20000
    assert trained['corrected_protocol_sha256']==sha(SOURCE/'protocol.json')
    assert trained['runtime_manifest_sha256']==sha(ROOT/'corrected-engine-v1-manifest.json')


def idle_capacity(p,gpus):
    occupied={line.split(',')[0].strip() for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines() if line.strip()}
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.free,memory.used,memory.reserved,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    samples={}
    for line in rows.splitlines():
        index,uuid,pci,free,used,reserved,util=[x.strip() for x in line.split(',')];gpu=int(index)
        if gpu not in gpus:continue
        assert uuid==p['gpus'][str(gpu)]
        free,used,reserved,util=map(int,[free,used,reserved,util]);assert min(free,used,reserved,util)>=0
        samples[gpu]=dict(gpu=gpu,uuid=uuid,pci=pci,free_mib=free,used_mib=used,driver_reserved_mib=reserved,util=util,idle=uuid not in occupied and used<256 and util==0)
    assert set(samples)==set(gpus)
    return samples
