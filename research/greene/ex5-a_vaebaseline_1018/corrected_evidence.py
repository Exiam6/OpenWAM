"""Retain failed preflight, corrected lineage and every charged validation attempt."""
import json
import math
from pathlib import Path
from build_cache import sha


def read(path):return json.loads(Path(path).read_text())


def release_path(root):
    root=Path(root);corrected=root/'corrected-full-training-release-v1.json'
    return corrected if corrected.exists() else root/'full-training-release.json'


def prior_cost(root,release):
    root=Path(root)
    if 'corrected_protocol_sha256' not in release:
        state=read(release.get('preflight_state_path',root/'preflight-run.json'))
        assert state['passed'] and state['status']=='complete' and 0<state['occupied_seconds']<=9000
        return state['occupied_seconds']
    assert release['original_preflight_status']=='failed'
    protocol=root/'corrected-protocol-v1/protocol.json'
    assert sha(protocol)==release['corrected_protocol_sha256']
    state_path=root/'dedicated-preflight-run.json';state=read(state_path)
    assert sha(state_path)==release['evidence_sha256'][str(state_path)]
    assert state['status']=='failed' and state['passed'] is False
    total=state['occupied_seconds'];assert math.isfinite(total) and total>0
    cache=next(x['execution']['occupied_seconds'] for x in state['stages'] if x['name']=='cache')
    for name in ['deployment-diagnostic-0807','encoder-precision-diagnostic-0812','fixed-policy-diagnostic-0825','corrected-tiny-validation-0827']:
        path=root/name/'execution.json';record=read(path)
        assert sha(path)==release['evidence_sha256'][str(path)]
        assert record['status']=='complete' and record['original_evidence_unchanged']
        execution=record['execution'];assert execution['complete'] and execution['exit_code']==0
        seconds=execution['occupied_seconds'];assert math.isfinite(seconds) and seconds>0;total+=seconds
    assert 0<cache<7200 and 0<total-cache<1800
    assert abs(total-release['prior_occupied_gpu_seconds'])<1e-6
    return total


def extra_evidence(root,release):
    if 'corrected_protocol_sha256' not in release:return set()
    root=Path(root);prior_cost(root,release)
    protocol=root/'corrected-protocol-v1/protocol.json';plan=read(protocol)
    paths={protocol,root/'corrected-protocol-v1-tests.json',root/'corrected-protocol-v1-tests.log',root/'corrected-training-capacity-wait-v1.json',root/'corrected-training-capacity-wait-v1-process.json',root/'corrected-training-capacity-wait-v1.log',root/'corrected-training-admission-v1.log',root/'corrected-training-release-v1.log',root/'deployment-contract/result.json'}
    for path,digest in plan['files'].items():
        path=Path(path);assert sha(path)==digest;paths.add(path)
    tests=read(root/'corrected-protocol-v1-tests.json');assert tests['passed'] and tests['tests_run']==24 and tests['protocol_sha256']==sha(protocol)
    assert sha(tests['log'])==tests['log_sha256']
    bundle_path=root/'corrected-validation-bundle-1018.json';bundle=read(bundle_path)
    assert sha(bundle['bundle'])==bundle['sha256'];paths.update([bundle_path,Path(bundle['bundle'])])
    wait=read(root/'corrected-training-capacity-wait-v1.json')
    assert wait['status']=='complete' and wait['protocol_sha256']==sha(protocol)
    cpu=root/'corrected-cpu-construction-audit-0857'
    construction=read(cpu/'result.json')
    assert construction['passed'] and construction['gpu_seconds']==0 and construction['seeds']==[42,43,44]
    assert len(construction['rows'])==3
    for row in construction['rows']:
        assert row['cpu_only'] and not row['weights_saved'] and not row['checkpoint_reused'] and not row['cuda_initialized']
        assert row['corrected_protocol_sha256']==sha(protocol)
        paths.add(cpu/f"seed{row['seed']}.json")
    paths.add(cpu/'result.json')
    assert all(path.is_file() for path in paths)
    return paths


def verify_initialization(root,seed):
    root=Path(root);path=root/'runs'/f'vae48_native-seed{seed}'/'initialization.json';record=read(path)
    assert record['seed']==seed
    assert record['trained_official_policy_weights_loaded'] is False and record['cuda_initialized'] is False
    assert record['matched_trainable_tensors']==676
    expected={'video_backbone.dit.patch_embedding.weight','video_backbone.dit.head.head.weight','video_backbone.dit.head.head.bias'}
    excluded=record['dimension_specific_exclusions']
    assert len(excluded)==3 and {x['name'] for x in excluded}==expected
    assert all(x['raw_shape']!=x['vae_shape'] for x in excluded)
    source=root.parent/'rae-factorial-20261010/initialization'/f'common-seed{seed}.safetensors'
    assert Path(record['common_initialization']).resolve()==source.resolve()
    registration=read(root/'gpu-registration.json')
    assert sha(source)==registration['asset_files'][str(source)]
    return path
