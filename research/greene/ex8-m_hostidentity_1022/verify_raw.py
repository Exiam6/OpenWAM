"""Read-only offline audit. Never imports native GPU helpers or dispatches work."""
import hashlib
import json
import math
from pathlib import Path
import sys

F=Path('/home/zifanz4/openwam-runtime/ex8-m_hostidentity_1022')
N=F.parent/'nightly-gpu'
J='ex8m-native-host-identity'
sys.path.insert(0,str(F/'code'))
from integration import OPERATIONS, DEPENDENCIES
sys.path.insert(0,str(DEPENDENCIES/'native_sample_contract'))
from contract import validate
from inventory import unit_path


def check(ok, message):
    if not ok: raise ValueError(message)


def verify_samples(scientific, config, journal):
    """Reconstruct raw inventories; phase labels remain supervisor evidence.

    Reconstructed timestamps validate retained ordering/freshness, not current
    wall-clock freshness or an independent native phase timestamp witness.
    """
    journal=Path(journal).resolve()
    phases=['before_prewarm']+['after_'+n for n in OPERATIONS]
    counts={name:0 for name in phases}
    samples=scientific['samples']
    check(type(samples) is list and 39<=len(samples)<=128,'Invalid sample count')
    seen=set();previous=-math.inf;last_stable=-1;witness=False
    criteria=dict(host=config['host'],device_map={int(k):v for k,v in config['device_map'].items()},
                  assigned=config['assigned'],own_group=config['own_group'],gdm_uid=config['gdm_uid'])
    for index,row in enumerate(samples):
        receipt=row['evidence_receipt'];target=Path(receipt['path'])
        check(not target.is_symlink(),'Symlink receipt')
        check(target.resolve()==journal/('%04d.json'%index),'Duplicate, unordered or foreign receipt')
        check(str(target.resolve()) not in seen,'Repeated receipt')
        seen.add(str(target.resolve()))
        raw=target.read_bytes()
        check(len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256'],'Receipt digest/size mismatch')
        record=json.loads(raw);check(record['kind']=='validated_sample','Failure journal record')
        packet=record['packet'];outer=packet['evidence'];stamp=row['monotonic']
        check(type(stamp) in (int,float) and math.isfinite(stamp) and stamp>=previous,'Invalid sample order')
        check(outer['monotonic_start']>=previous,'Overlapping or replayed raw sample')
        rebuilt=validate(packet,**criteria,phase_started=outer['monotonic_start'],now=stamp,max_age=4)
        previous=stamp
        for field,value in rebuilt.items():check(row.get(field)==value,'Raw/summary mismatch: '+field)
        check(row.get('stable_phase') is (row['phase']==row['finished_phase']),'Incorrect phase stability')
        if row['stable_phase']:
            check(row['phase'] in phases,'Unexpected stable phase')
            ordinal=phases.index(row['phase']);check(ordinal>=last_stable,'Replayed phase order')
            last_stable=ordinal;counts[row['phase']]+=1
        witness=witness or bool(rebuilt['contexts'])
    check(set(str(x.resolve()) for x in journal.glob('*.json'))==seen,'Unreferenced journal evidence')
    check(all(n>=3 for n in counts.values()),'Missing distinct stable samples')
    check(witness,'No assigned context observed')
    return dict(distinct_raw_packets=len(seen),phase_counts=counts,raw_inventories_reconstructed=True,
                phase_labels_independently_authenticated=False,continuous_isolation_proven=False)


def audit():
    out=dict(GPU_diagnostic_validated=False,native_L2_L3_passed=False,full_release_allowed=False,
             full_paper_reproduced=False,audit_kind='INDEPENDENT_READ_ONLY_RAW_RECONSTRUCTION')
    required=[F/'result.json',F/'observer-config.json',N/'claims'/(J+'.json'),N/'runs'/J/'result.json']
    if not all(x.exists() for x in required):
        return dict(out,assessment='pending evidence',missing=[str(x) for x in required if not x.exists()])
    try:
        s,cfg,claim,worker=[json.loads(x.read_text()) for x in required]
        protocol=json.loads((F/'protocol.json').read_text())
        for name,digest in protocol['input_sha256'].items():
            check(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'Frozen input changed: '+name)
        from audit_outcome import audit as baseline_audit
        baseline=baseline_audit();check(baseline['GPU_diagnostic_validated'],'Conservative native terminal audit failed: '+baseline['assessment'])
        job=json.loads((N/'jobs'/(J+'.json')).read_text())
        check(job['id']==claim['id']==worker['id']==J,'Wrong fixed job identity')
        check(job['scientific_identity']==claim['scientific_identity']==worker['scientific_identity'],'Scientific identity mismatch')
        check(cfg['assigned']==claim['gpu_uuid']==worker['gpu_uuid'],'Observer allocation mismatch')
        check(claim['host']==worker['host']=='gpu01' and cfg['host']==protocol['hostnames']['gpu01'],'Observer host mismatch')
        check(cfg['device_map']==protocol['device_map'],'Observer device map mismatch')
        check(Path(unit_path(cfg['own_group'])).name==claim['unit']+'.service','Wrong dispatcher cgroup')
        proof=verify_samples(s,cfg,F/'journal')
        return dict(out,**proof,GPU_diagnostic_validated=True,assessment='Native prefix only; full scene L2/L3 still required')
    except Exception as exc:
        return dict(out,assessment='failed closed',error=repr(exc))

if __name__=='__main__':print(json.dumps(audit(),indent=2))
