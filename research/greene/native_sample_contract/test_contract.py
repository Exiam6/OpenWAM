import copy
import json
from pathlib import Path
import sys
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native_observer_prep'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native_sample_supervision'))
import test_observer as fixtures
from supervise import sample
from contract import validate, ContractError

MAP, OWN, OTHER = fixtures.MAP, fixtures.OWN, fixtures.OTHER


def packet(root=None, reader=None):
    began=time.monotonic()
    value=fixtures.ObserverTests().run_observer(root=root,reader=reader)
    return json.loads(json.dumps(dict(value=value, evidence=dict(
        monotonic_start=began,monotonic_end=time.monotonic(),returncode=0,cleanup_reaped=True))))


class ContractTests(unittest.TestCase):
    def check(self,p,**kw):
        args=dict(host='host',device_map=MAP,assigned=MAP[0],own_group=OWN,
                  gdm_uid=120,phase_started=0,now=time.monotonic())
        args.update(kw)
        return validate(p,**args)
    def reject(self,p,**kw):
        with self.assertRaises(ContractError) as cm:self.check(p,**kw)
        self.assertIs(cm.exception.evidence,p)
        json.dumps(cm.exception.evidence)
    def test_empty_before_prewarm(self):
        self.assertEqual(self.check(packet())['contexts'],[])
    def test_assigned_roundtrip(self):
        self.assertEqual(len(self.check(packet(fixtures.fixture([(0,2)])),require_assigned=True)['contexts']),1)
    def test_assigned_required(self):self.reject(packet(),require_assigned=True)
    def test_peer_retained_and_rejected(self):
        p=packet(fixtures.fixture([(1,2)]))
        self.assertEqual(p['value']['contexts'][0]['uuid'],MAP[1])
        self.reject(p)
    def test_raw_summary_disagreement_cannot_hide_peer(self):
        p=packet(fixtures.fixture([(1,2)]));p['value']['contexts']=[]
        self.reject(p)
    def test_summary_cannot_hide_foreign_assigned(self):
        p=packet(fixtures.fixture([(0,2)]))
        p['value']['evidence']['identities']['2']['cgroup']=OTHER
        self.reject(p)
    def test_wrong_host_map_unit(self):
        for field,value in [('device_map',{}),('own_unit','/other')]:
            p=packet();p['value'][field]=value;self.reject(p)
        p=packet();p['value']['evidence']['host']='other';self.reject(p)
    def test_complete_and_worker_status(self):
        for value in (False,1,'true',None):
            p=packet();p['value']['complete']=value;self.reject(p)
        for field,value in [('cleanup_reaped',False),('returncode',True),('returncode',1),('error','failure'),('cleanup_error','unreaped')]:
            p=packet();p['evidence'][field]=value;self.reject(p)
    def test_missing_schema(self):
        for key in ('value','evidence'):
            p=packet();del p[key];self.reject(p)
        for key in ('queries','identities','monotonic_start','monotonic_end'):
            p=packet();del p['value']['evidence'][key];self.reject(p)
    def test_queries_raw_and_order(self):
        for field,value in [('argv',['wrong']),('returncode',False),('returncode',7),('stdout',None),('timed_out',True)]:
            p=packet();p['value']['evidence']['queries'][0][field]=value;self.reject(p)
        p=packet();p['value']['evidence']['queries'].reverse();self.reject(p)
    def test_invalid_times(self):
        for value in (True,float('nan'),float('inf'),'1',None):
            p=packet();p['evidence']['monotonic_end']=value;self.reject(p)
    def test_stale_and_prephase(self):
        p=packet();self.reject(p,now=time.monotonic()+10)
        self.reject(p,phase_started=time.monotonic()+1)
        p['value']['evidence']['queries'][0]['finished']=time.monotonic()+10;self.reject(p)
    def test_invalid_budgets(self):
        for value in (True,0,-1,11,float('nan')):self.reject(packet(),max_age=value)
    def test_identity_schema_and_unused(self):
        for field,value in [('pid',True),('pid',3),('start_ticks',0),('start_ticks',True),('uids',[0,0,0]),('uids',[True]*4),('executable','relative'),('cgroup',None)]:
            p=packet(fixtures.fixture([(0,2)]));p['value']['evidence']['identities']['2'][field]=value;self.reject(p)
        p=packet();p['value']['evidence']['identities']['2']=fixtures.proc();self.reject(p)
    def test_verified_system_display_and_foreign_peer(self):
        p=packet(fixtures.display_fixture(),lambda pid:fixtures.proc(pid,OTHER,120,fixtures.XORG))
        self.assertEqual(len(self.check(p)['system_displays']),1)
        p['value']['evidence']['identities']['3']['executable']='/tmp/Xorg';self.reject(p)
        p=packet(fixtures.fixture([(1,2)]),lambda pid:fixtures.proc(pid,OTHER))
        self.assertEqual(len(self.check(p)['foreign_contexts']),1)
    def test_real_worker_peer_blocks_prepared_operations(self):
        root=Path(__file__).resolve().parents[1]
        code=('import sys,json; sys.path.insert(0,'+repr(str(root/'native_observer_prep'))+'); '
              'import test_observer as f; print(json.dumps(f.ObserverTests().run_observer(root=f.fixture([(1,2)]))))')
        calls=[]
        failures=[]
        operations={name:lambda name=name:calls.append(name) or True for name in fixtures.OPERATIONS}
        def observe():
            began=time.monotonic()
            p=sample([sys.executable,'-c',code],seconds=2)
            try:return self.check(p,phase_started=began)
            except ContractError as exc:
                failures.append(exc.evidence)
                raise
        result=fixtures.Supervisor().run(operations,observe,lambda:None,MAP[0],
                                        seconds=2,sample_timeout=1,interval=.001)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(calls,[])
        self.assertIn('Own peer GPU context',str(result))
        self.assertEqual(failures[0]['value']['contexts'][0]['uuid'],MAP[1])
        self.assertIn('nvidia_smi_log',failures[0]['value']['evidence']['queries'][0]['stdout'])

    def test_positive_composed_sequence_is_only_cpu_preparation(self):
        calls=[]
        operations={name:lambda name=name:calls.append(name) or True for name in fixtures.OPERATIONS}
        def observe():return self.check(packet(fixtures.fixture([(0,2)])),require_assigned=True)
        result=fixtures.Supervisor().run(operations,observe,lambda:None,MAP[0],
                                        seconds=2,sample_timeout=.5,interval=.001)
        self.assertEqual(result['status'],'supervisor_sequence_passed')
        self.assertEqual(calls,list(fixtures.OPERATIONS))
        self.assertFalse(result['full_release_allowed'])
        self.assertFalse(result['native_L2_L3_passed'])
        self.assertFalse(result['full_paper_reproduced'])

    def test_real_worker_observer_supervisor_contract(self):
        root=Path(__file__).resolve().parents[1]
        code=('import sys,json; sys.path.insert(0,'+repr(str(root/'native_observer_prep'))+'); '
              'import test_observer as f; print(json.dumps(f.ObserverTests().run_observer(root=f.fixture([(0,2)]))))')
        began=time.monotonic()
        p=sample([sys.executable,'-c',code],seconds=2)
        r=self.check(p,phase_started=began,require_assigned=True)
        self.assertEqual(r['contexts'][0]['uuid'],MAP[0])

if __name__=='__main__':unittest.main()
