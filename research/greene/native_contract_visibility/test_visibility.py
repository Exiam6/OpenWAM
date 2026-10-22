import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'native_sample_contract'))
import test_contract as c
packet, f = c.packet, c.fixtures
from contract import ContractError

class VisibilityTests(c.ContractTests):
    def denied(self,pid=3,group=f.OTHER,uid=120):
        value=f.proc(pid,group,uid,None)
        value['executable_permission_denied']=True
        return value
    def display(self,**kw):
        return packet(f.display_fixture(),lambda pid:self.denied(pid,**kw))
    def test_denied_display_composed(self):
        p=self.display()
        self.assertEqual(len(self.check(p)['system_displays']),1)
        self.assertIsNone(p['value']['evidence']['identities']['3']['executable'])
    def test_denied_foreign_peer_retained(self):
        p=packet(f.fixture([(1,2)]),lambda pid:self.denied(pid,uid=1000))
        self.assertEqual(len(self.check(p)['foreign_contexts']),1)
    def test_denied_own_assigned_retained(self):
        p=packet(f.fixture([(0,2)]),lambda pid:self.denied(pid,group=f.OWN,uid=1000))
        self.assertEqual(len(self.check(p,require_assigned=True)['contexts']),1)
    def test_denied_own_peer_display_rejected(self):
        p=self.display(group=f.OWN)
        self.assertEqual(len(self.check(p)['contexts']),1)
        root=f.fixture([(1,3)])
        row=root.findall('./gpu')[1].find('./processes/process_info')
        row.find('type').text='G';row.find('process_name').text=f.XORG
        p=packet(root,lambda pid:self.denied(pid,group=f.OWN,uid=1000))
        self.assertEqual(p['value']['contexts'][0]['uuid'],f.MAP[1])
        self.reject(p)
    def test_denied_exe_requires_explicit_boolean(self):
        for flag in (False,None,1,'true',[]):
            p=self.display();p['value']['evidence']['identities']['3']['executable_permission_denied']=flag
            self.reject(p)
        p=self.display();del p['value']['evidence']['identities']['3']['executable_permission_denied'];self.reject(p)
    def test_readable_exe_rejects_denied_flag(self):
        for flag in (True,None,1,'false'):
            p=packet(f.fixture([(0,2)]))
            p['value']['evidence']['identities']['2']['executable_permission_denied']=flag
            self.reject(p)
    def test_denied_display_criteria_still_enforced(self):
        for field,value in [('uids',[1000]*4),('executable','/tmp/Xorg')]:
            p=self.display();p['value']['evidence']['identities']['3'][field]=value;self.reject(p)
        for field,value in [('type','C'),('process_name','/tmp/Xorg'),('used_memory','17 MiB')]:
            p=self.display();raw=p['value']['evidence']['queries'][0]['stdout']
            import xml.etree.ElementTree as ET
            xml=ET.fromstring(raw);xml.find('./gpu/processes/process_info/'+field).text=value
            p['value']['evidence']['queries'][0]['stdout']=ET.tostring(xml,encoding='unicode');self.reject(p)

if __name__=='__main__':unittest.main()
