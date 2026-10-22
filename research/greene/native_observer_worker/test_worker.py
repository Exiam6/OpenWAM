import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import worker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native_evidence_adapter"))
sys.path.insert(0, str(ROOT / "native_observer_prep"))
from adapter import DurableObserver, Journal, SampleError, ContractError
import test_observer as f


def config():
    return dict(host="host", device_map={str(k): v for k, v in f.MAP.items()},
                assigned=f.MAP[0], own_group=f.OWN, gdm_uid=120, seconds=2,
                pause_paths=[worker.GLOBAL_PAUSE])


@contextlib.contextmanager
def trusted():
    with patch("worker.socket.gethostname", return_value="host"), \
         patch("worker.Path.read_text", return_value=f.OWN), \
         patch("worker.pwd.getpwnam", return_value=SimpleNamespace(pw_uid=120)):
        yield


class WorkerTests(unittest.TestCase):
    def load(self, raw=None, changes=None, digest=None):
        with tempfile.TemporaryDirectory() as directory:
            value=config();value.update(changes or {})
            raw=json.dumps(value).encode() if raw is None else raw
            p=Path(directory)/"assignment.json";p.write_bytes(raw)
            return worker.load_config(p, digest or hashlib.sha256(raw).hexdigest())

    def test_frozen_schema_and_converted_map(self):
        self.assertEqual(self.load()["device_map"], f.MAP)

    def test_changed_config_and_oversize_rejected(self):
        for kw in (dict(digest="0"*64), dict(raw=b"x"*65537)):
            with self.subTest(kw=list(kw)), self.assertRaises(worker.WorkerError):self.load(**kw)

    def test_duplicate_key_rejected(self):
        with self.assertRaises(worker.WorkerError):self.load(raw=b'{"host":"a","host":"b"}')

    def test_malformed_schema_rejected(self):
        for changes in (dict(extra=1), dict(device_map={}), dict(assigned="wrong"),
                        dict(gdm_uid=True), dict(seconds=True), dict(seconds=float("nan")),
                        dict(seconds=5), dict(own_group="0::/wrong"),
                        dict(pause_paths=[]), dict(pause_paths=["relative"]),
                        dict(pause_paths=[worker.GLOBAL_PAUSE,"/tmp/../pause"]),
                        dict(pause_paths=[worker.GLOBAL_PAUSE]*2)):
            with self.subTest(changes=changes),self.assertRaises((worker.WorkerError,RuntimeError)):
                self.load(changes=changes)

    def test_broken_symlink_pause_is_active(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"pause";p.symlink_to(Path(d)/"missing")
            with self.assertRaises(worker.WorkerError):worker.guard([str(p)])

    def test_global_pause_before_config_read(self):
        with patch("worker.os.path.lexists",return_value=True),self.assertRaises(worker.WorkerError):
            worker.load_config("/absent", "0"*64)

    def test_success_uses_existing_observer(self):
        c=self.load();result={"complete":True}
        with trusted(),patch("worker.observe",return_value=result) as observe:
            self.assertIs(worker.run(c),result)
        observe.assert_called_once_with("host",f.MAP,f.MAP[0],f.OWN,120,seconds=2)

    def test_host_unit_and_uid_rejected_before_query(self):
        c=self.load()
        for target,value in (("worker.socket.gethostname","wrong"),
                             ("worker.Path.read_text",f.OTHER),
                             ("worker.pwd.getpwnam",SimpleNamespace(pw_uid=999))):
            with self.subTest(target=target),trusted(),patch(target,return_value=value), \
                 patch("worker.observe") as observe,self.assertRaises(RuntimeError):worker.run(c)
            observe.assert_not_called()

    def test_pause_before_query(self):
        c=self.load()
        with trusted(),patch("worker.os.path.lexists",return_value=True), \
             patch("worker.observe") as observe,self.assertRaises(worker.WorkerError):worker.run(c)
        observe.assert_not_called()

    def test_pause_after_query_retains_rejected_result(self):
        c=self.load();result={"complete":True,"raw":"preserve"}
        with trusted(),patch("worker.os.path.lexists",side_effect=[False,False,True]), \
             patch("worker.observe",return_value=result),self.assertRaises(worker.WorkerError) as caught:
            worker.run(c)
        self.assertEqual(caught.exception.evidence,result)

    def test_changed_unit_after_query_retains_result(self):
        c=self.load();result={"complete":True}
        with trusted(),patch("worker.Path.read_text",side_effect=[f.OWN,f.OTHER]), \
             patch("worker.observe",return_value=result),self.assertRaises(worker.WorkerError) as caught:
            worker.run(c)
        self.assertEqual(caught.exception.evidence,result)

    def test_main_failure_retains_raw_observer_evidence(self):
        error=worker.ObservationError("denied",{"queries":[{"stdout":"raw","stderr":"denied"}]})
        with patch("worker.load_config",return_value={}),patch("worker.run",side_effect=error), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            status=worker.main(["--config","unused","--sha256","0"*64])
        self.assertEqual(status,1)
        self.assertEqual(json.loads(output.getvalue())["observer_evidence"],error.evidence)

    def test_unmodified_cli_rejects_bad_hash_before_queries(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"config.json";path.write_text(json.dumps(config()))
            result=subprocess.run([sys.executable,worker.__file__,"--config",str(path),
                       "--sha256","0"*64],capture_output=True,text=True,timeout=3)
            self.assertEqual(result.returncode,1)
            value=json.loads(result.stdout)
            self.assertFalse(value["complete"])
            self.assertIn("hash mismatch",value["worker_error"])

    def test_main_post_query_rejection_keeps_raw(self):
        error=worker.WorkerError("paused after query",evidence={"complete":True,"raw":"preserved"})
        with patch("worker.load_config",return_value={}),patch("worker.run",side_effect=error), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            status=worker.main(["--config","unused","--sha256","0"*64])
        self.assertEqual(status,1)
        self.assertEqual(json.loads(output.getvalue())["rejected_observation"],error.evidence)

    def child(self,d,index=0,fail=False):
        path=Path(d)/"config.json";path.write_text(json.dumps(config()))
        code=("import sys;sys.path.insert(0,"+repr(str(Path(worker.__file__).parent))+");"
              "import worker;from unittest.mock import patch;from types import SimpleNamespace;"
              "sys.path.insert(0,"+repr(str(ROOT/'native_observer_prep'))+");import test_observer as f;"
              "worker.socket.gethostname=lambda:'host';"
              "worker.Path.read_text=lambda self:f.OWN;"
              "worker.pwd.getpwnam=lambda name:SimpleNamespace(pw_uid=120);"
              "worker.observe=lambda *a,**kw:f.ObserverTests().run_observer(root=f.fixture([("+str(index)+",2)]));")
        if fail:
            code+=("worker.observe=lambda *a,**kw:(_ for _ in ()).throw("
                   "worker.ObservationError('query failure',{'queries':[{'stdout':'preserved raw'}]}));")
        code+="sys.exit(worker.main("+repr(["--config",str(path),"--sha256",hashlib.sha256(path.read_bytes()).hexdigest()])+"))"
        return [sys.executable,"-c",code]

    def test_real_child_worker_composes_with_durable_contract(self):
        with tempfile.TemporaryDirectory() as d:
            observe=DurableObserver(self.child(d),Journal(Path(d)/"journal"),host="host",
                   device_map=f.MAP,assigned=f.MAP[0],own_group=f.OWN,gdm_uid=120,seconds=3)
            result=observe();self.assertTrue(result["complete"])
            record=json.loads((Path(d)/"journal/0000.json").read_text())
            self.assertEqual(record["kind"],"validated_sample")
            self.assertIn("nvidia_smi_log",record["packet"]["value"]["evidence"]["queries"][0]["stdout"])

    def test_real_child_peer_rejected_and_retained(self):
        with tempfile.TemporaryDirectory() as d:
            observe=DurableObserver(self.child(d,1),Journal(Path(d)/"journal"),host="host",
                   device_map=f.MAP,assigned=f.MAP[0],own_group=f.OWN,gdm_uid=120,seconds=3)
            with self.assertRaises(ContractError):observe()
            record=json.loads((Path(d)/"journal/0000.json").read_text())
            self.assertEqual(record["kind"],"contract_failure")
            self.assertEqual(record["packet"]["value"]["contexts"][0]["uuid"],f.MAP[1])

    def test_real_child_failure_raw_evidence_survives(self):
        with tempfile.TemporaryDirectory() as d:
            observe=DurableObserver(self.child(d,fail=True),Journal(Path(d)/"journal"),host="host",
                   device_map=f.MAP,assigned=f.MAP[0],own_group=f.OWN,gdm_uid=120,seconds=3)
            with self.assertRaises(SampleError):observe()
            record=json.loads((Path(d)/"journal/0000.json").read_text())
            self.assertEqual(record["kind"],"worker_failure")
            raw=json.loads(record["packet"]["stdout"])
            self.assertEqual(raw["observer_evidence"]["queries"][0]["stdout"],"preserved raw")

if __name__=="__main__":unittest.main()
