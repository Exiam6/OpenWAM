"""CPU-only integration seam; no GPU entrypoint or native imports."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native_sample_supervision'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native_sample_contract'))
from supervise import sample, SampleError
from contract import validate, ContractError


class EvidenceError(RuntimeError):
    pass


class Journal:
    """Exclusive durable bounded records. A failed record poisons this journal.

    Directory must be a newly allocated trusted local run directory. Partial
    records remain evidence; no overwrite, recovery retry or deletion is done.
    Independent dispatcher timeout still bounds filesystem/kernel stalls.
    """
    def __init__(self, directory, *, max_records=128, max_bytes=64*1024*1024,
                 record_bytes=4*1024*1024):
        for n, ceiling in [(max_records,128),(max_bytes,64*1024*1024),
                           (record_bytes,4*1024*1024)]:
            if type(n) is not int or not 0 < n <= ceiling:
                raise ValueError('Invalid journal bound')
        self.directory=Path(directory)
        self.directory.mkdir(exist_ok=False)
        parent_fd=os.open(self.directory.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(parent_fd)
        finally:os.close(parent_fd)
        self.max_records,self.max_bytes,self.record_bytes=max_records,max_bytes,record_bytes
        self.records=0
        self.bytes=0
        self.failed=False
        self.lock=threading.Lock()

    def append(self, record):
        with self.lock:
            if self.failed:raise EvidenceError('Journal already failed; no retry')
            try:
                raw=(json.dumps(record, allow_nan=False, sort_keys=True)+'\n').encode('utf-8')
                if (self.records >= self.max_records or len(raw)>self.record_bytes
                        or self.bytes+len(raw)>self.max_bytes):
                    raise EvidenceError('Journal bound exceeded')
                target=self.directory/('%04d.json'%self.records)
                with target.open('xb') as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                fd=os.open(self.directory,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
                self.records+=1
                self.bytes+=len(raw)
                return dict(path=str(target),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            except Exception as exc:
                self.failed=True
                raise EvidenceError('Evidence persistence failed; stop observation chain') from exc


class DurableObserver:
    """Connect trusted sample worker, schema contract and durable packet journal.

    Per-call start is trusted locally; existing supervisor separately brackets
    stable phases. No hostile-worker authentication or continuous isolation.
    A failed observation permanently closes this callable, without blind retry.
    """
    def __init__(self, argv, journal, *, host, device_map, assigned, own_group,
                 gdm_uid, seconds=4.0, max_bytes=1024*1024, env=None):
        self.argv=tuple(argv)
        self.journal=journal
        self.criteria=copy.deepcopy(dict(host=host,device_map=device_map,
                 assigned=assigned,own_group=own_group,gdm_uid=gdm_uid))
        self.seconds,self.max_bytes=seconds,max_bytes
        self.env=copy.deepcopy(env)
        self.failed=False
        self.lock=threading.Lock()

    def __call__(self):
        with self.lock:
            if self.failed:raise EvidenceError('Observer already failed; no retry')
            began=time.monotonic()
            try:
                packet=sample(self.argv,seconds=self.seconds,max_bytes=self.max_bytes,env=self.env)
            except SampleError as exc:
                self.failed=True
                self.journal.append(dict(kind='worker_failure',error=repr(exc),packet=exc.evidence))
                raise
            try:
                result=validate(packet,**self.criteria,phase_started=began,
                                now=time.monotonic(),max_age=self.seconds)
            except ContractError as exc:
                self.failed=True
                self.journal.append(dict(kind='contract_failure',error=repr(exc),packet=exc.evidence))
                raise
            try:
                receipt=self.journal.append(dict(kind='validated_sample',packet=packet))
            except Exception:
                self.failed=True
                raise
            result['evidence_receipt']=receipt
            return result
