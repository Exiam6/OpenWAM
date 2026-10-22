"""CPU-only validation of a trusted frozen worker's serialized inventory."""
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native_observer_prep'))
from observer import SMI, system_display
from inventory import snapshot, unit_path


class ContractError(RuntimeError):
    def __init__(self, message, packet):
        super().__init__(message)
        self.evidence = packet


def validate(packet, *, host, device_map, assigned, own_group, gdm_uid,
             phase_started, now, max_age=5.0, require_assigned=False):
    """Return reconstructed safe sample or retain bounded packet in failure.

    Caller supplies trusted same-host monotonic times, frozen identity/map and
    phase start. Only call on a packet returned by supervise.sample; this is
    neither worker authentication nor an atomic/continuous isolation proof.
    """
    def check(ok, message):
        if not ok:
            raise ValueError(message)
    def number(v):
        return type(v) in (int, float) and math.isfinite(v)
    try:
        check(number(max_age) and 0 < max_age <= 10, 'Invalid freshness budget')
        check(number(phase_started) and number(now), 'Invalid caller time')
        check(type(require_assigned) is bool, 'Invalid assigned criterion')
        outer = packet['evidence']
        check(outer.get('cleanup_reaped') is True and type(outer.get('returncode')) is int
              and outer['returncode'] == 0 and 'error' not in outer
              and 'cleanup_error' not in outer, 'Unsuccessful supervised worker')
        value = packet['value']
        check(value.get('complete') is True, 'Incomplete sample')
        e = value['evidence']
        check(e['host'] == host and 'error' not in e, 'Wrong host or observer error')
        queries = e['queries']
        check(type(queries) is list and len(queries) == 2, 'Incomplete queries')
        commands = [[SMI, '-q', '-x'],
                    [SMI, '--query-gpu=index,uuid', '--format=csv,noheader']]
        times = [phase_started, outer['monotonic_start'], e['monotonic_start']]
        for q, command in zip(queries, commands):
            check(q['argv'] == command and type(q['returncode']) is int
                  and q['returncode'] == 0 and not q.get('timed_out'), 'Wrong or failed query')
            check(type(q['stdout']) is str and type(q['stderr']) is str, 'Missing raw query')
            times.extend([q['started'], q['finished']])
        times.extend([e['monotonic_end'], outer['monotonic_end'], now])
        check(all(number(t) for t in times), 'Nonfinite or invalid timestamp')
        check(all(a <= b for a, b in zip(times, times[1:])), 'Stale, pre-phase or unordered sample')
        check(now - outer['monotonic_start'] <= max_age, 'Sample freshness exceeded')
        identities = e['identities']
        check(type(identities) is dict, 'Missing identities')
        for key, identity in identities.items():
            pid = identity['pid']
            check(type(pid) is int and pid > 0 and key == str(pid), 'Invalid identity PID')
            check(type(identity['start_ticks']) is int and identity['start_ticks'] > 0,
                  'Invalid start ticks')
            uids = identity['uids']
            check(type(uids) in (list, tuple) and len(uids) == 4
                  and all(type(u) is int and u >= 0 for u in uids), 'Invalid UID identity')
            executable = identity['executable']
            denied = identity.get('executable_permission_denied', False)
            # Preserve the corrected observer's explicit ptrace denial. This
            # represents unavailable evidence; ownership/display rules below
            # still reconstruct every raw context independently.
            check(type(denied) is bool and (
                  (executable is None and denied is True) or
                  (type(executable) is str and executable.startswith('/') and denied is False)),
                  'Invalid executable visibility')
            check(type(identity['cgroup']) is str, 'Invalid cgroup')
        used = set()
        def group(pid):
            used.add(str(pid))
            return identities[str(pid)]['cgroup']
        rebuilt = snapshot(queries[0]['stdout'], queries[1]['stdout'], device_map,
                           assigned, own_group, group,
                           lambda row: system_display(row, identities[str(row['pid'])], gdm_uid))
        check(used == set(identities), 'Unreferenced or missing identities')
        # JSON object keys convert frozen integer indices into strings.
        check(value['device_map'] == {str(k): v for k, v in device_map.items()}, 'Wrong serialized map')
        check(value['own_unit'] == unit_path(own_group), 'Wrong dispatcher unit')
        for field in ('contexts', 'foreign_contexts', 'system_displays'):
            check(value[field] == rebuilt[field], 'Raw/summary disagreement: ' + field)
        check(all(row['uuid'] == assigned for row in rebuilt['contexts']), 'Own peer GPU context')
        check(not require_assigned or bool(rebuilt['contexts']), 'Assigned context not observed')
        return rebuilt
    except Exception as exc:
        raise ContractError(str(exc), packet) from exc
