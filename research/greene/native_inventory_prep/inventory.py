"""CPU-tested preparation only; no native imports, signals, or launch entrypoint."""
import csv
import io
from pathlib import Path
import re
import xml.etree.ElementTree as ET

UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
UNIT = re.compile(r"openwam-nightly-job-[A-Za-z0-9_.\\-]+[.]service")

class InventoryError(RuntimeError):
    pass

def require(value, message):
    if not value:
        raise InventoryError(message)

def unit_path(text):
    rows = text.strip().splitlines()
    require(len(rows) == 1 and rows[0].startswith("0::/"), "Unified cgroup v2 required")
    path = rows[0][3:]
    parts = path.split("/")
    require(all(p not in (".", "..") for p in parts), "Invalid cgroup path")
    positions = [i for i, p in enumerate(parts) if UNIT.fullmatch(p)]
    require(len(positions) == 1, "Exactly one dispatcher service required")
    return "/".join(parts[:positions[0]+1])

def belongs(text, own_unit):
    rows = text.strip().splitlines()
    require(len(rows) == 1 and rows[0].startswith("0::/"), "Unreadable unified cgroup")
    path = rows[0][3:]
    require(all(p not in (".", "..") for p in path.split("/")), "Invalid cgroup path")
    return path == own_unit or path.startswith(own_unit + "/")

def start_ticks(text, pid):
    require(text.startswith(str(pid) + " ("), "Unexpected process identity")
    fields = text.rsplit(")", 1)[1].split()
    require(len(fields) >= 20 and fields[19].isdigit(), "Malformed /proc stat")
    return int(fields[19])

def process_group(pid, proc=Path("/proc")):
    """Fail closed on disappearance/permissions/reuse; never signal any process."""
    try:
        root = proc / str(pid)
        before = start_ticks((root / "stat").read_text(), pid)
        group = (root / "cgroup").read_text()
        after = start_ticks((root / "stat").read_text(), pid)
        require(before == after, "PID reused during ownership read")
        return group
    except (OSError, ValueError, IndexError) as exc:
        raise InventoryError("Unavailable process ownership") from exc

def snapshot(xml_text, csv_text, expected, assigned, own_group, read_group=process_group):
    """Validate one host's complete eight-card snapshot; retain all own contexts.

    expected is a frozen index->UUID map, not inferred from these observations.
    XML and CSV must be obtained by separately bounded read-only queries.
    A future production adapter must establish freshness and host identity.
    """
    require(set(expected) == set(range(8)), "Expected frozen eight-card map")
    require(all(isinstance(u, str) and UUID.fullmatch(u) for u in expected.values()), "Invalid expected UUID")
    require(len(set(expected.values())) == 8 and assigned in expected.values(), "Duplicate/unknown assignment")
    own_unit = unit_path(own_group)
    try:
        nodes = ET.fromstring(xml_text).findall("gpu")
        require(len(nodes) == 8, "Incomplete XML devices")
        xml_ids = [g.findtext("uuid") for g in nodes]
        require(len(set(xml_ids)) == 8 and set(xml_ids) == set(expected.values()), "XML device identity mismatch")
        rows = list(csv.reader(io.StringIO(csv_text)))
        require(len(rows) == 8 and all(len(r) == 2 for r in rows), "Incomplete CSV identities")
        pairs = [(int(r[0].strip()), r[1].strip()) for r in rows]
        require(len(set(i for i, _ in pairs)) == 8 and dict(pairs) == expected, "CSV device identity mismatch")
        contexts, foreign, seen = [], [], set()
        for gpu in nodes:
            uuid = gpu.findtext("uuid")
            process_list = gpu.find("processes")
            require(process_list is not None, "Missing process inventory")
            require(not (process_list.text or "").strip() and
                    all(c.tag == "process_info" for c in process_list),
                    "Unsupported process inventory")
            for process in gpu.findall("./processes/process_info"):
                raw = process.findtext("pid") or ""
                require(raw.isdigit() and int(raw) > 0, "Malformed PID")
                pid = int(raw)
                require((uuid, pid) not in seen, "Duplicate process entry")
                seen.add((uuid, pid))
                kind = process.findtext("type")
                require(kind in ("C", "G", "C+G"), "Unknown GPU context type")
                memory = process.findtext("used_memory") or ""
                require(re.fullmatch(r"[0-9]+ MiB", memory), "Malformed context memory")
                name = process.findtext("process_name")
                require(isinstance(name, str) and bool(name), "Missing process name")
                group = read_group(pid)
                owned = belongs(group, own_unit)
                row = dict(uuid=uuid, pid=pid, type=kind, memory_mib=int(memory.split()[0]), name=name)
                if owned:
                    contexts.append(row)
                else:
                    foreign.append(row)
        # Raw XML lacks trusted UID. An assigned-card foreign display also
        # fails closed until a future adapter verifies the root/gdm exemption.
        require(not any(p["uuid"] == assigned for p in foreign),
                "Foreign or unverified process on assigned card")
        return dict(complete=True, contexts=contexts, foreign_contexts=foreign,
                    device_map=dict(pairs), own_unit=own_unit)
    except (ET.ParseError, ValueError, TypeError, IndexError, OSError) as exc:
        raise InventoryError("Malformed or unavailable inventory") from exc
