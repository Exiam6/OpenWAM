"""List the GPU targets embedded in a CUDA fatbin inside a shared library.

No cuobjdump on this box, but the format is simple enough: the .nv_fatbin
section holds one or more containers (magic 0xBA55ED50), each a sequence of
entries whose header carries kind (1 = PTX, 2 = SASS/ELF) and the SM number.
Only the headers are read; payloads are skipped by their padded size.
"""
import struct
import sys

MAGIC = 0xBA55ED50
KIND = {1: "PTX", 2: "SASS", 4: "DWARF"}


def section(path, name):
    data = open(path, "rb").read()
    if data[:4] != b"\x7fELF":
        raise SystemExit("not an ELF: " + path)
    is64 = data[4] == 2
    if not is64:
        raise SystemExit("32-bit ELF not handled")
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    shdrs = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack_from("<IIQQQQ", data, off)
        shdrs.append((sh_name, sh_offset, sh_size))
    str_off = shdrs[e_shstrndx][1]
    for sh_name, sh_offset, sh_size in shdrs:
        end = data.index(b"\x00", str_off + sh_name)
        if data[str_off + sh_name:end].decode() == name:
            return data[sh_offset:sh_offset + sh_size]
    raise SystemExit(f"no section {name} in {path}")


def targets(blob):
    out = set()
    pos = 0
    while True:
        pos = blob.find(struct.pack("<I", MAGIC), pos)
        if pos < 0:
            break
        magic, version, hsize, fsize = struct.unpack_from("<IHHQ", blob, pos)
        p, end = pos + hsize, pos + hsize + fsize
        while p + 0x20 <= end:
            kind, ver, ehsize, padded, _u0, payload, _u1, arch = struct.unpack_from("<HHIQIIII", blob, p)
            if ehsize == 0 or padded == 0:
                break
            if kind in KIND and 30 <= arch <= 200:
                out.add((KIND[kind], arch))
            p += ehsize + padded
        pos = end if end > pos else pos + 4
    return out


for path in sys.argv[1:]:
    t = targets(section(path, ".nv_fatbin"))
    sass = sorted(a for k, a in t if k == "SASS")
    ptx = sorted(a for k, a in t if k == "PTX")
    print(path.rsplit("/", 1)[-1])
    print("  SASS (native kernels):", " ".join(f"sm_{a}" for a in sass) or "none")
    print("  PTX  (JIT-able)      :", " ".join(f"compute_{a}" for a in ptx) or "none")
