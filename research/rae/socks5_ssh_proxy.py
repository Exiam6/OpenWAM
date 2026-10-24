#!/usr/bin/env python3
"""OpenSSH ProxyCommand bridge through the local Tailscale SOCKS5 listener.

Usage: ssh -o 'ProxyCommand=python research/rae/socks5_ssh_proxy.py %h %p' ...
"""

from __future__ import annotations

import os
import select
import socket
import struct
import sys


def recv_exact(stream: socket.socket, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        block = stream.recv(size - len(value))
        if not block:
            raise ConnectionError("SOCKS5 proxy closed during handshake")
        value.extend(block)
    return bytes(value)


def connect(host: str, port: int) -> socket.socket:
    stream = socket.create_connection(("127.0.0.1", 1055), timeout=15)
    stream.sendall(b"\x05\x01\x00")
    if recv_exact(stream, 2) != b"\x05\x00":
        raise ConnectionError("SOCKS5 proxy rejected no-authentication method")
    try:
        address = socket.inet_pton(socket.AF_INET, host)
        request = b"\x05\x01\x00\x01" + address
    except OSError:
        encoded = host.encode("idna")
        if len(encoded) > 255:
            raise ValueError("Host name is too long for SOCKS5")
        request = b"\x05\x01\x00\x03" + bytes([len(encoded)]) + encoded
    stream.sendall(request + struct.pack("!H", port))
    header = recv_exact(stream, 4)
    if header[:1] != b"\x05" or header[1] != 0:
        raise ConnectionError(f"SOCKS5 connect failed: {header.hex()}")
    address_type = header[3]
    if address_type == 1:
        recv_exact(stream, 4)
    elif address_type == 4:
        recv_exact(stream, 16)
    elif address_type == 3:
        recv_exact(stream, recv_exact(stream, 1)[0])
    else:
        raise ConnectionError(f"Unknown SOCKS5 address type {address_type}")
    recv_exact(stream, 2)
    stream.settimeout(None)
    return stream


def forward(stream: socket.socket) -> None:
    input_open = True
    while True:
        ready, _, _ = select.select(([0] if input_open else []) + [stream], [], [])
        if 0 in ready:
            block = os.read(0, 65536)
            if block:
                stream.sendall(block)
            else:
                input_open = False
                stream.shutdown(socket.SHUT_WR)
        if stream in ready:
            block = stream.recv(65536)
            if not block:
                return
            view = memoryview(block)
            while view:
                view = view[os.write(1, view):]


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: socks5_ssh_proxy.py HOST PORT")
    host, port_text = sys.argv[1:]
    try:
        with connect(host, int(port_text)) as stream:
            forward(stream)
    except (ConnectionError, OSError, ValueError) as exc:
        print(f"SOCKS5 SSH proxy: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
