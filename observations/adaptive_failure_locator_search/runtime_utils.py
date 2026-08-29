#!/usr/bin/env python3
"""Read-only GPU selection and collision-resistant free-port discovery."""

from __future__ import annotations

import argparse
import socket
import subprocess


def select_gpu(candidates: str, minimum_free_gb: float) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, free_mib, gpu_util, memory_util = (
            int(value.strip()) for value in line.split(",")
        )
        rows.append(
            {
                "index": index,
                "free_mib": free_mib,
                "gpu_util": gpu_util,
                "memory_util": memory_util,
            }
        )
    if candidates.lower() not in {"all", "auto", ""}:
        allowed = {int(value.strip()) for value in candidates.split(",") if value.strip()}
        rows = [row for row in rows if row["index"] in allowed]
    eligible = [row for row in rows if row["free_mib"] >= float(minimum_free_gb) * 1024]
    if not eligible:
        detail = ", ".join(
            f"GPU{row['index']}:free={row['free_mib']/1024:.1f}GiB,util={row['gpu_util']}%"
            for row in rows
        )
        raise RuntimeError(f"no candidate GPU has >= {minimum_free_gb:g} GiB free; {detail}")
    eligible.sort(
        key=lambda row: (row["gpu_util"], row["memory_util"], -row["free_mib"], row["index"])
    )
    return int(eligible[0]["index"])


def free_port(start: int) -> int:
    for port in range(int(start), 65536):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free TCP port at or above {start}")


def port_available(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gpu = commands.add_parser("gpu")
    gpu.add_argument("--candidates", default="all")
    gpu.add_argument("--min-free-gb", type=float, default=24.0)
    port = commands.add_parser("port")
    port.add_argument("--start", type=int, default=36000)
    check = commands.add_parser("check-port")
    check.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.command == "gpu":
        print(select_gpu(args.candidates, args.min_free_gb))
    elif args.command == "port":
        print(free_port(args.start))
    else:
        return 0 if port_available(args.port) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
