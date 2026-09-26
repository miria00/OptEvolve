#!/usr/bin/env python
"""Print this machine's device record: kind, platform, memory, measured
f32/f64 matmul TFLOPS.

Works on a CPU-pinned box (the atlas default next to a vLLM server): the GPU
is probed in a short subprocess with JAX_PLATFORMS=cuda and preallocation
disabled, so probe memory stays in the low hundreds of MB.

Usage:
    python scripts/probe_hardware.py [--no-gpu] [--json-out PATH]
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gpu", action="store_true",
                    help="probe only the process-default (CPU-pinned) device")
    ap.add_argument("--json-out", default=None,
                    help="also write the record to this JSON file")
    args = ap.parse_args()

    import atlas  # noqa: F401  (pins JAX_PLATFORMS=cpu + x64 first)
    from atlas.backend.device import detect_device

    rec = dict(detect_device(prefer_gpu=not args.no_gpu))
    rec["hostname"] = socket.gethostname()
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    text = json.dumps(rec, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
