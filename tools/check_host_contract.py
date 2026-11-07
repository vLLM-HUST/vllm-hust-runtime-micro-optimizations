#!/usr/bin/env python3
"""Verify the reviewed Host files without importing the NPU runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_hust_runtime_micro_optimizations.contracts import (
    HOST_REPOSITORY,
    HOST_REVISION,
    verify_host,
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} /path/to/vllm-ascend-hust")
    checks = verify_host(sys.argv[1])
    print(
        json.dumps(
            {
                "repository": HOST_REPOSITORY,
                "revision": HOST_REVISION,
                "status": "pass",
                "files": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
