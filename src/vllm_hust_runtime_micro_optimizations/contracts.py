"""Pinned Host source contracts used by the default-off mechanisms."""

from __future__ import annotations

import hashlib
from pathlib import Path

HOST_REPOSITORY = "vLLM-HUST/vllm-ascend-hust"
HOST_REVISION = "7602797593170da4c531f1d8882a7b82dce8a1bc"

FILES = {
    "vllm_ascend/worker/model_runner_v1.py": "d554280ddaf8ec25043cff9195773a03d16b4f331fb868550ce7842c11d4aa62",
    "vllm_ascend/attention/attention_v1.py": "44c43d3610fbc5d50b2a52ef82d5b7ff54b7a1121e5b83b6c9d046f17004c36f",
    "vllm_ascend/ops/layernorm.py": "7c03ecf2e28c75335688a78e78444bac6e2d38be9c42d0798d18ea221158473b",
    "vllm_ascend/worker/worker.py": "10ef420b01c7e0355e27c455808e84a2b0c3b2f66e50e495ad7c178cebaabdac",
}


class HostContractError(RuntimeError):
    """Raised before patching when the reviewed Host source has changed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: str | Path, relative: str) -> dict[str, str]:
    actual = sha256(Path(path))
    expected = FILES[relative]
    if actual != expected:
        raise HostContractError(
            f"Host source mismatch for {relative}: expected {expected}, got {actual}"
        )
    return {"path": relative, "sha256": actual}


def verify_host(root: str | Path) -> list[dict[str, str]]:
    root = Path(root)
    return [verify_file(root / relative, relative) for relative in FILES]
