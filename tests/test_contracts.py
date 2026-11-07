import ast
import textwrap
from pathlib import Path

import pytest

from vllm_hust_runtime_micro_optimizations.contracts import (
    HostContractError,
    verify_host,
)
from vllm_hust_runtime_micro_optimizations.rewrites import (
    guard_empty_padding,
    reuse_query_start_location,
)


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    matches = [
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
        for method in node.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and method.name == method_name
    ]
    assert len(matches) == 1
    return textwrap.dedent(ast.get_source_segment(source, matches[0]))


def test_current_host_contract_if_requested():
    import os

    root = os.getenv("VLLM_ASCEND_HOST_ROOT")
    if not root:
        pytest.skip("set VLLM_ASCEND_HOST_ROOT for the checked-out Host contract")
    checks = verify_host(root)
    assert len(checks) == 4
    empty_padding = _method_source(
        Path(root) / "vllm_ascend/worker/model_runner_v1.py",
        "NPUModelRunner",
        "_build_attention_metadata",
    )
    qsl = _method_source(
        Path(root) / "vllm_ascend/attention/attention_v1.py",
        "AscendAttentionMetadataBuilder",
        "build",
    )
    compile(guard_empty_padding(empty_padding), "<current-empty-padding>", "exec")
    compile(reuse_query_start_location(qsl), "<current-qsl>", "exec")


def test_host_contract_fails_closed(tmp_path: Path):
    for relative in (
        "vllm_ascend/worker/model_runner_v1.py",
        "vllm_ascend/attention/attention_v1.py",
        "vllm_ascend/ops/layernorm.py",
        "vllm_ascend/worker/worker.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("changed\n")
    with pytest.raises(HostContractError):
        verify_host(tmp_path)
