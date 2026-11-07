"""Mountable, default-off runtime micro-optimization family."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import textwrap
import types
from pathlib import Path

from .contracts import HOST_REVISION, verify_file
from .rewrites import guard_empty_padding, reuse_query_start_location

_KNOWN = {"empty-padding", "qsl-reuse", "residual-scale"}
_INSTALLED: dict[str, tuple[object, str, object]] = {}
_EMITTED_EVENTS: set[str] = set()
_LOGGER = logging.getLogger(__name__)
_STATE = {
    "host_revision": HOST_REVISION,
    "selected": [],
    "empty_padding": {
        "executed_slot_mapping_fills": 0,
        "skipped_slot_mapping_fills": 0,
        "executed_block_table_fills": 0,
        "skipped_block_table_fills": 0,
    },
    "qsl_reuse": {
        "calls": 0,
        "eligible": 0,
        "reused": 0,
        "native_copies": 0,
        "avoided_bytes": 0,
    },
    "residual_scale": {
        "installed_modules": [],
        "residual_calls": 0,
        "native_calls": 0,
    },
}


def _emit_receipt_once(event: str) -> None:
    """Emit one machine-readable receipt per worker and activation event."""
    if event in _EMITTED_EVENTS:
        return
    _EMITTED_EVENTS.add(event)
    payload = receipt()
    payload.update({"event": event, "pid": os.getpid()})
    _LOGGER.info(
        "VLLM_HUST_RUNTIME_MICRO_RECEIPT %s", json.dumps(payload, sort_keys=True)
    )


def _replace_method(
    owner, name: str, transform, relative_path: str, globals_: dict
) -> None:
    if name in _INSTALLED:
        raise RuntimeError(f"{name} is already installed")
    original = getattr(owner, name)
    path = Path(inspect.getsourcefile(original) or "")
    verify_file(path, relative_path)
    source = textwrap.dedent(inspect.getsource(original))
    modified = transform(source)
    namespace = dict(original.__globals__)
    namespace.update(globals_)
    # The input is the SHA-256-pinned Host method plus one exact rewrite.
    exec(  # noqa: S102
        compile(modified, f"{path}::runtime-micro:{name}", "exec"), namespace
    )
    replacement = namespace[name]
    functools.update_wrapper(replacement, original)
    setattr(owner, name, replacement)
    _INSTALLED[name] = (owner, name, original)


def _install_empty_padding() -> None:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    _replace_method(
        NPUModelRunner,
        "_build_attention_metadata",
        guard_empty_padding,
        "vllm_ascend/worker/model_runner_v1.py",
        {
            "_runtime_micro_state": _STATE,
            "_runtime_micro_emit_activation": _emit_receipt_once,
        },
    )


def _qsl_eligible(builder, common, cpu) -> bool:
    import torch

    gpu = common.query_start_loc
    parallel = builder.vllm_config.parallel_config
    return bool(
        builder.__class__.__name__ == "AscendAttentionMetadataBuilder"
        and builder.pcp_size == 1
        and parallel.decode_context_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and builder.speculative_config is None
        and common.num_reqs == common.num_actual_tokens == common.max_query_len == 1
        and tuple(cpu.shape) == tuple(gpu.shape) == (2,)
        and cpu.device.type == "cpu"
        and gpu.device.type == "npu"
        and gpu.device == builder.device
        and cpu.dtype == gpu.dtype == torch.int32
        and cpu.is_contiguous()
        and gpu.is_contiguous()
    )


def _choose_qsl(builder, common, cpu):
    state = _STATE["qsl_reuse"]
    state["calls"] += 1
    eligible = _qsl_eligible(builder, common, cpu)
    state["eligible"] += int(eligible)
    if eligible:
        state["reused"] += 1
        state["avoided_bytes"] += cpu.numel() * cpu.element_size()
        _emit_receipt_once("qsl-reuse-exercised")
        return common.query_start_loc
    state["native_copies"] += 1
    return cpu.pin_memory().to(builder.device, non_blocking=True)


def _install_qsl_reuse() -> None:
    from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder

    _replace_method(
        AscendAttentionMetadataBuilder,
        "build",
        reuse_query_start_location,
        "vllm_ascend/attention/attention_v1.py",
        {"_runtime_micro_choose_qsl": _choose_qsl},
    )


def _cached_gemma_forward(module, x, residual=None):
    if residual is None:
        _STATE["residual_scale"]["native_calls"] += 1
        return module._runtime_micro_original_forward(x, residual)
    _STATE["residual_scale"]["residual_calls"] += 1
    import torch
    import torch_npu
    from vllm_ascend.ops.layernorm import _HAS_ACLNN_ADD_RMS_NORM_BIAS
    from vllm_ascend.utils import enable_custom_op

    if enable_custom_op() and _HAS_ACLNN_ADD_RMS_NORM_BIAS:
        x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
            x, residual, module._runtime_micro_scale, None, module.variance_epsilon
        )
    else:
        x, _, residual = torch_npu.npu_add_rms_norm(
            x, residual, module._runtime_micro_scale, module.variance_epsilon
        )
    return x, residual


def _install_residual_scale() -> None:
    from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm
    from vllm_ascend.worker.worker import NPUWorker

    verify_file(
        inspect.getsourcefile(AscendGemmaRMSNorm) or "",
        "vllm_ascend/ops/layernorm.py",
    )
    verify_file(
        inspect.getsourcefile(NPUWorker) or "",
        "vllm_ascend/worker/worker.py",
    )
    source = inspect.getsource(AscendGemmaRMSNorm.forward_oot)
    if "1.0 + self.weight" not in source:
        raise RuntimeError("Reviewed Gemma residual-scale seam is absent")

    original = NPUWorker.load_model
    if "load_model" in _INSTALLED:
        raise RuntimeError("load_model is already installed")

    @functools.wraps(original)
    def after_load(worker, *args, **kwargs):
        result = original(worker, *args, **kwargs)
        model = worker.model_runner.model
        installed = []
        for name, module in model.named_modules():
            if not isinstance(module, AscendGemmaRMSNorm):
                continue
            if hasattr(module, "_runtime_micro_original_forward"):
                raise RuntimeError(f"Residual-scale cache already owns {name}")
            module._runtime_micro_original_forward = module._forward_method
            module.register_buffer(
                "_runtime_micro_scale", 1.0 + module.weight, persistent=False
            )
            module._forward_method = types.MethodType(_cached_gemma_forward, module)
            installed.append(name)
        _STATE["residual_scale"]["installed_modules"] = installed
        if installed:
            _emit_receipt_once("residual-scale-bound")
        return result

    NPUWorker.load_model = after_load
    _INSTALLED["load_model"] = (NPUWorker, "load_model", original)


def load() -> None:
    """vLLM plugin entry point. Empty selection is deliberately a no-op."""
    raw = os.getenv("VLLM_HUST_RUNTIME_MICRO_OPTIMIZATIONS", "")
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected) - _KNOWN)
    if unknown:
        raise RuntimeError(f"Unknown runtime micro-optimization(s): {unknown}")
    if len(selected) != len(set(selected)):
        raise RuntimeError("Duplicate runtime micro-optimization selector")
    _STATE["selected"] = selected
    for item in selected:
        {
            "empty-padding": _install_empty_padding,
            "qsl-reuse": _install_qsl_reuse,
            "residual-scale": _install_residual_scale,
        }[item]()
    if selected:
        _emit_receipt_once("hooks-installed")


def receipt() -> dict:
    """Return a JSON-serializable activation receipt."""
    import copy

    result = copy.deepcopy(_STATE)
    result["installed"] = sorted(_INSTALLED)
    result["receipt_sha256"] = hashlib.sha256(
        repr(
            (result["host_revision"], result["selected"], result["installed"])
        ).encode()
    ).hexdigest()
    return result
