import importlib
import json
import logging

import pytest


def fresh_module():
    import vllm_hust_runtime_micro_optimizations as module

    return importlib.reload(module)


def test_default_load_is_noop(monkeypatch):
    monkeypatch.delenv("VLLM_HUST_RUNTIME_MICRO_OPTIMIZATIONS", raising=False)
    module = fresh_module()
    module.load()
    assert module.receipt()["selected"] == []
    assert module.receipt()["installed"] == []


def test_unknown_selector_fails_closed(monkeypatch):
    monkeypatch.setenv("VLLM_HUST_RUNTIME_MICRO_OPTIMIZATIONS", "not-a-mechanism")
    module = fresh_module()
    with pytest.raises(RuntimeError, match="Unknown"):
        module.load()


def test_duplicate_selector_fails_closed(monkeypatch):
    monkeypatch.setenv("VLLM_HUST_RUNTIME_MICRO_OPTIMIZATIONS", "qsl-reuse,qsl-reuse")
    module = fresh_module()
    with pytest.raises(RuntimeError, match="Duplicate"):
        module.load()


def test_selected_load_emits_structured_receipt(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_HUST_RUNTIME_MICRO_OPTIMIZATIONS", "empty-padding")
    module = fresh_module()
    monkeypatch.setattr(module, "_install_empty_padding", lambda: None)

    with caplog.at_level(logging.INFO):
        module.load()

    line = next(
        record.message
        for record in caplog.records
        if record.message.startswith("VLLM_HUST_RUNTIME_MICRO_RECEIPT ")
    )
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["event"] == "hooks-installed"
    assert payload["selected"] == ["empty-padding"]
    assert payload["pid"] > 0
