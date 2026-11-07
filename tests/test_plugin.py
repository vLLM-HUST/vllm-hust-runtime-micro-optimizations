import importlib

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
