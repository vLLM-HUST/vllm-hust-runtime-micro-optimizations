# vLLM-HUST Runtime Micro-optimizations

Default-off runtime metadata and micro-overhead optimization MOD family for
vLLM-HUST. The package is mountable through vLLM's general-plugin entry point,
but loading it does nothing unless mechanisms are explicitly selected.

## Scope

Decode metadata transfer, copy reuse, block-table reuse, fragment assembly and sampling-path micro-overheads.

This repository is the canonical integration and evidence home for the family. Every optimization is opt-in and disabled by default until its own compatibility and performance qualification is complete.

## Current mechanisms

| Mechanism | Selector | Current state |
| --- | --- | --- |
| Empty-padding write guard | `empty-padding` | Ported to current V1 runner; pending NPU performance qualification |
| Query-start-location reuse | `qsl-reuse` | Ported to current attention builder; pending NPU performance qualification |
| Gemma residual-scale cache | `residual-scale` | Generalized beyond the original model shape; pending NPU performance qualification |
| RoPE compatibility fix | — | Retired: the fix is already part of the current Host |

Install the package, then opt in per worker with a comma-separated list:

```bash
VLLM_HUST_RUNTIME_MICRO_OPTIMIZATIONS=empty-padding,qsl-reuse \
  vllm serve ...
```

Unknown selectors and incompatible Host source fail closed. A successful
installation can be audited with
`vllm_hust_runtime_micro_optimizations.receipt()`.

## Compatibility check

The current source contract is pinned to `vLLM-HUST/vllm-ascend-hust` commit
`7602797593170da4c531f1d8882a7b82dce8a1bc`. Before an NPU run:

```bash
python tools/check_host_contract.py /path/to/vllm-ascend-hust
pytest -q
```

Passing these checks establishes source compatibility only. It is not a
performance qualification and must not be used to publish speedup numbers.

## Ownership

- Maintainer: [ShuhaoZhangTony](https://github.com/ShuhaoZhangTony)
- Advisors: none recorded
- Organization: [vLLM-HUST](https://github.com/vLLM-HUST)

## Evidence

Measured single-MOD, composition, and compatibility results are published through [vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark) and presented with the corresponding MOD in the [MOD catalog](https://vllm-hust.sage.org.ai/plugins.html).
