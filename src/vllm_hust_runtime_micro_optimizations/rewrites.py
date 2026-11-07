"""Pure source transformations, separated so they can be tested without NPU."""

from __future__ import annotations

EMPTY_PADDING_SITES = (
    (
        "slot_mapping[num_tokens:num_tokens_padded].fill_(-1)",
        "num_tokens < num_tokens_padded",
        "slot_mapping",
    ),
    (
        "blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)",
        "num_reqs < num_reqs_padded",
        "block_table",
    ),
)

QSL_EXPRESSION = "query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)"


def guard_empty_padding(source: str) -> str:
    """Guard the two empty-slice device writes and reject an unknown seam."""
    for expression, condition, counter in EMPTY_PADDING_SITES:
        lines = source.splitlines()
        matches = [i for i, line in enumerate(lines) if line.strip() == expression]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one Host seam for {expression!r}; got {matches}")
        index = matches[0]
        indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
        lines[index : index + 1] = [
            f"{indent}if {condition}:",
            f"{indent}    {expression}",
            f"{indent}    _runtime_micro_state['empty_padding']['executed_{counter}_fills'] += 1",
            f"{indent}else:",
            f"{indent}    _runtime_micro_state['empty_padding']['skipped_{counter}_fills'] += 1",
        ]
        source = "\n".join(lines) + "\n"
    return source


def reuse_query_start_location(source: str) -> str:
    if source.count(QSL_EXPRESSION) != 1:
        raise RuntimeError("Expected exactly one query-start-location Host seam")
    return source.replace(
        QSL_EXPRESSION,
        "_runtime_micro_choose_qsl(self, common_attn_metadata, query_start_loc_cpu)",
    )
