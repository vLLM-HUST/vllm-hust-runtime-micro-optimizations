import pytest

from vllm_hust_runtime_micro_optimizations.rewrites import (
    guard_empty_padding,
    reuse_query_start_location,
)


def test_empty_padding_rewrite_guards_both_writes():
    source = """def build():
    slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
    blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
"""
    result = guard_empty_padding(source)
    assert "if num_tokens < num_tokens_padded:" in result
    assert "if num_reqs < num_reqs_padded:" in result
    compile(result, "<test>", "exec")


@pytest.mark.parametrize("missing", ["slot_mapping", "blk_table_tensor"])
def test_empty_padding_rewrite_fails_closed(missing):
    source = """def build():
    slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
    blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
"""
    with pytest.raises(RuntimeError):
        guard_empty_padding("\n".join(line for line in source.splitlines() if missing not in line))


def test_qsl_rewrite_changes_only_reviewed_expression():
    source = """def build(self, common_attn_metadata, query_start_loc_cpu):
    return query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)
"""
    result = reuse_query_start_location(source)
    assert "_runtime_micro_choose_qsl" in result
    assert ".pin_memory()" not in result
    compile(result, "<test>", "exec")


def test_qsl_rewrite_fails_closed_on_duplicate_seam():
    expression = "query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)"
    with pytest.raises(RuntimeError):
        reuse_query_start_location(expression + "\n" + expression)
