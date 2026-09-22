#!/usr/bin/env bash
set -euo pipefail
cd /sgl-workspace/sglang
python -m pytest -q --tb=short -p no:cacheprovider \
    test/registered/unit/entrypoints/openai/test_encoding_dsv41.py \
    test/registered/unit/disaggregation/test_dsv41_dspark_pd.py \
    test/registered/unit/mem_cache/test_unified_tree_insert_cursor.py \
    test/registered/unit/spec/test_dspark_draft_metadata.py \
    test/registered/unit/layers/test_dsv4_bounded_prefill_graph.py \
    test/registered/spec/dspark/test_dspark_dp_tier.py \
    test/registered/unit/model_executor/test_pool_configurator.py \
    test/registered/unit/layers/test_dsv4_nonpaged_indexer.py \
    test/registered/unit/models/test_deepseek_v4_shared_expert_fusion.py \
    /opt/sglang-fl/tests/dsv41
