# SPDX-License-Identifier: Apache-2.0
import numpy as np
from cacheflowv3.config import CacheFlowConfig, GiB
from cacheflowv3.hashing import chunk_keys, namespace_seed
from cacheflowv3.layout import ChunkGeometry, plan_ops


def test_plan_ops_pull_layerwise():
    g = ChunkGeometry(num_layers=3, layer_bytes=100)
    groups = plan_ops(g, 10_000, 7, [1_000_000, 2_000_000], 9, layer_groups=True)
    assert len(groups) == 3
    l1 = groups[1]
    # staging is layer-major: layer 1 of chunk 0 at (1*2+0)*100, chunk 1 at (1*2+1)*100
    assert list(l1["laddr"]) == [10_200, 10_300]
    assert list(l1["raddr"]) == [1_000_100, 2_000_100]
    assert set(l1["lkey"]) == {7} and set(l1["rkey"]) == {9} and set(l1["len"]) == {100}
    (flat,) = plan_ops(g, 10_000, 7, [1_000_000, 2_000_000], 9, layer_groups=False)
    assert len(flat) == 6 and list(flat["laddr"][2:4]) == [10_200, 10_300]


def test_plan_ops_push_subset_and_layer_range():
    g = ChunkGeometry(num_layers=4, layer_bytes=10)
    (ops,) = plan_ops(
        g,
        500,
        1,
        [7000],
        2,
        layer_groups=False,
        staging_is_local=False,
        staging_chunks=[2],
        n_staging=3,
        layers=(1, 3),
    )
    # server side: local = entry, remote = client staging (layer l, chunk 2 of 3)
    assert list(ops["laddr"]) == [7010, 7020]
    assert list(ops["raddr"]) == [500 + (1 * 3 + 2) * 10, 500 + (2 * 3 + 2) * 10]
    assert set(ops["lkey"]) == {2} and set(ops["rkey"]) == {1}


def test_chunk_keys_chain_and_namespace():
    seed = namespace_seed("m", 1)
    toks = list(range(1000))
    k = chunk_keys(toks, 256, 10, seed)
    assert len(k) == 3  # only full chunks
    assert chunk_keys(toks[:600], 256, 10, seed) == k[:2]  # prefix property
    other = list(toks)
    other[5] = -1
    assert (
        chunk_keys(other, 256, 10, seed)[2] != k[2]
    )  # chaining: early change propagates
    assert chunk_keys(toks, 256, 10, namespace_seed("m", 2))[0] != k[0]


def test_config_v1_aliases():
    c = CacheFlowConfig.from_extra_config(
        {
            "use_doca_buffer_pool": "false",
            "tokens_per_block": 128,
            "common_prefix_num_tokens": 1000,
            "num_staging_buffers": 16,
            "block_size": 536870912,
            "max_blocks": 20,
            "async_transfers": False,
        }
    )
    assert c.use_registered_buffer_pool is False and c.chunk_tokens == 128
    assert c.max_cached_tokens == 1000 and c.staging_pool_gb == 8.0
    assert c.dpu_capacity_gb == 10.0 and c.async_transfers is False
    assert c.cacheable_tokens(5000) == 896  # capped at 1000, rounded to chunks
    c2 = CacheFlowConfig.from_extra_config(
        {"offload_full_prompt": True, "common_prefix_num_tokens": 10}
    )
    assert c2.cacheable_tokens(5000) == 5000 - 5000 % 256
    assert (
        CacheFlowConfig.from_extra_config({"min_cached_tokens": 1024}).cacheable_tokens(
            1000
        )
        == 0
    )


def test_config_rejects_bad_modes():
    import pytest

    with pytest.raises(ValueError):
        CacheFlowConfig.from_extra_config({"transfer_mode": "sideways"})
    with pytest.raises(ValueError):
        CacheFlowConfig.from_extra_config({"data_path": "gpudirect"})
    assert GiB == 1 << 30
    assert np.dtype("uint64").itemsize == 8
