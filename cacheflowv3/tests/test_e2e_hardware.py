# SPDX-License-Identifier: Apache-2.0
"""End-to-end save -> lookup -> load against a live CacheFlow v3 server.

Needs a GPU, the RDMA link and a running server; skipped otherwise:

    CACHEFLOW_SERVER=10.0.1.2 pytest -q tests/test_e2e_hardware.py

Every ablation switch that changes the data path is exercised, and the loaded
KV must match the saved KV bit for bit.
"""

import itertools
import os
import uuid

import pytest
import torch
from cacheflowv3.config import CacheFlowConfig
from cacheflowv3.engine import TransferEngine
from cacheflowv3.hashing import chunk_keys, namespace_seed
from cacheflowv3.kv_access import KVAccessor

SERVER = os.environ.get("CACHEFLOW_SERVER")
pytestmark = pytest.mark.skipif(
    not SERVER or not torch.cuda.is_available(),
    reason="set CACHEFLOW_SERVER and use a GPU host",
)

L, NB, B, H, D = 6, 96, 16, 8, 128
C, N_CHUNKS = 64, 4


def _cfg(**kw) -> CacheFlowConfig:
    return CacheFlowConfig.from_extra_config(
        {
            "server_host": SERVER,
            "chunk_tokens": C,
            "staging_pool_gb": 0.25,
            "stats_log_interval_s": 1e9,
            **kw,
        }
    )


def _slots(block_ids, start, stop):
    t = torch.arange(start, stop)
    return torch.tensor(block_ids)[t // B], t % B


def _kv(fill):
    return [fill(2, NB, B, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(L)]


def _save(engine, keys, kv, block_ids):
    blk, off = _slots(block_ids, 0, len(keys) * C)
    job = engine.begin_save(
        "req-save", keys, blk.view(-1, C).cuda(), off.view(-1, C).cuda()
    )
    assert job is not None
    for layer in range(L):
        engine.save_layer(job, layer, kv[layer])
    engine.end_save_step(job)
    assert job.done.wait(60)
    return job


def _load(engine, keys, kv_dst, block_ids, tok_start, tok_stop):
    r = engine.session.ctl.request(
        "lookup", keys=keys, ranks=1, pin=True, lease_ms=30_000
    )
    assert r["hits"] == len(keys)
    c0, c1 = tok_start // C, -(-tok_stop // C)
    job = engine.begin_load(
        "req-load",
        r["lease"],
        r["entries"][0][c0:c1],
        r["rkey"],
        c0,
        tok_start,
        tok_stop,
        block_ids,
    )
    for layer in range(L):
        engine.load_layer(job, layer, kv_dst[layer])
    engine.finish_load(job)
    torch.cuda.synchronize()
    assert not job.failed


MODES = list(
    itertools.product(
        ["pull", "push"],
        ["mapped", "staged"],
        [True, False],
        [True, False],
        [True, False],
    )
)


@pytest.mark.parametrize("mode,path,layerwise,overlap,pool", MODES)
def test_roundtrip(mode, path, layerwise, overlap, pool):
    engine = TransferEngine(
        _cfg(
            transfer_mode=mode,
            data_path=path,
            layerwise_load=layerwise,
            overlap_dma_with_copy=overlap,
            use_registered_buffer_pool=pool,
        ),
        rank=0,
        world=1,
        device=torch.device("cuda", 0),
    )
    try:
        src = _kv(torch.randn)
        engine.set_geometry(KVAccessor(src[0], B), L, B)
        tokens = torch.randint(0, 32000, (N_CHUNKS * C + 10,)).tolist()
        keys = chunk_keys(tokens, C, N_CHUNKS, namespace_seed("e2e", uuid.uuid4()))
        src_blocks = torch.randperm(NB)[:20].tolist()
        assert _save(engine, keys, src, src_blocks).ok

        # full-prefix load into different blocks, last token left for vLLM
        dst_blocks = torch.randperm(NB)[:20].tolist()
        dst = _kv(torch.zeros)
        _load(engine, keys, dst, dst_blocks, 0, N_CHUNKS * C - 1)
        sb, so = _slots(src_blocks, 0, N_CHUNKS * C - 1)
        db, do = _slots(dst_blocks, 0, N_CHUNKS * C - 1)
        for layer in range(L):
            assert torch.equal(dst[layer][:, db, do], src[layer][:, sb, so]), (
                f"layer {layer}"
            )
        lb, lo = _slots(dst_blocks, N_CHUNKS * C - 1, N_CHUNKS * C)
        assert not dst[0][:, lb, lo].any()  # the final token was not written

        # partial load (vLLM already has the first chunk locally)
        dst2 = _kv(torch.zeros)
        _load(engine, keys, dst2, dst_blocks, C, 3 * C + 5)
        db, do = _slots(dst_blocks, C, 3 * C + 5)
        sb, so = _slots(src_blocks, C, 3 * C + 5)
        assert torch.equal(dst2[2][:, db, do], src[2][:, sb, so])
        written = int((dst2[2] != 0).any(dim=(3, 4)).sum())
        assert written == 2 * (3 * C + 5 - C)
    finally:
        engine.close()


def test_skip_existing_dedups_saves():
    engine = TransferEngine(_cfg(), rank=0, world=1, device=torch.device("cuda", 0))
    try:
        src = _kv(torch.randn)
        engine.set_geometry(KVAccessor(src[0], B), L, B)
        tokens = torch.randint(0, 32000, (N_CHUNKS * C,)).tolist()
        keys = chunk_keys(tokens, C, N_CHUNKS, namespace_seed("dedup", uuid.uuid4()))
        blocks = list(range(20))
        _save(engine, keys, src, blocks)
        before = engine.snapshot()
        _save(engine, keys, src, blocks)
        after = engine.snapshot()
        assert after["save_chunks_exists"] - before["save_chunks_exists"] == N_CHUNKS
        assert after["save_bytes"] == before["save_bytes"]
    finally:
        engine.close()
