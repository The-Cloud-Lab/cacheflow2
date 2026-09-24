# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from cacheflowv3.kv_access import KVAccessor, UnsupportedKVLayout

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

B, H, D, NB = 16, 4, 8, 32


def _cache(kind):
    if kind == "kv_first":
        return torch.randn(2, NB, B, H, D, device="cuda", dtype=torch.float16)
    if kind == "blocks_first":
        return torch.randn(NB, 2, B, H, D, device="cuda", dtype=torch.float16)
    return torch.randn(NB, B, H * D, device="cuda", dtype=torch.float16)


def _token(kv, kind, blk, off):
    if kind == "kv_first":
        return torch.stack([kv[0, blk, off].flatten(), kv[1, blk, off].flatten()])
    if kind == "blocks_first":
        return torch.stack([kv[blk, 0, off].flatten(), kv[blk, 1, off].flatten()])
    return kv[blk, off].flatten()[None]


@cuda
@pytest.mark.parametrize("kind", ["kv_first", "blocks_first", "mla"])
def test_gather_scatter_roundtrip(kind):
    kv = _cache(kind)
    acc = KVAccessor(kv, B)
    assert acc.kind == kind
    n, C = 3, 20  # chunk size need not be a multiple of the block size
    block_ids = torch.randperm(NB)[:8].tolist()
    t = torch.arange(n * C)
    blk = torch.tensor(block_ids)[t // B].view(n, C).cuda()
    off = (t % B).view(n, C).cuda()
    out = torch.empty(
        n, acc.planes, C, acc.feat, device="cuda", dtype=kv.dtype
    ).permute(1, 0, 2, 3)
    acc.gather(kv, blk, off, out)
    for i, j in [(0, 0), (1, 7), (2, 19)]:
        ref = _token(kv, kind, int(blk[i, j]), int(off[i, j]))
        assert torch.equal(out[:, i, j], ref)
    # scatter a token subset into a fresh cache and compare
    kv2 = torch.zeros_like(kv)
    sel_j = torch.tensor([0, 1, 1, 2], device="cuda")
    sel_p = torch.tensor([5, 0, 19, 3], device="cuda")
    acc.scatter(kv2, blk[sel_j, sel_p], off[sel_j, sel_p], out[:, sel_j, sel_p])
    for j, p in zip(sel_j.tolist(), sel_p.tolist()):
        b, o = int(blk[j, p]), int(off[j, p])
        assert torch.equal(_token(kv2, kind, b, o), _token(kv, kind, b, o))
    assert int((kv2 != 0).sum()) == 4 * acc.planes * acc.feat


def test_rejects_unknown_layout():
    with pytest.raises(UnsupportedKVLayout):
        KVAccessor(torch.zeros(3, 5, 7), 16)
