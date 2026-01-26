#!/usr/bin/env python3
"""
Integration test for DOCA KV Cache Offload

This script tests the full pipeline:
1. CUDA pinned memory allocation
2. DOCA buffer registration
3. GPU -> Host -> DPU transfer

Run this on the host machine with DPU offloader running on the DPU.

Usage:
    # First, start the DPU offloader on the BlueField:
    ssh <dpu_ip> /path/to/dpu_offloader

    # Then run this test on the host:
    python test_offload.py

Requirements:
    - CUDA installed and working
    - libhost_provider.so built and accessible
    - DPU offloader running on BlueField
"""

import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from cuda_utils import PinnedBuffer, PinnedBufferPool, CUDARuntime
from doca_kv_offload import DOCAKVOffloadClient, find_bluefield_pci_address


def test_cuda_pinned_memory():
    """Test CUDA pinned memory allocation."""
    print("\n=== Test 1: CUDA Pinned Memory ===")

    # Allocate various sizes
    sizes = [1024, 1024 * 1024, 16 * 1024 * 1024]  # 1KB, 1MB, 16MB

    for size in sizes:
        print(f"  Allocating {size / 1024:.0f} KB pinned buffer...", end=" ")
        start = time.time()
        buf = PinnedBuffer(size)
        elapsed = (time.time() - start) * 1000
        print(f"OK (addr=0x{buf.address:x}, {elapsed:.2f}ms)")
        buf.free()

    print("  PASSED: Pinned memory allocation works")
    return True


def test_buffer_pool():
    """Test buffer pool functionality."""
    print("\n=== Test 2: Buffer Pool ===")

    pool = PinnedBufferPool(1024 * 1024, num_buffers=4)

    # Acquire all buffers
    buffers = []
    for i in range(4):
        buf = pool.acquire()
        buffers.append(buf)
        print(f"  Acquired buffer {i}: 0x{buf.address:x}")

    # Release all
    for buf in buffers:
        pool.release(buf)
    print("  Released all buffers")

    # Re-acquire (should reuse)
    buf = pool.acquire()
    print(f"  Re-acquired buffer: 0x{buf.address:x}")
    pool.release(buf)

    pool.close()
    print("  PASSED: Buffer pool works")
    return True


def test_doca_connection(pci_addr: str):
    """Test DOCA connection to DPU."""
    print(f"\n=== Test 3: DOCA Connection (PCI: {pci_addr}) ===")

    try:
        print(f"  Connecting to DPU at {pci_addr}...", end=" ")
        start = time.time()
        client = DOCAKVOffloadClient(pci_addr)
        elapsed = (time.time() - start) * 1000
        print(f"OK ({elapsed:.2f}ms)")

        print("  PASSED: DOCA connection successful")
        return client
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def test_buffer_registration(client: DOCAKVOffloadClient, size: int = 1024 * 1024):
    """Test buffer registration with DPU."""
    print(f"\n=== Test 4: Buffer Registration ({size / 1024:.0f} KB) ===")

    # Allocate pinned buffer
    buf = PinnedBuffer(size)
    print(f"  Allocated pinned buffer at 0x{buf.address:x}")

    # Register with DOCA
    print("  Registering buffer with DPU...", end=" ")
    start = time.time()
    buffer_id = client.register_buffer(buf.address, size)
    elapsed = (time.time() - start) * 1000
    print(f"OK (buffer_id={buffer_id}, {elapsed:.2f}ms)")

    print("  PASSED: Buffer registration works")
    return buf, buffer_id


def test_dma_transfer(client: DOCAKVOffloadClient, buffer_id: int, size: int):
    """Test DMA transfer to DPU."""
    print(f"\n=== Test 5: DMA Transfer ({size / 1024:.0f} KB) ===")

    # Initiate transfer
    print("  Initiating transfer...", end=" ")
    start = time.time()
    transfer_id = client.transfer(buffer_id, offset=0, length=size)
    initiate_time = (time.time() - start) * 1000
    print(f"OK (transfer_id={transfer_id}, {initiate_time:.2f}ms)")

    # Wait for completion
    print("  Waiting for completion...", end=" ")
    start = time.time()
    client.wait_transfer(transfer_id, timeout_ms=5000)
    wait_time = (time.time() - start) * 1000
    print(f"OK ({wait_time:.2f}ms)")

    # Calculate bandwidth
    total_time_sec = (initiate_time + wait_time) / 1000
    bandwidth_gbps = (size * 8) / (total_time_sec * 1e9)
    print(f"  Transfer bandwidth: {bandwidth_gbps:.2f} Gbps")

    print("  PASSED: DMA transfer works")
    return True


def test_statistics(client: DOCAKVOffloadClient):
    """Test statistics retrieval."""
    print("\n=== Test 6: Statistics ===")

    stats = client.get_stats()
    print(f"  Total transfers: {stats.total_transfers}")
    print(f"  Total bytes: {stats.total_bytes}")
    print(f"  Failed transfers: {stats.failed_transfers}")
    print(f"  Avg latency: {stats.avg_latency_us:.2f} us")
    print(f"  Peak bandwidth: {stats.peak_bandwidth_gbps:.2f} Gbps")

    print("  PASSED: Statistics retrieval works")
    return True


def test_multiple_transfers(client: DOCAKVOffloadClient, num_transfers: int = 10,
                            size: int = 1024 * 1024):
    """Test multiple consecutive transfers."""
    print(f"\n=== Test 7: Multiple Transfers ({num_transfers} x {size / 1024:.0f} KB) ===")

    # Allocate buffer
    buf = PinnedBuffer(size)
    buffer_id = client.register_buffer(buf.address, size)

    latencies = []
    start_total = time.time()

    for i in range(num_transfers):
        start = time.time()
        transfer_id = client.transfer(buffer_id, offset=0, length=size)
        client.wait_transfer(transfer_id, timeout_ms=5000)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        print(f"  Transfer {i + 1}: {latency:.2f}ms")

    total_time = time.time() - start_total
    total_bytes = num_transfers * size
    avg_bandwidth = (total_bytes * 8) / (total_time * 1e9)

    print(f"  Average latency: {sum(latencies) / len(latencies):.2f}ms")
    print(f"  Min latency: {min(latencies):.2f}ms")
    print(f"  Max latency: {max(latencies):.2f}ms")
    print(f"  Aggregate bandwidth: {avg_bandwidth:.2f} Gbps")

    # Cleanup
    client.unregister_buffer(buffer_id)
    buf.free()

    print("  PASSED: Multiple transfers work")
    return True


def run_all_tests(pci_addr: str):
    """Run all tests."""
    print("=" * 60)
    print("DOCA KV Cache Offload Integration Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: CUDA pinned memory
    try:
        if test_cuda_pinned_memory():
            passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 2: Buffer pool
    try:
        if test_buffer_pool():
            passed += 1
    except Exception as e:
        print(f"  FAILED: {e}")
        failed += 1

    # Test 3-7 require DPU connection
    client = None
    try:
        client = test_doca_connection(pci_addr)
        if client:
            passed += 1

            # Test 4: Buffer registration
            buf, buffer_id = test_buffer_registration(client)
            if buf:
                passed += 1

                # Test 5: DMA transfer
                if test_dma_transfer(client, buffer_id, buf.size):
                    passed += 1

                # Cleanup
                client.unregister_buffer(buffer_id)
                buf.free()

            # Test 6: Statistics
            if test_statistics(client):
                passed += 1

            # Test 7: Multiple transfers
            if test_multiple_transfers(client):
                passed += 1
        else:
            failed += 5
            print("\n  Skipping remaining tests (no DPU connection)")

    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1

    finally:
        if client:
            client.close()

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="DOCA KV Offload Integration Tests")
    parser.add_argument(
        "--pci-addr",
        help="PCI address of BlueField DPU (auto-detect if not specified)"
    )
    parser.add_argument(
        "--cuda-only",
        action="store_true",
        help="Only run CUDA tests (no DPU required)"
    )
    args = parser.parse_args()

    # Get PCI address
    pci_addr = args.pci_addr
    if pci_addr is None:
        pci_addr = find_bluefield_pci_address()
        if pci_addr:
            print(f"Auto-detected BlueField at: {pci_addr}")
        else:
            print("WARNING: No BlueField DPU found")
            if not args.cuda_only:
                print("Use --pci-addr to specify manually or --cuda-only for CUDA-only tests")
                sys.exit(1)

    if args.cuda_only:
        # Run only CUDA tests
        test_cuda_pinned_memory()
        test_buffer_pool()
        return

    # Run all tests
    success = run_all_tests(pci_addr)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
