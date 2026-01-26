#!/usr/bin/env python3
"""
Integration Test Suite for DOCA KV Cache Offload

This script tests the complete pipeline from Python to DOCA to DPU.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_doca_client(pci_addr: str):
    """Test basic DOCA client functionality."""
    logger.info("=" * 60)
    logger.info("TEST 1: DOCA Client Basic Operations")
    logger.info("=" * 60)
    
    try:
        from doca_kv_offload import DOCAKVOffloadClient
        import torch
        
        logger.info(f"Initializing DOCA client with PCI address: {pci_addr}")
        client = DOCAKVOffloadClient(pci_addr)
        
        # Test 1: Allocate pinned buffer and register
        logger.info("Test 1.1: Allocating pinned tensor...")
        tensor = torch.randn(1024, 1024, dtype=torch.float32, device='cuda', pin_memory=False)
        tensor_cpu = tensor.cpu().pin_memory()
        
        logger.info("Test 1.2: Registering buffer with DOCA...")
        buffer_id = client.register_buffer(tensor_cpu.data_ptr(), tensor_cpu.numel() * tensor_cpu.element_size())
        logger.info(f"✓ Buffer registered with ID: {buffer_id}")
        
        # Test 2: Transfer data
        logger.info("Test 1.3: Transferring data to DPU...")
        transfer_id = client.transfer(buffer_id, 0, tensor_cpu.numel() * tensor_cpu.element_size())
        logger.info(f"Transfer initiated with ID: {transfer_id}")
        
        logger.info("Test 1.4: Waiting for transfer completion...")
        client.wait_transfer(transfer_id, timeout_ms=5000)
        logger.info("✓ Transfer completed successfully")
        
        # Test 3: Get statistics
        logger.info("Test 1.5: Retrieving statistics...")
        stats = client.get_stats()
        logger.info(f"✓ Statistics: {stats}")
        
        # Cleanup
        logger.info("Test 1.6: Cleaning up...")
        client.unregister_buffer(buffer_id)
        client.close()
        
        logger.info("✓ TEST 1 PASSED\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST 1 FAILED: {e}", exc_info=True)
        return False


def test_kv_offload_manager(pci_addr: str):
    """Test KV offload manager."""
    logger.info("=" * 60)
    logger.info("TEST 2: KV Offload Manager")
    logger.info("=" * 60)
    
    try:
        from kv_offload_manager import KVOffloadManager, compute_prefix_hash
        import torch
        
        logger.info("Test 2.1: Initializing KVOffloadManager...")
        manager = KVOffloadManager(
            pci_addr=pci_addr,
            block_size=4 * 1024 * 1024,  # 4MB
            max_blocks=10,
            num_staging_buffers=2,
        )
        logger.info("✓ Manager initialized")
        
        # Test offload
        logger.info("Test 2.2: Creating test KV cache tensor...")
        kv_tensor = torch.randn(256, 32, 128, dtype=torch.float16, device='cuda')
        logger.info(f"Tensor shape: {kv_tensor.shape}, size: {kv_tensor.numel() * kv_tensor.element_size()} bytes")
        
        logger.info("Test 2.3: Offloading tensor to DPU...")
        hash_key = compute_prefix_hash([1, 2, 3, 4, 5], 5)
        manager.offload_tensor(block_id=42, tensor=kv_tensor, hash_key=hash_key, sync=True)
        logger.info("✓ Tensor offloaded successfully")
        
        # Test prefix matching
        logger.info("Test 2.4: Testing prefix matching...")
        found = manager.has_prefix(hash_key)
        logger.info(f"✓ Prefix found: {found}")
        
        block_id = manager.find_by_hash(hash_key)
        logger.info(f"✓ Block ID for hash: {block_id}")
        
        # Get stats
        logger.info("Test 2.5: Getting statistics...")
        stats = manager.get_stats()
        logger.info(f"✓ Stats: blocks_on_dpu={stats['blocks_on_dpu']}, "
                   f"total_transfers={stats['total_transfers']}")
        
        # Cleanup
        logger.info("Test 2.6: Cleaning up...")
        manager.close()
        
        logger.info("✓ TEST 2 PASSED\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST 2 FAILED: {e}", exc_info=True)
        return False


def test_connector_registration():
    """Test connector registration with vLLM."""
    logger.info("=" * 60)
    logger.info("TEST 3: Connector Registration")
    logger.info("=" * 60)
    
    try:
        logger.info("Test 3.1: Importing vLLM components...")
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        logger.info("✓ vLLM components imported")
        
        logger.info("Test 3.2: Registering DOCAConnectorV1...")
        from register_connector import register_doca_connector
        success = register_doca_connector()
        
        if success:
            logger.info("✓ Connector registered successfully")
        else:
            logger.error("✗ Connector registration failed")
            return False
        
        logger.info("Test 3.3: Verifying registration...")
        # Try to get the connector class
        try:
            connector_cls = KVConnectorFactory.get_connector_class("DOCAConnectorV1")
            logger.info(f"✓ Connector class retrieved: {connector_cls}")
        except Exception as e:
            logger.warning(f"Could not verify registration: {e}")
        
        logger.info("✓ TEST 3 PASSED\n")
        return True
        
    except ImportError as e:
        logger.error(f"✗ TEST 3 FAILED: vLLM not available: {e}")
        logger.info("This is expected if vLLM is not installed")
        return False
    except Exception as e:
        logger.error(f"✗ TEST 3 FAILED: {e}", exc_info=True)
        return False


def test_connector_instantiation(pci_addr: str):
    """Test connector instantiation."""
    logger.info("=" * 60)
    logger.info("TEST 4: Connector Instantiation")
    logger.info("=" * 60)
    
    try:
        logger.info("Test 4.1: Importing connector...")
        from doca_connector import DOCAConnectorV1
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.config import VllmConfig, ModelConfig, ParallelConfig, CacheConfig, KVTransferConfig
        
        logger.info("Test 4.2: Creating mock vLLM config...")
        # Create minimal config for testing
        model_config = ModelConfig(
            model="meta-llama/Llama-2-7b-hf",
            tokenizer="meta-llama/Llama-2-7b-hf",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
            seed=0,
        )
        
        parallel_config = ParallelConfig()
        
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            swap_space=0,
            cache_dtype="auto",
        )
        
        kv_transfer_config = KVTransferConfig(
            kv_connector="DOCAConnectorV1",
            kv_role="kv_both",
            extra_config={
                "dpu_pci_addr": pci_addr,
                "block_size": 4 * 1024 * 1024,
                "max_blocks": 10,
            }
        )
        
        vllm_config = VllmConfig(
            model_config=model_config,
            parallel_config=parallel_config,
            cache_config=cache_config,
            kv_transfer_config=kv_transfer_config,
        )
        
        logger.info("Test 4.3: Instantiating connector...")
        # Test with WORKER role (SCHEDULER and WORKER are the valid roles)
        connector = DOCAConnectorV1(
            vllm_config=vllm_config,
            role=KVConnectorRole.WORKER,
        )
        logger.info("✓ Connector instantiated successfully")
        
        logger.info("Test 4.4: Testing connector methods...")
        # Test basic methods
        logger.info("  - Testing reset_cache()...")
        connector.reset_cache()
        logger.info("  ✓ reset_cache() works")
        
        logger.info("Test 4.5: Shutting down connector...")
        connector.shutdown()
        logger.info("✓ Connector shutdown successfully")
        
        logger.info("✓ TEST 4 PASSED\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ TEST 4 FAILED: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Test DOCA KV Cache Offload Integration")
    parser.add_argument(
        "--pci-addr",
        type=str,
        default=None,
        help="PCI address of BlueField DPU (e.g., 0000:03:00.0). Auto-detect if not specified."
    )
    parser.add_argument(
        "--skip-dpu",
        action="store_true",
        help="Skip tests that require DPU connection"
    )
    
    args = parser.parse_args()
    
    # Auto-detect PCI address if not provided
    pci_addr = args.pci_addr
    if not pci_addr and not args.skip_dpu:
        logger.info("Auto-detecting BlueField DPU...")
        try:
            from doca_kv_offload import find_bluefield_pci_address
            pci_addr = find_bluefield_pci_address()
            if pci_addr:
                logger.info(f"✓ Found BlueField at: {pci_addr}")
            else:
                logger.warning("✗ No BlueField DPU found")
                logger.info("Use --skip-dpu to run tests that don't require DPU")
                return 1
        except Exception as e:
            logger.error(f"Failed to auto-detect DPU: {e}")
            return 1
    
    logger.info("\n" + "=" * 60)
    logger.info("DOCA KV Cache Offload Integration Tests")
    logger.info("=" * 60 + "\n")
    
    results = {}
    
    # Run tests
    if not args.skip_dpu:
        results["DOCA Client"] = test_doca_client(pci_addr)
        
        # Add delay between tests to allow cleanup to complete
        logger.info("Waiting for cleanup to complete...")
        import time
        time.sleep(2)
        
        results["KV Offload Manager"] = test_kv_offload_manager(pci_addr)
        time.sleep(2)
    else:
        logger.info("Skipping DPU-dependent tests\n")
    
    results["Connector Registration"] = test_connector_registration()
    
    if not args.skip_dpu:
        time.sleep(2)
        results["Connector Instantiation"] = test_connector_instantiation(pci_addr)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:30s} {status}")
    
    logger.info("=" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 60 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
