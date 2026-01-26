#!/usr/bin/env python3
"""
Example: Using vLLM with DOCA KV Cache Offloading

This script demonstrates how to use vLLM with the DOCA connector
for KV cache offloading to a BlueField DPU.

Prerequisites:
1. DPU service running: sudo ./build_dpu/dpu_offloader
2. Environment variables set:
   export LD_LIBRARY_PATH=/mnt/nvme_bench/cacheflow/DOCA_Backend/build:$LD_LIBRARY_PATH
   export PYTHONPATH=/mnt/nvme_bench/cacheflow/DOCA_Backend/python:$PYTHONPATH
3. vLLM installed with DOCA connector registered

Usage:
    python example_vllm_with_doca.py --model meta-llama/Llama-2-7b-hf
"""

import sys
from pathlib import Path

# Add DOCA Backend python directory to path for connector imports
_doca_python_dir = str(Path(__file__).parent / "python")
if _doca_python_dir not in sys.path:
    sys.path.append(_doca_python_dir)  # Append, not prepend, to avoid shadowing vLLM

import argparse
import json
import logging
import os
import tempfile
from typing import List

logging.basicConfig(level=logging.DEBUG if os.environ.get('DEBUG') else logging.INFO)
logger = logging.getLogger(__name__)


def create_kv_config(pci_addr: str = None) -> dict:
    """Create KV transfer config dictionary."""
    
    # Auto-detect PCI address if not provided
    if pci_addr is None:
        try:
            from doca_kv_offload import find_bluefield_pci_address
            pci_addr = find_bluefield_pci_address()
            logger.info(f"Auto-detected BlueField DPU at: {pci_addr}")
        except Exception as e:
            logger.warning(f"Could not auto-detect PCI address: {e}")
            pci_addr = "0000:0c:00.0"  # Default
            logger.info(f"Using default PCI address: {pci_addr}")
    
    config = {
        "kv_connector": "DOCAConnectorV1",
        "kv_role": "kv_both",
        "kv_rank": 0,
        "kv_parallel_size": 1,
        "kv_buffer_device": "cuda",
        "kv_buffer_size": 1000000000,  # 1GB
        "doca_pci_addr": pci_addr,
        "doca_block_size": 16777216,   # 16MB
        "doca_max_blocks": 256
    }
    
    logger.info(f"Created KV transfer config: {json.dumps(config, indent=2)}")
    return config


def test_vllm_with_doca(model_name: str, prompts: List[str]):
    """Test vLLM with DOCA offloading."""
    
    logger.info("="*60)
    logger.info("vLLM with DOCA KV Cache Offloading Example")
    logger.info("="*60)
    
    # Step 1: Register connector
    logger.info("\nStep 1: Registering DOCA connector...")
    try:
        from register_connector import register_doca_connector
        if register_doca_connector():
            logger.info("✓ DOCA connector registered")
        else:
            logger.error("✗ Failed to register connector")
            return False
    except Exception as e:
        logger.error(f"✗ Connector registration failed: {e}")
        return False
    
    # Step 2: Create KV config
    logger.info("\nStep 2: Creating KV transfer config...")
    kv_config = create_kv_config()
    kv_config_str = json.dumps(kv_config)  # vLLM expects JSON string, not file
    
    # Step 3: Initialize vLLM with DOCA offloading
    logger.info(f"\nStep 3: Initializing vLLM with model: {model_name}")
    logger.info("This may take a few minutes to download and load the model...")
    
    try:
        logger.debug(f"sys.path: {sys.path}")
        from vllm import LLM, SamplingParams
        
        llm = LLM(
            model=model_name,
            kv_transfer_config=kv_config_str,  # Pass as JSON string
            gpu_memory_utilization=0.8,
            max_model_len=2048,  # Start small for testing
            trust_remote_code=True
        )
        logger.info("✓ vLLM initialized successfully with DOCA offloading")
    except Exception as e:
        logger.error(f"✗ Failed to initialize vLLM: {e}")
        return False
    
    # Step 4: Generate responses
    logger.info("\nStep 4: Generating responses...")
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100
    )
    
    logger.info(f"\nPrompts ({len(prompts)}):")
    for i, prompt in enumerate(prompts):
        logger.info(f"  {i+1}. {prompt}")
    
    try:
        logger.info("\nGenerating (KV cache will be offloaded to DPU)...")
        outputs = llm.generate(prompts, sampling_params)
        
        logger.info("\n" + "="*60)
        logger.info("Results:")
        logger.info("="*60)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated = output.outputs[0].text
            logger.info(f"\nPrompt {i+1}: {prompt}")
            logger.info(f"Generated: {generated}")
            logger.info("-"*60)
        
        logger.info("\n✓ Generation completed successfully!")
        logger.info("Check DPU terminal for offload activity logs")
        
    except Exception as e:
        logger.error(f"✗ Generation failed: {e}", exc_info=True)
        return False
    
    logger.info("\n" + "="*60)
    logger.info("Example completed successfully!")
    logger.info("="*60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM with DOCA KV cache offloading"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",  # Small model for quick testing
        help="HuggingFace model name (default: facebook/opt-125m for quick testing)"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "The future of AI is",
            "Once upon a time",
        ],
        help="List of prompts to generate from"
    )
    parser.add_argument(
        "--pci-addr",
        type=str,
        default=None,
        help="BlueField DPU PCI address (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    # Check environment
    logger.info("Checking environment...")
    if "PYTHONPATH" not in os.environ:
        logger.warning("PYTHONPATH may not be set correctly")
        logger.warning("Run: export PYTHONPATH=/mnt/nvme_bench/cacheflow/DOCA_Backend/python:$PYTHONPATH")
    
    if "LD_LIBRARY_PATH" not in os.environ or "build" not in os.environ["LD_LIBRARY_PATH"]:
        logger.warning("LD_LIBRARY_PATH may not be set correctly")
        logger.warning("Run: export LD_LIBRARY_PATH=/mnt/nvme_bench/cacheflow/DOCA_Backend/build:$LD_LIBRARY_PATH")
    
    # Run test
    success = test_vllm_with_doca(args.model, args.prompts)
    
    if success:
        logger.info("\n🎉 Success! vLLM is working with DOCA offloading!")
        logger.info("\nNext steps:")
        logger.info("1. Try a larger model: --model meta-llama/Llama-2-7b-hf")
        logger.info("2. Test multi-turn conversations for prefix matching benefits")
        logger.info("3. Monitor DPU logs to see transfer activity")
        return 0
    else:
        logger.error("\n❌ Test failed. Please check:")
        logger.error("1. DPU service is running: ssh dpu 'ps aux | grep dpu_offloader'")
        logger.error("2. Integration tests pass: python3 test_integration.py")
        logger.error("3. Environment variables are set correctly")
        return 1


if __name__ == "__main__":
    exit(main())
