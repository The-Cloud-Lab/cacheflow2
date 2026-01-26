"""
Example: Using vLLM with DOCA KV Cache Offloading

This demonstrates how to use vLLM with KV cache offloading to a BlueField DPU.
"""

import logging
import sys

# Register the DOCA connector before importing vLLM
from register_connector import register_doca_connector
register_doca_connector()

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run vLLM with DOCA offloading."""
    
    # Configuration
    model_name = "meta-llama/Llama-2-7b-hf"
    dpu_pci_addr = "0000:03:00.0"  # Adjust to your DPU's PCI address
    
    logger.info(f"Initializing vLLM with DOCA offloading to {dpu_pci_addr}")
    
    # Configure KV transfer with DOCA connector
    kv_transfer_config = KVTransferConfig(
        kv_connector="DOCAConnectorV1",
        kv_role="kv_both",  # Both send and receive
        extra_config={
            "dpu_pci_addr": dpu_pci_addr,
            "block_size": 16 * 1024 * 1024,  # 16MB per block
            "max_blocks": 256,
            "tokens_per_block": 1024,
            "async_transfers": True,
        }
    )
    
    # Initialize vLLM with DOCA offloading
    llm = LLM(
        model=model_name,
        kv_transfer_config=kv_transfer_config,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
    )
    
    logger.info("vLLM initialized successfully")
    
    # Create test prompts with shared prefix
    shared_prefix = "Once upon a time in a land far away, " * 50
    prompts = [
        shared_prefix + "there lived a brave knight.",
        shared_prefix + "there was a magical kingdom.",
        shared_prefix + "there dwelt a wise wizard.",
    ]
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
    )
    
    logger.info("Running first batch (will cache to DPU)...")
    outputs = llm.generate(prompts[:1], sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        logger.info(f"Prompt: {prompt[:50]}...")
        logger.info(f"Generated: {generated}")
    
    logger.info("\nRunning second batch (should load from DPU cache)...")
    outputs = llm.generate(prompts[1:], sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated = output.outputs[0].text
        logger.info(f"Prompt: {prompt[:50]}...")
        logger.info(f"Generated: {generated}")
    
    logger.info("\n✓ Example completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
