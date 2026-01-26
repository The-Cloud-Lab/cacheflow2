"""
Register DOCAConnectorV1 with vLLM's KVConnector factory.

This script should be imported before using vLLM with DOCA offloading.
"""

import logging

logger = logging.getLogger(__name__)

def register_doca_connector():
    """Register DOCAConnectorV1 with vLLM (idempotent - safe to call multiple times)."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        
        # Handle both relative and absolute imports
        try:
            from .doca_connector import DOCAConnectorV1
        except ImportError:
            from doca_connector import DOCAConnectorV1
        
        # Register the connector with the new API signature
        # KVConnectorFactory.register_connector expects: (name, module_path, class_name)
        # Use just "doca_connector" as the module path (it's installed via pip install -e .)
        module_name = "doca_connector"
        class_name = DOCAConnectorV1.__name__
        
        try:
            # Use the 3-argument form
            KVConnectorFactory.register_connector("DOCAConnectorV1", module_name, class_name)
            logger.info("Successfully registered DOCAConnectorV1 with vLLM")
            return True
        except ValueError as e:
            # Check if it's an "already registered" error
            if "already registered" in str(e).lower():
                logger.info("DOCAConnectorV1 is already registered")
                return True
            else:
                raise
        
    except ImportError as e:
        logger.error(f"Failed to import vLLM components: {e}")
        logger.error("Make sure vLLM is installed and in PYTHONPATH")
        return False
    except Exception as e:
        logger.error(f"Failed to register DOCAConnectorV1: {e}")
        return False


# Auto-register when module is imported
if __name__ != "__main__":
    register_doca_connector()


if __name__ == "__main__":
    # Manual registration for testing
    logging.basicConfig(level=logging.INFO)
    success = register_doca_connector()
    if success:
        print("✓ DOCAConnectorV1 registered successfully")
    else:
        print("✗ Failed to register DOCAConnectorV1")
        exit(1)
