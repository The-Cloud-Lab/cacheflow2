# SPDX-License-Identifier: Apache-2.0
"""CacheFlow v3: a network-attached prefix KV cache pool for vLLM.

The pool lives in BlueField-3 Arm DRAM and is reached from the GPU host over
RDMA (RoCE). Two halves:

  * cacheflowv3.server    runs on the BlueField-3 (numpy + libcfrdma only)
  * cacheflowv3.connector the vLLM KV connector (CacheFlowConnectorV3), loaded via
                          kv_connector_module_path="cacheflowv3.connector"

This package root deliberately imports nothing heavy so the server can import
it on the DPU without torch or vLLM installed.
"""

__version__ = "3.0.0"
