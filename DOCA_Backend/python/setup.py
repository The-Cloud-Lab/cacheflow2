"""
Setup script for DOCA KV Cache Offload Python package.
"""

from setuptools import setup

setup(
    name="doca-kv-offload",
    version="0.1.0",
    description="KV Cache Offloading to NVIDIA BlueField DPU for vLLM",
    author="DOCA Backend Team",
    py_modules=[
        'doca_connector',
        'kv_offload_manager',
        'doca_kv_offload',
        'doca_cuda_utils',
        'register_connector',
    ],
    install_requires=[
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
