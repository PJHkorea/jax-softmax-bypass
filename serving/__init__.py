"""
Advanced Serving Infrastructure Layer
File: serving/__init__.py
"""

from .vllm_hotplug_entrypoint import initialize_wave_serving_engine
from .cluster_bootstrap import auto_bootstrap_hardware_mesh
from .kv_vessel_manager import WaveKVCacheContainer

# Clean public interface specifications mapped for external serving orchestrators and FastAPI router endpoints.
__all__ = [
    "initialize_wave_serving_engine",
    "auto_bootstrap_hardware_mesh",
    "WaveKVCacheContainer"
]
