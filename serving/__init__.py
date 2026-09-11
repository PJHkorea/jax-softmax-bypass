# -*- coding: utf-8 -*-
"""
Homeostasis Spatial Bus - Advanced Serving Infrastructure Layer
File: serving/__init__.py
"""

from .vllm_hotplug_entrypoint import initialize_wave_serving_engine
from .cluster_bootstrap import auto_bootstrap_hardware_mesh
from .kv_vessel_manager import WaveKVCacheContainer

# 외부 서빙 오케스트레이터 및 FastAPI 엔드포인트단으로 사상할 청정 인터페이스 명세
__all__ = [
    "initialize_wave_serving_engine",
    "auto_bootstrap_hardware_mesh",
    "WaveKVCacheContainer"
]
