# -*- coding: utf-8 -*-
"""File: serving/cluster_bootstrap.py"""
import os
import jax
from core_formula.spmd_sharding_lanes import establish_global_hardware_sharding_lanes

def auto_bootstrap_hardware_mesh():
    """K8s/Ray 환경 변수를 스캔하여 전역 분산 하드웨어 가드레일을 자동 록킹합니다."""
    # Ray 또는 PyTorch 분산 환경 변수 추적
    world_size = int(os.environ.get("WORLD_SIZE", len(jax.devices())))
    rank = int(os.environ.get("RANK", 0))
    
    if world_size >= 4:
        # 실전 인프라 규격에 부합하도록 2차원 토폴로지 자동 분할
        # 예: 8개 GPU -> Data Parallel 2, Tensor(Model) Parallel 4
        num_model_parts = 4 if world_size >= 4 else world_size
        num_data_repl = world_size // num_model_parts
        global_mesh = establish_global_hardware_sharding_lanes(
            num_data_replicas=num_data_repl, 
            num_model_partitions=num_model_parts
        )
    else:
        global_mesh = establish_global_hardware_sharding_lanes(1, world_size)
        
    return global_mesh
