"""File: serving/cluster_bootstrap.py"""
import os
import jax
from core_formula.spmd_sharding_lanes import establish_global_hardware_sharding_lanes

def auto_bootstrap_hardware_mesh():
    """
    Scans K8s/Ray/PyTorch environment variables to automatically lock down global distributed hardware guardrails.
    """
    # Tracks distributed environment metrics initialized by orchestrators (Ray, Torchrun, etc.)
    world_size = int(os.environ.get("WORLD_SIZE", len(jax.devices())))
    rank = int(os.environ.get("RANK", 0))
    
    if world_size >= 4:
        # Executes automatic 2D topology grid splitting to match production infrastructure specifications.
        # e.g., 8 GPUs -> Data Parallelism (DP)=2, Tensor/Model Parallelism (TP)=4
        num_model_parts = 4 if world_size >= 4 else world_size
        num_data_repl = world_size // num_model_parts
        global_mesh = establish_global_hardware_sharding_lanes(
            num_data_replicas=num_data_repl, 
            num_model_partitions=num_model_parts
        )
    else:
        # Fallback layout configuration mapped for single-node multi-GPU or local staging runs
        global_mesh = establish_global_hardware_sharding_lanes(1, world_size)
        
    return global_mesh
