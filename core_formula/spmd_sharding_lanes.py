"""
JAX XLA SPMD Sharding Orchestrator
File: core_formula/spmd_sharding_lanes.py
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

def establish_global_hardware_sharding_lanes(num_data_replicas: int = 4, num_model_partitions: int = 8):
    """
    [GLOBAL HARDWARE MESH MATRIX SETUP]
    Scans active physical accelerator clusters to allocate a 2D topology matrix layout, 
    isolating device resources into distinct data-parallel lanes and model-parallel axis parameters.
    """
    # 1. Intercepts all available physical accelerator (GPU/TPU) sockets from the current runtime environment.
    devices = jax.devices()
    total_devices = len(devices)
    required_devices = num_data_replicas * num_model_partitions
    
    if total_devices < required_devices:
        raise RuntimeError(
            f"[SHARDING CLUSTER ERROR] Insufficient hardware accelerator assets! (Required: {required_devices}, Found: {total_devices})\n"
            f"Please verify the physical node topology interconnection tracks immediately."
        )
        
    # 2. Partitions the bare-metal device grid array to match designated distributed execution layouts.
    hardware_grid = mesh_utils.create_device_mesh((num_data_replicas, num_model_partitions))
    global_mesh = Mesh(hardware_grid, ('data', 'model'))
    
    print(f"[CLUSTER] SPMD Silicon Sharding Constitution Enforced | Topology: {num_data_replicas}x{num_model_partitions}")
    return global_mesh

def apply_wave_attention_sharding_rules(global_mesh: Mesh, q: jax.Array, k: jax.Array, v: jax.Array) -> tuple:
    """
    [MULTI-DIMENSIONAL TENSOR SHARDING INJECTION - Rank-Aware Track Hardening]
    Dynamically tracks the input manifold tensor rank (3D vs 4D) to enforce precise parallel partition 
    specifications, completely preventing bit-level layout fragmentation or redundant All-Gather communication overheads.
    """
    # ------------------------------------------------------------------------
    # [INFRA INTERLOCK: DYNAMIC RANK RESOLUTION BASED PARALLEL SPECIFICATION MAPPING]
    # ------------------------------------------------------------------------
    stream_rank = q.ndim
    
    if stream_rank == 4:
        # Layout track specification for standard 4D attention rails: [Batch, NumHeads, SeqLen, HeadDim]
        # Shards the Batch dimension across 'data' parallel grids, while isolating 'NumHeads' over 'model' 
        # parallel arrays to stream vectors directly into high-speed accelerator SRAM structures.
        attention_input_spec = P('data', 'model', None, None)
    elif stream_rank == 3:
        # Layout track specification for early hijacking 3D attention rails: [Batch, SeqLen, EmbedDim]
        # Distributes the Batch dimension across 'data' tracks while splitting and locking the total 'EmbedDim' dimension over 'model' slots.
        attention_input_spec = P('data', None, 'model')
    else:
        # Universal fallback guardrail to strictly lock the trailing hidden channel dimension and Batch tracking axes 
        # even if dynamic array mutation states arise within the network paths.
        attention_input_spec = P('data', *(None,) * (stream_rank - 2), 'model')
    
    # Synthesizes the global XLA compiler optimization blueprint.
    sharding_rule = NamedSharding(global_mesh, attention_input_spec)
    
    # ------------------------------------------------------------------------
    # [COMPILER FENCE: FORCE DRIVEN STATIC SHARDING CONSTRAINT CONVOLUTION]
    # ------------------------------------------------------------------------
    # Intercepts memory pointer addresses and simultaneously injects hardware sharding constraints 
    # to permanently freeze the compiler optimization compilation paths, preemptively killing multi-node NCCL communication noise.
    q_sharded = jax.lax.with_sharding_constraint(q, sharding_rule)
    k_sharded = jax.lax.with_sharding_constraint(k, sharding_rule)
    v_sharded = jax.lax.with_sharding_constraint(v, sharding_rule)
    
    return q_sharded, k_sharded, v_sharded


def verify_context_vessel_sharding_coherence(global_mesh: Mesh, context_vessel: jax.Array) -> jax.Array:
    """
    [O(1) CONTEXT VESSEL MEMORY LOCK - Variable Rank Layout Hardening]
    Tracks the mutable array topology of the wave-contracted global context vessel, physically hard-locking 
    its device memory placement to guarantee the container never boundaries or experiences redundant copy lag across node ranks.
    """
    # ------------------------------------------------------------------------
    # [INFRA INTERLOCK: VESSEL RANK RECOGNITION BASED SHARD SPECIFICATION BINDING]
    # ------------------------------------------------------------------------
    vessel_rank = context_vessel.ndim
    
    if vessel_rank == 4:
        # Layout track specification for standard 4D wave containers: [Batch, NumHeads, MeshShape, HeadDim]
        # Asserts that the Batch distribution ('data') and head isolation ('model') retain absolute alignment continuity.
        vessel_spec = P('data', 'model', None, None)
    elif vessel_rank == 3:
        # Layout track specification for compressed 3D variant wave manifolds: [Batch, MeshShape, Dim]
        vessel_spec = P('data', None, 'model')
    else:
        # Safeguard fallback to rigidly enforce Batch sharding and trailing channel parallel splitting 
        # even if the layout view is altered during deep autoregressive inference iterations.
        vessel_spec = P('data', *(None,) * (vessel_rank - 2), 'model')
        
    vessel_sharding = NamedSharding(global_mesh, vessel_spec)

    
    # ------------------------------------------------------------------------
    # [COMPILER FENCE: HARD HLO SHARDING CONSTRAINT FOR GRADIENT PARITY CONSERVATION]
    # ------------------------------------------------------------------------
    # Blocks latency bottlenecks caused by unwanted tensor layout mutations (Resharding) 
    # or memory fragment allocation spikes within multi-node pools before the context_vessel 
    # interfaces with the Q stream to decode Euclidean token topologies inside the core engine.
    # Avoids runtime operations like jax.device_put; instead, deploys a compile-time static 
    # constraint fence to rigidly freeze the O(1) wave space layout structure.
    vessel_locked = jax.lax.with_sharding_constraint(context_vessel, vessel_sharding)
    
    return vessel_locked
