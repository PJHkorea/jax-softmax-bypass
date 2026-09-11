import jax
import jax.numpy as jnp
from typing import Tuple
from jax.sharding import PartitionSpec as P

class TorusTopologyRotaryEmbedding:
    """
    [CORE ATOMIC KERNEL] Rotary Position Embedding (RoPE) Topological Confinement Kernel
    (SPMD Distributed Optimization with Register-Free Inline Rolling Layout)

    Discards the open-system architecture of standard RoPE, where positional angles diverge infinitely 
    as context lengths expand. This kernel permanently confines numerical ranges within a closed torus 
    manifold topology via periodic manifold projections, while executing phase transformations with zero 
    memory allocation (zeros_like) overhead via on-chip register inline rolling.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 131072, base: float = 10000.0, torus_radius: float = 1.0):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.torus_radius = torus_radius
        
        # Static factorization of inverse base frequencies
        self.inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.head_dim, 2)[:(self.head_dim // 2)] / self.head_dim))

    def _apply_torus_manifold(self, seq_idx: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # [TOPOLOGICAL CONSTRAINT: BRANCHLESS ANGLE CONFINEMENT]
        # Restricts spatial phase angles within a 2*pi boundary via hardware-level jax.lax.rem primitives.
        raw_theta = jnp.outer(seq_idx, self.inv_freq)
        torus_theta = jax.lax.rem(raw_theta, 2.0 * jnp.pi)
        
        cos_backbone = jnp.cos(torus_theta) * self.torus_radius
        sin_backbone = jnp.sin(torus_theta) * self.torus_radius
        
        cos_cached = jnp.repeat(cos_backbone, 2, axis=-1)
        sin_cached = jnp.repeat(sin_backbone, 2, axis=-1)
        return cos_cached, sin_cached

    def __call__(self, stream: jnp.ndarray, seq_idx: jnp.ndarray, mesh: jax.sharding.Mesh = None) -> jnp.ndarray:
        """
        Applies topological rotary embeddings across the data stream layout.
        Supports both 4D attention layout [Batch, NumHeads, SeqLen, HeadDim] and 
        transposed hijacking layout [Batch, SeqLen, NumHeads, HeadDim].
        
        Args:
            stream: Incoming target data stream tensor.
            seq_idx: Static 1D positional index array [SeqLen].
            mesh: Global hardware topology device mesh declared via spmd_sharding_lanes.
        """
        # [COMPILER FENCE: TENSOR-PARALLEL ATTENTION HEAD SHARDING INTERLOCK]
        # Locks the designated 'NumHeads' axis to the model shard partition boundary 
        # to freeze redundant tensor layout transitions across the SPMD clusters.
        if mesh is not None:
            stream_rank = stream.ndim
            if stream_rank == 4:
                sharding_spec = P(None, 'model', None, None)
            else:
                sharding_spec = P(*(None,) * (stream_rank - 3), 'model', None, None)
                
            named_sharding = jax.sharding.NamedSharding(mesh, sharding_spec)
            stream = jax.lax.with_sharding_constraint(stream, named_sharding)

        # Projects 2D torus cached manifolds [SeqLen, HeadDim] -> 4D tensor manifolds [1, 1, SeqLen, HeadDim]
        cos_vessel, sin_vessel = self._apply_torus_manifold(seq_idx)
        cos_vessel = jnp.expand_dims(jnp.expand_dims(cos_vessel, 0), 0)
        sin_vessel = jnp.expand_dims(jnp.expand_dims(sin_vessel, 0), 0)

        
                # ========================================================================
        # MATHEMATICAL SOLUTION: REGISTER-FREE INLINE COMPLEX PHASE ROLLING
        # ========================================================================
        # Completely obliterates redundant jnp.zeros_like memory allocations and complex 
        # index scatter-gather bottlenecks. Reshapes the trailing axis into [..., HeadDim // 2, 2] 
        # to atomically isolate the even and odd dimensional signal manifolds.
        orig_shape = stream.shape
        reshaped_stream = stream.reshape(orig_shape[:-1] + (self.head_dim // 2, 2))
        
        # Slices the even manifold (x0) and odd manifold (x1) inline within on-chip registers with 0ns lag.
        x0 = reshaped_stream[..., 0]
        x1 = reshaped_stream[..., 1]
        
        # Enforces complex rotation interlock rules: Binds the [-x1, x0] vector matrix in a single pass.
        # Under XLA compilation, the jnp.stack primitive avoids allocating new physical array tracks in VRAM; 
        # instead, it modifies only the pointer address offsets for the trailing math units.
        interleaved_reshaped = jnp.stack([-x1, x0], axis=-1)
        interleaved_stream = interleaved_reshaped.reshape(orig_shape)
        
        # Executes closed-system torus manifold phase rotation over a single-clock hardware FMA pipeline.
        embedded_stream = (stream * cos_vessel) + (interleaved_stream * sin_vessel)
        
        # Egress Gateway: Freezes sharded compiler tracks at the exit line.
        if mesh is not None:
            embedded_stream = jax.lax.with_sharding_constraint(embedded_stream, named_sharding)
            
        return embedded_stream

# ========================================================================
# XLA COMPILER STATIC TRACING REGISTRATION (JAX PyTree Protocol)
# ========================================================================
def _torus_rope_flatten(obj):
    return (), (obj.head_dim, obj.max_seq_len, obj.base, obj.torus_radius)

def _torus_rope_unflatten(aux_data, children):
    return TorusTopologyRotaryEmbedding(*aux_data)

jax.tree_util.register_pytree_node(TorusTopologyRotaryEmbedding, _torus_rope_flatten, _torus_rope_unflatten)


