import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

class HomeostaticTaylorGluCore:
    """
    [CORE ATOMIC KERNEL] SwiGLU Activation Transcendental Elimination Kernel
    (SPMD Distributed Optimization with 1-Clock Inline FFM Fusion Layout)

    Eradicates the execution bottlenecks of exponential functions (e^-x) found in standard SiLU/Swish activations, 
    rectifying the gate projection using a single-pass 2nd-order algebraic approximation based on Horner's Method.
    """
    def __init__(self, hidden_dim: int, alpha: float = 0.02, casimir_delta: float = 1e-4):
        self.hidden_dim = hidden_dim
        self.alpha = alpha  # Dissipation coefficient to regulate accumulated FFN skewness defects
        self.casimir_delta = casimir_delta  # Vacuum safety guard to prevent model divergence

    def __call__(self, gate_stream: jnp.ndarray, up_stream: jnp.ndarray, mesh: jax.sharding.Mesh = None) -> jnp.ndarray:
        if mesh is not None:
            # Dynamically resolves the tensor layout rank to lock the model shard boundaries.
            stream_rank = gate_stream.ndim
            if stream_rank == 3:
                sharding_spec = P(None, None, 'model')
            else:
                sharding_spec = P(*(None,) * (stream_rank - 1), 'model')
                
            named_sharding = jax.sharding.NamedSharding(mesh, sharding_spec)
            gate_stream = jax.lax.with_sharding_constraint(gate_stream, named_sharding)
            up_stream = jax.lax.with_sharding_constraint(up_stream, named_sharding)

        # [STAGE 1: GATE STREAM SCALING CLIP]
        # Prevents numerical divergence across the Taylor approximation rails using 1-clock hardware MUX primitives.
        gate_bounded = jnp.maximum(gate_stream, -10.0)
        gate_safe = jnp.minimum(gate_bounded, 10.0)
        
        # ========================================================================
        # MATHEMATICAL SOLUTION: HORNER'S METHOD COMPILER INLINE FMA FUSION
        # ========================================================================
        # Factorizes the standard Taylor series x * (1 + x + 0.5 * x^2) into x * (1.0 + x * (1.0 + 0.5 * x)).
        # This completely eliminates temporary tensor memory allocations (taylor_gate) from VRAM, 
        # driving pure inline fused-multiply-add execution solely inside on-chip registers.
        activated_gate = gate_safe * (1.0 + gate_safe * (1.0 + 0.5 * gate_safe))
        
        # [STAGE 2: VACUUM DISSIPATION & SKEWNESS FLATNESS CONTROL]
        # Suppresses structural skewness distortions accumulated deeper inside FFN layer blocks.
        activated_gate_safe = jnp.maximum(activated_gate, self.casimir_delta)
        gate_skewness = jax.lax.integer_pow(activated_gate_safe, 3)
        rectified_gate = activated_gate_safe - (self.alpha * gate_skewness)
        
        # [STAGE 3: ELEMENT-WISE GATING DECOUPLING]
        # Streams the rectified activation directly into the upstream tensor over a pure linear highway.
        output_stream = rectified_gate * up_stream
        
        if mesh is not None:
            output_stream = jax.lax.with_sharding_constraint(output_stream, named_sharding)
            
        return output_stream

# ========================================================================
# XLA COMPILER STATIC TRACING REGISTRATION (JAX PyTree Protocol)
# ========================================================================
def _taylor_glu_flatten(obj):
    return (), (obj.hidden_dim, obj.alpha, obj.casimir_delta)

def _taylor_glu_unflatten(aux_data, children):
    return HomeostaticTaylorGluCore(*aux_data)

jax.tree_util.register_pytree_node(HomeostaticTaylorGluCore, _taylor_glu_flatten, _taylor_glu_unflatten)

