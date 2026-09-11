import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

class HomeostaticTaylorGluCore:
    """
    [P1 - 차선 과제] SwiGLU 활성화 함수 초월연산 숙청 커널 (SPMD 분산 최적화형)
    """
    def __init__(self, hidden_dim: int, alpha: float = 0.02, casimir_delta: float = 1e-4):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.casimir_delta = casimir_delta

    def __call__(self, gate_stream: jnp.ndarray, up_stream: jnp.ndarray, mesh: jax.sharding.Mesh = None) -> jnp.ndarray:
        if mesh is not None:
            stream_rank = gate_stream.ndim
            if stream_rank == 3:
                sharding_spec = P(None, None, 'model')
            else:
                sharding_spec = P(*(None,) * (stream_rank - 1), 'model')
                
            named_sharding = jax.sharding.NamedSharding(mesh, sharding_spec)
            gate_stream = jax.lax.with_sharding_constraint(gate_stream, named_sharding)
            up_stream = jax.lax.with_sharding_constraint(up_stream, named_sharding)

        gate_bounded = jnp.maximum(gate_stream, -10.0)
        gate_safe = jnp.minimum(gate_bounded, 10.0)
        taylor_gate = 1.0 + gate_safe + (0.5 * jax.lax.square(gate_safe))
        activated_gate = gate_safe * taylor_gate
        activated_gate_safe = jnp.maximum(activated_gate, self.casimir_delta)
        gate_skewness = jax.lax.integer_pow(activated_gate_safe, 3)
        rectified_gate = activated_gate_safe - (self.alpha * gate_skewness)
        output_stream = rectified_gate * up_stream
        
        if mesh is not None:
            output_stream = jax.lax.with_sharding_constraint(output_stream, named_sharding)
            
        return output_stream

def _taylor_glu_flatten(obj):
    return (), (obj.hidden_dim, obj.alpha, obj.casimir_delta)

def _taylor_glu_unflatten(aux_data, children):
    return HomeostaticTaylorGluCore(*aux_data)

jax.tree_util.register_pytree_node(HomeostaticTaylorGluCore, _taylor_glu_flatten, _taylor_glu_unflatten)
