import jax
import jax.numpy as jnp
from typing import Tuple

class TorusTopologyRotaryEmbedding:
    """
    [P0 - 공동 최우선 과제] 회전 위치 임베딩(RoPE) 위상 감금 커널
    문맥 길이가 길어질 때 위치 각도가 무한히 발산하는 열린 계 구조를 폐기하고,
    오직 주기적 다양체 사영을 통해 닫힌 도넛 위상(Torus) 표면 범위 안으로 수치를 영구 구속합니다.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 131072, base: float = 10000.0, torus_radius: float = 1.0):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.torus_radius = torus_radius
        
        # 기저 주파수 정적 레일화
        self.inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.head_dim, 2)[:(self.head_dim // 2)] / self.head_dim))

    def _apply_torus_manifold(self, seq_idx: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 기하학적 제약: jax.lax.rem(나머지 연산)을 통해 분기 없이 2*pi 주기 내로 각도 감금
        raw_theta = jnp.outer(seq_idx, self.inv_freq)
        torus_theta = jax.lax.rem(raw_theta, 2.0 * jnp.pi)
        
        cos_backbone = jnp.cos(torus_theta) * self.torus_radius
        sin_backbone = jnp.sin(torus_theta) * self.torus_radius
        
        cos_cached = jnp.repeat(cos_backbone, 2, axis=-1)
        sin_cached = jnp.repeat(sin_backbone, 2, axis=-1)
        return cos_cached, sin_cached

    def __call__(self, stream: jnp.ndarray, seq_idx: jnp.ndarray) -> jnp.ndarray:
        """
        Input stream shape: [Batch, NumHeads, SeqLen, HeadDim]
        Input seq_idx shape: [SeqLen]
        """
        cos_vessel, sin_vessel = self._apply_torus_manifold(seq_idx)
        cos_vessel = jnp.expand_dims(jnp.expand_dims(cos_vessel, 0), 0)
        sin_vessel = jnp.expand_dims(jnp.expand_dims(sin_vessel, 0), 0)
        
        # 브랜치리스 복소 회전 인터록 (부호 비트 반전 슬라이싱 기법)
        even_indices = jnp.arange(0, self.head_dim, 2)
        odd_indices = jnp.arange(1, self.head_dim, 2)
        
        stream_even = jnp.take(stream, even_indices, axis=-1)
        stream_odd = jnp.take(stream, odd_indices, axis=-1)
        
        rotated_odd = -stream_odd
        interleaved_stream = jnp.zeros_like(stream)
        interleaved_stream = interleaved_stream.at[..., even_indices].set(rotated_odd)
        interleaved_stream = interleaved_stream.at[..., odd_indices].set(stream_even)
        
        # 닫힌계 토러스 위상 회전 집행
        embedded_stream = (stream * cos_vessel) + (interleaved_stream * sin_vessel)
        return embedded_stream

# XLA 컴파일러 전용 PyTree 정적 등록
def _torus_rope_flatten(obj):
    return (), (obj.head_dim, obj.max_seq_len, obj.base, obj.torus_radius)

def _torus_rope_unflatten(aux_data, children):
    return TorusTopologyRotaryEmbedding(*aux_data)

jax.tree_util.register_pytree_node(TorusTopologyRotaryEmbedding, _torus_rope_flatten, _torus_rope_unflatten)
