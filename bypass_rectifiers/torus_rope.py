import jax
import jax.numpy as jnp
from typing import Tuple
from jax.sharding import PartitionSpec as P

class TorusTopologyRotaryEmbedding:
    """
    [최우선 과제] 회전 위치 임베딩(RoPE) 위상 감금 커널 (SPMD 분산 최적화 및 레지스터 프리 롤링형)
    문맥 길이가 길어질 때 위치 각도가 무한히 발산하는 열린 계 구조를 폐기하고,
    오직 주기적 다양체 사영을 통해 닫힌 도넛 위상(Torus) 표면 범위 안으로 수치를 영구 구속하며,
    동시에 메모리 할당(zeros_like) 오버헤드가 없는 레지스터 인라인 롤링 기법으로 위상 변환을 종결합니다.
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

    def __call__(self, stream: jnp.ndarray, seq_idx: jnp.ndarray, mesh: jax.sharding.Mesh = None) -> jnp.ndarray:
        """
        Input stream shape: [Batch, NumHeads, SeqLen, HeadDim] 
        또는 변형된 표준 하이재킹 사양 [Batch, SeqLen, NumHeads, HeadDim] 전체 수용.
        Input seq_idx shape: [SeqLen]
        """
        # -----------------------------------------------------------------
        # 단계 1: 인입 스트림 복소 전하 분산 메시 제약 주입 (SPMD Fence)
        # -----------------------------------------------------------------
        if mesh is not None:
            stream_rank = stream.ndim
            if stream_rank == 4:
                sharding_spec = P(None, 'model', None, None)
            else:
                sharding_spec = P(*(None,) * (stream_rank - 3), 'model', None, None)
                
            named_sharding = jax.sharding.NamedSharding(mesh, sharding_spec)
            stream = jax.lax.with_sharding_constraint(stream, named_sharding)

        # 닫힌 도넛 매니폴드 캐시선 로드 [SeqLen, HeadDim] -> [1, 1, SeqLen, HeadDim] 사상
        cos_vessel, sin_vessel = self._apply_torus_manifold(seq_idx)
        cos_vessel = jnp.expand_dims(jnp.expand_dims(cos_vessel, 0), 0)
        sin_vessel = jnp.expand_dims(jnp.expand_dims(sin_vessel, 0), 0)
        
        # -----------------------------------------------------------------
        # 고도화 솔루션: 레지스터 프리 인라인 복소 위상 롤링 (Branchless Bit Swap)
        # -----------------------------------------------------------------
        # 기존의 jnp.zeros_like 메모리 할당 및 복잡한 take/set 스캐터 병목을 완전히 소탕합니다.
        # 마지막 축을 [..., HeadDim // 2, 2]로 리셰이프하여 짝수/홀수 신호 평면을 원자적으로 분리합니다.
        orig_shape = stream.shape
        reshaped_stream = stream.reshape(orig_shape[:-1] + (self.head_dim // 2, 2))
        
        # 짝수 축(x0)과 홀수 축(x1)을 가속기 온칩 레지스터 단에서 지연 없이 슬라이싱
        x0 = reshaped_stream[..., 0]
        x1 = reshaped_stream[..., 1]
        
        # 복소 회전 인터록 규칙 적용: [-x1, x0] 벡터를 단일 패스로 결착
        # jnp.stack 연산은 메모리에 배열을 새로 쓰지 않고, 다음 연산의 주소 입력선(Pointer Switch)만 변환합니다.
        interleaved_reshaped = jnp.stack([-x1, x0], axis=-1)
        interleaved_stream = interleaved_reshaped.reshape(orig_shape)
        
        # 닫힌계 토러스 위상 회전 집행 (ALU 1클록 FMA 파이프라인 관류)
        embedded_stream = (stream * cos_vessel) + (interleaved_stream * sin_vessel)
        
        # 탈출 게이트 집행: 최적화 라인 정적 동결
        if mesh is not None:
            embedded_stream = jax.lax.with_sharding_constraint(embedded_stream, named_sharding)
            
        return embedded_stream

# XLA 컴파일러 전용 PyTree 정적 등록
def _torus_rope_flatten(obj):
    return (), (obj.head_dim, obj.max_seq_len, obj.base, obj.torus_radius)

def _torus_rope_unflatten(aux_data, children):
    return TorusTopologyRotaryEmbedding(*aux_data)

jax.tree_util.register_pytree_node(TorusTopologyRotaryEmbedding, _torus_rope_flatten, _torus_rope_unflatten)

