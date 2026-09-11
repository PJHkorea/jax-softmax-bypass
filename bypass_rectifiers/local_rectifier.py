import jax
import jax.numpy as jnp

class LocalHomeostaticRectifier:
    """
    [P0 - 공동 최우선 과제] 레이어 정규화(Norm) 전역 락 거세 커널
    소프트맥스가 탈피된 평면에서 유입되는 극단적인 데이터 스케일 변동을
    전역 감축(Global Reduction Sync Lock) 없이 국소 대수 변환 및 3차 왜도 소산 기전으로 정류합니다.
    """
    def __init__(self, head_dim: int, alpha: float = 0.05, casimir_delta: float = 1e-4):
        self.head_dim = head_dim
        self.alpha = alpha  # 왜도 소산 감쇄 계수
        self.casimir_delta = casimir_delta  # 진공 고사 방지 하한 가드
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Input x shape: [Batch, NumHeads, SeqLen, HeadDim]"""
        # 제약 1: 하방/상방 폭발 방화벽 (Branchless Clipping MUX)
        x_bounded = jnp.maximum(x, -50.0)
        x_safe = jnp.minimum(x_bounded, 50.0)

        # 제약 2: 국소 에너지 대칭 패리티 투영 (HBM 버스 접근 무력화)
        local_energy = jax.lax.square(x_safe)
        safe_energy_vessel = jnp.maximum(local_energy, self.casimir_delta)
        
        # 부호근 역수(rsqrt) 프리미티브로 단 1클록 만에 인라인 정규화
        recip_local_scale = jax.lax.rsqrt(safe_energy_vessel)
        x_normalized = x_safe * recip_local_scale

        # 제약 3: 3차 국소 왜도 소산 제동 (수치적 점성 소산 모사)
        local_skewness = jax.lax.integer_pow(x_normalized, 3)
        x_rectified = x_normalized - (self.alpha * local_skewness)

        return x_rectified

# XLA 컴파일러 전용 PyTree 정적 등록
def _rectifier_flatten(obj):
    return (), (obj.head_dim, obj.alpha, obj.casimir_delta)

def _rectifier_unflatten(aux_data, children):
    return LocalHomeostaticRectifier(*aux_data)

jax.tree_util.register_pytree_node(LocalHomeostaticRectifier, _rectifier_flatten, _rectifier_unflatten)
