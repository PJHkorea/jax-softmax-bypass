import jax
import jax.numpy jnp

class HomeostaticTaylorGluCore:
    """
    [P1 - 차선 과제] SwiGLU 활성화 함수 초월연산 숙청 커널
    지수함수(e^-x)의 연산 병목과 FFN 레이어 특유의 통계적 왜도 왜곡을 완전히 도려내고,
    단일 사이클 FMA 기기어로 매핑되는 2차 테일러 급수 평면으로 정류합니다.
    """
    def __init__(self, hidden_dim: int, alpha: float = 0.02, casimir_delta: float = 1e-4):
        self.hidden_dim = hidden_dim
        self.alpha = alpha  # 비대칭 국소 왜도 보정 계수
        self.casimir_delta = casimir_delta  # 게이트 에너지 고사 방지 하한선

    def __call__(self, gate_stream: jnp.ndarray, up_stream: jnp.ndarray) -> jnp.ndarray:
        """
        Input gate_stream shape: [Batch, SeqLen, HiddenDim]
        Input up_stream shape:   [Batch, SeqLen, HiddenDim]
        """
        # 제약 1: 게이트 인풋 스케일 클리핑 방화벽 (1클록 MUX 명령어 매핑)
        gate_bounded = jnp.maximum(gate_stream, -10.0)
        gate_safe = jnp.minimum(gate_bounded, 10.0)

        # 제약 2: 무분기 테일러 2차 대수 평면 변환 (f(x) = x * (1 + x + 0.5x^2))
        taylor_gate = 1.0 + gate_safe + (0.5 * jax.lax.square(gate_safe))
        activated_gate = gate_safe * taylor_gate

        # 제약 3: 진공 가드 및 국소 왜도 평탄화 (FFN 누적 왜곡 차단)
        activated_gate_safe = jnp.maximum(activated_gate, self.casimir_delta)
        gate_skewness = jax.lax.integer_pow(activated_gate_safe, 3)
        rectified_gate = activated_gate_safe - (self.alpha * gate_skewness)

        # 제약 4: 엘리먼트와이즈 게이팅 결착
        output_stream = rectified_gate * up_stream
        return output_stream

# XLA 컴파일러 전용 PyTree 정적 등록
def _taylor_glu_flatten(obj):
    return (), (obj.hidden_dim, obj.alpha, obj.casimir_delta)

def _taylor_glu_unflatten(aux_data, children):
    return HomeostaticTaylorGluCore(*aux_data)

jax.tree_util.register_pytree_node(HomeostaticTaylorGluCore, _taylor_glu_flatten, _taylor_glu_unflatten)
