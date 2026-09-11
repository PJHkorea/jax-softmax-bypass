import jax
import jax.numpy as jnp

class LocalHomeostaticRectifier:
    """
    [P0 - 공동 최우선 과제] 레이어 정규화(Norm) 전역 락 거세 커널 (LLaMA & Gemma 통합형)
    소프트맥스가 탈피된 평면에서 유입되는 극단적인 데이터 스케일 변동을
    전역 감축(Global Reduction Sync Lock) 없이 국소 대수 변환 및 3차 왜도 소산 기전으로 정류하며,
    동시에 LLaMA(표준) 및 Gemma(오프셋 스케일)의 가중치를 분기 없이 하이재킹합니다.
    """
    def __init__(self, head_dim: int, alpha: float = 0.05, casimir_delta: float = 1e-4):
        self.head_dim = head_dim
        self.alpha = alpha  # 왜도 소산 감쇄 계수
        self.casimir_delta = casimir_delta  # 진공 고사 방지 하한 가드
        
    def __call__(self, x: jnp.ndarray, gamma: jnp.ndarray = None, use_gemma_offset: bool = False) -> jnp.ndarray:
        """
        Input x shape: [Batch, NumHeads, SeqLen, HeadDim] 
        또는 표준 LLM 레이어 입력 형태인 [Batch, SeqLen, Dim] 전체를 완벽 수용.
        
        Args:
            x: 입력 데이터 스트림
            gamma: 원본 체크포인트에서 로드된 고유 정규화 가중치 텐서 [HeadDim] 혹은 [Dim]
            use_gemma_offset: True일 경우 Gemma 스타일(1.0 + gamma), False일 경우 LLaMA 스타일(gamma) 강제 적용
        """
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

        # ==================== LLaMA / Gemma 범용 하이재킹 도킹 레일 ====================
        # 가중치 주입이 없을 경우(독립 테스트/PoC 모드) 연산 오버헤드 없이 즉시 탈출
        if gamma is None:
            return x_rectified

        # 물리적 하드웨어 MUX 레벨로 스케일 팩터 오프셋 평탄화
        # use_gemma_offset 조건에 따라 부호 분기(Branch) 없이 1.0 + gamma 평면과 gamma 평면을 원자적으로 전환
        scale_factor = jnp.where(use_gemma_offset, 1.0 + gamma, gamma)
        
        # 브로드캐스팅(Broadcasting) 수착: 
        # x_rectified 차원 구조가 [B, H, L, D]일 때 gamma가 [D] 또는 [1, 1, 1, D] 형태여도 정합되도록 뷰 유지
        return x_rectified * scale_factor

# XLA 컴파일러 전용 PyTree 정적 등록
def _rectifier_flatten(obj):
    return (), (obj.head_dim, obj.alpha, obj.casimir_delta)

def _rectifier_unflatten(aux_data, children):
    return LocalHomeostaticRectifier(*aux_data)

jax.tree_util.register_pytree_node(LocalHomeostaticRectifier, _rectifier_flatten, _rectifier_unflatten)
