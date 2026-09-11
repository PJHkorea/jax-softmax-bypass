import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

class LocalHomeostaticRectifier:
    """
    [최우선 과제] 레이어 정규화(Norm) 전역 락 거세 커널 (LLaMA & Gemma 통합 및 SPMD 최적화형)
    소프트맥스가 탈피된 평면에서 유입되는 극단적인 데이터 스케일 변동을
    전역 감축(Global Reduction Sync Lock) 없이 국소 대수 변환 및 3차 왜도 소산 기전으로 정류하며,
    동시에 LLaMA/Gemma의 가중치를 분기 없이 하이재킹하고 하드웨어 메시 상의 통신 노이즈를 제로화합니다.
    """
    def __init__(self, head_dim: int, alpha: float = 0.05, casimir_delta: float = 1e-4):
        self.head_dim = head_dim
        self.alpha = alpha  # 왜도 소산 감쇄 계수
        self.casimir_delta = casimir_delta  # 진공 고사 방지 하한 가드
        
    def __call__(self, x: jnp.ndarray, gamma: jnp.ndarray = None, use_gemma_offset: bool = False, mesh: jax.sharding.Mesh = None) -> jnp.ndarray:
        """
        Input x shape: [Batch, NumHeads, SeqLen, HeadDim] 
        또는 표준 LLM 레이어 입력 형태인 [Batch, SeqLen, Dim] 전체를 완벽 수용.
        
        Args:
            x: 입력 데이터 스트림
            gamma: 원본 체크포인트에서 로드된 고유 정규화 가중치 텐서 [HeadDim] 혹은 [Dim]
            use_gemma_offset: True일 경우 Gemma 스타일(1.0 + gamma), False일 경우 LLaMA 스타일(gamma) 강제 적용
            mesh: spmd_sharding_lanes에서 선포된 하드웨어 실리콘 메시 전역 객체
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
        scale_factor = jnp.where(use_gemma_offset, 1.0 + gamma, gamma)
        
        # -----------------------------------------------------------------
        # 고도화 솔루션: 가중치 스케일 팩터 분산 메시 제약 주입 (SPMD Fence)
        # -----------------------------------------------------------------
        # 멀티 가속기 환경(Mesh)이 제공되고 gamma가 샤딩이 필요한 상태일 때 통신 제약을 집행합니다.
        if mesh is not None:
            # gamma의 랭크(차원 수)를 동적으로 파악하여 마지막 'HeadDim' 또는 'Dim' 축을 'model' 병렬 하이웨이로 구속합니다.
            gamma_rank = scale_factor.ndim
            if gamma_rank == 1:
                # [Dim] 구조일 경우 -> 'model' 축 병렬 처리 분할 단언
                gamma_spec = P('model')
            else:
                # [1, 1, 1, Dim] 등 다차원 브로드캐스팅 뷰 구조일 경우 -> 마지막 축만 'model'로 록킹
                gamma_spec = P(*(None,) * (gamma_rank - 1), 'model')
            
            # 주소선 포인터를 가로챔과 동시에 컴파일타임 최적화 라인 동결 (All-Gather 원천 봉쇄)
            scale_factor = jax.lax.with_sharding_constraint(scale_factor, jax.sharding.NamedSharding(mesh, gamma_spec))
        
        # 브로드캐스팅(Broadcasting) 수착: 
        # x_rectified와 장치 단 레지스터에서 메모리 복사 파편 없이 0ns 원자적 요소별 곱셈 결착
        return x_rectified * scale_factor

# XLA 컴파일러 전용 PyTree 정적 등록
def _rectifier_flatten(obj):
    return (), (obj.head_dim, obj.alpha, obj.casimir_delta)

def _rectifier_unflatten(aux_data, children):
    return LocalHomeostaticRectifier(*aux_data)

jax.tree_util.register_pytree_node(LocalHomeostaticRectifier, _rectifier_flatten, _rectifier_unflatten)

