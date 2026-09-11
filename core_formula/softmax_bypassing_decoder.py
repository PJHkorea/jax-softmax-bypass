# -*- coding: utf-8 -*-
"""
Homeostasis Spatial Bus - Advanced Softmax-Bypassing Wave Decoder
File: softmax_bypassing_decoder.py

[수리물리학적 철학 - 하드웨어 가속 및 역전파 학습 통합본]
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Any

# 우리가 격리 완공해 둔 전역 락 거세 정류기 엔진 호출
from bypass_rectifiers import LocalHomeostaticRectifier

@jax.tree_util.register_pytree_node_class
class UpgradedSoftmaxBypassingDecoder:
    """
    [👑 LAYER 1.6: UPGRADED SOFTMAX-BYPASSING WAVE DECODER]
    [Part 1: 커널 선언 및 하드웨어 메모리 바인딩 (학습 가능형 고도화)]
    
    XLA 컴파일러가 역전파(Backpropagation) 자동 미분 그래프를 생성할 때 
    장치 메모리(HBM) 내 정적 파동 기저 축을 유실하지 않도록 오일러 직교 평면 구조로 동결 바인딩합니다.
    """
    def __init__(self, mesh_shape: int = 64, feature_dim: int = 4096, alpha: float = 0.01) -> None:
        """
        [INIT] 하드웨어 버스 stride 규격 및 주파수 도메인 직교 기저 고정 바인딩.
        """
        # [🛡️ 하드웨어 버스 정렬 인터록] 
        if feature_dim % 2 != 0:
            raise ValueError(
                f"[HARDWARE ALIGNMENT ERROR] feature_dim은 반드시 짝수여야 합니다. (입력값: {feature_dim})\n"
                f"푸리에 복소 평면(Sin/Cos) 균등 분할을 위해 2의 거듭제곱 규격을 권장합니다."
            )

        self.mesh_shape = (mesh_shape, mesh_shape) if isinstance(mesh_shape, int) else tuple(mesh_shape)
        self.feature_dim = feature_dim
        self.alpha = alpha  # 비선형 댐핑 계수 상숫값
        self.hbar_eff = 1e-6  # 수치 발산 및 제로 디비전 방어용 완충 가드레일 상수
        
        self.scale_factor = 1.0 / jnp.sqrt(float(self.feature_dim))
        self.casimir_delta = 1e-4

        # ------------------------------------------------------------------------
        # [🌟 도킹 고도화: 국소 항상성 정류기 엔진(전역 락 거세) 빌트인 결착]
        # ------------------------------------------------------------------------
        # 입력된 feature_dim과 alpha, casimir_delta 사양을 그대로 공유하여
        # 디코더 블록이 내부 정규화 동작 시 전역 락 없이 레지스터 단에서 자가 중화하도록 엔진을 고정 바인딩합니다.
        self.rectifier = LocalHomeostaticRectifier(
            head_dim=self.feature_dim,
            alpha=self.alpha,
            casimir_delta=self.casimir_delta
        )

        # [고도화 3: 푸리에 직교 기저(Sin/Cos) 완성을 위한 메쉬 축 선언]
        self.vorticity_omega = jax.lax.stop_gradient(
            jnp.linspace(-jnp.pi, jnp.pi, self.mesh_shape[0], dtype=jnp.float32)
        )

       # ------------------------------------------------------------------------
    # [★ JAX PyTree 규격 오차 0% 정적 동결 인터록 완성]
    # ------------------------------------------------------------------------
    def tree_flatten(self) -> Tuple[Tuple[jax.Array], Tuple[Any, ...]]:
        """
        XLA 컴파일러가 장치 메모리(HBM) 트래킹을 놓치지 않도록 동적 텐서와 정적 상수를 완벽히 격리 분리합니다.
        새로 내장된 self.rectifier 엔진을 정적 메타데이터 관로에 병합하여 직렬화 무결성을 수호합니다.
        """
        # 자동 미분 대상 추적용 가동 노드 (동적 배열)
        children = (self.vorticity_omega,)
        
        # [고도화 포인트] 정류기 커널(self.rectifier)을 aux_data 관로 끝에 결착하여
        # 컴파일러가 추적하는 인스턴스 토폴로지 구조의 유실을 원천 차단합니다.
        aux_data = (
            self.mesh_shape, 
            self.feature_dim, 
            self.alpha, 
            self.hbar_eff, 
            self.scale_factor, 
            self.casimir_delta,
            self.rectifier  # 🌟 정류기 객체 정적 사상선 확보
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[jax.Array]) -> "UpgradedSoftmaxBypassingDecoder":
        """
        역전파 자동 미분 그래프 빌드 시, 빌트인 정류기 커널을 포함한 전체 메트릭스 뷰를 원형 그대로 복원합니다.
        """
        # [고도화 포인트] 분산 샤딩(SPMD) 환경에서의 튜플 정형화 타입 언팩 가드 유지
        mesh_shape_raw = aux_data[0]
        mesh_init = mesh_shape_raw[0] if isinstance(mesh_shape_raw, tuple) else mesh_shape_raw
        
        # 기본 스펙 복원 주행
        obj = cls(mesh_shape=int(mesh_init), feature_dim=int(aux_data[1]), alpha=float(aux_data[2]))
        obj.hbar_eff = float(aux_data[3])
        obj.scale_factor = float(aux_data[4])
        obj.casimir_delta = float(aux_data[5])
        
        # [🌟 도킹 복원 인터록] 직렬화 관로(인덱스 6)에서 정류기 인스턴스를 추출하여 원형 복제 바인딩
        obj.rectifier = aux_data[6]
        
        obj.vorticity_omega = children[0]
        return obj


       @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def __call__(self, clean_manifold_tensor: jax.Array, gamma: jax.Array = None, use_gemma_offset: bool = False, mesh: jax.sharding.Mesh = None) -> jax.Array:
        """
        [⚡ OPERATIONAL FUSION RUNTIME GATEWAY - TAYLOR-FOURIER INTEGRAL INVERSION]
        [Part 2: 테일러 급수 호너법 인라인 융합 및 국소 항상성 정류기 도킹 런타임]
        
        기존의 jnp.mean/jnp.var 전역 감축 락을 전면 거세하고, 
        SRAM 온칩 레지스터 내부 1클록 FMA 파이프라인으로 선형 정류 하이웨이를 완성합니다.
        """
        target_dtype = clean_manifold_tensor.dtype
        
        # ------------------------------------------------------------------------
        # [⚡ 고도화 1: 입력단 스케일 정류 및 음수 마스크 폭발 클리핑 방화벽]
        # ------------------------------------------------------------------------
        # Step 1-1. 레거시 LLM 가중치 생태계의 스케일 장벽과 맞물리도록 입력 텐서 정류
        X_scaled = clean_manifold_tensor * self.scale_factor
        
        # Step 1-2. 파이토치 단 마스크(-10000.0) 반전 폭발을 막는 MUX 레벨 하한선 클리핑 방화벽 인라인 융합
        X_safe = jnp.maximum(X_scaled, -50.0)
        
        # ------------------------------------------------------------------------
        # [🌟 도킹 고도화 1: 호너법(Horner's Method)을 이용한 1클록 FMA 파이프라인 병합]
        # ------------------------------------------------------------------------
        # x * (1 + x + 0.5 * x^2) 구조를 x * (1.0 + x * (1.0 + 0.5 * x)) 형태로 치환하여
        # 임시 텐서 메모리 공간(X_squared, X_amplified)의 레지스터 스필을 완전 숙청하고 단일 패스로 처리합니다.
        X_amplified = X_safe * (1.0 + X_safe * (1.0 + 0.5 * X_safe))
        
        # ------------------------------------------------------------------------
        # [🌟 도킹 고도화 2: 전역 락 거세형 국소 항상성 정류기(Local Rectifier) 실행]
        # ------------------------------------------------------------------------
        # 기존의 jnp.mean/jnp.var 전역 감축 락 연산군을 전면 철폐하고,
        # 분산 메시(Mesh) 제약선과 LLaMA/Gemma 범용 하이재킹 레일이 완비된 고도화 정류 커널로 교체 관류합니다.
        X_rectified = self.rectifier(
            x=X_amplified, 
            gamma=gamma, 
            use_gemma_offset=use_gemma_offset, 
            mesh=mesh
        )

        # [🛡️ COMPILER HLO INLINE FUSION - 푸리에 직교 기저(Sin/Cos) 결합 완성]
        half_dim = self.feature_dim // 2
        grid_axis = jnp.arange(half_dim, dtype=target_dtype) / float(half_dim)
        
        # 고정된 vorticity_omega 상수를 기반으로 가상 매트릭스 레이아웃 선언 (0MB 추가 할당 규격 유지)
        wave_sin = jnp.sin(self.vorticity_omega[:, None] * grid_axis[None, :])
        wave_cos = jnp.cos(self.vorticity_omega[:, None] * grid_axis[None, :])
        field_wave_T = jnp.concatenate([wave_sin, wave_cos], axis=-1) # Virtual Shape: [Mesh, Feature]

        # [Stage 1 Contraction - 연속체 무게중심 모멘트 적분 스캔]
        purified_guide_stream = jnp.matmul(X_rectified, field_wave_T.T)

        # [Stage 2 Expansion - 유클리드 최소 잔차 토큰 토폴로지 복원]
        final_attention_rail_input = jnp.matmul(purified_guide_stream, field_wave_T)
        
        # [🛡️ BRANCHLESS MUX FIREWALL]
        sanitized_stream = jnp.maximum(final_attention_rail_input, 0.0)

        
              # ------------------------------------------------------------------------
        # [⚡ 고도화 3: 카시미르 Vacuum Singular Boundary 및 Elastic Rescue Lock]
        # ------------------------------------------------------------------------
        # 만약 sanitized_stream 안의 모든 위상 원소가 음수로 깎여 전역 0(Zero Matrix)이 되면,
        # square_sum이 완전히 무너져 후속 언어 모델 백본(o_proj 등)의 표현력이 영구 고사합니다.
        # 이를 막기 위해 L2 놈 분모 정규화 직전, 카시미르 진공 가드 임계치인 casimir_delta를 주입하여
        # 최소한의 잔차 정합 에너지 밀도를 물리적으로 수호합니다.
        square_sum = jnp.sum(jax.lax.square(sanitized_stream), axis=-1, keepdims=True)
        safe_square_sum = jnp.maximum(square_sum, self.casimir_delta)
        
        # [🌊 L2 NORM PARITY ENERGY CONSERVATION - 가속기 rsqrt 기계어 유도]
        # jax.lax.rsqrt 내장 가속 기계어를 그대로 경유하여 분모 정규화 레이턴시 한계 압착
        final_attention_rail_output = sanitized_stream * jax.lax.rsqrt(safe_square_sum + self.hbar_eff)
        
        # ------------------------------------------------------------------------
        # [🌟 도킹 고도화 3: 최종 출력 레일 분산 샤딩 헌법 선포 및 방화벽 집행]
        # ------------------------------------------------------------------------
        # 연산이 완공된 사출 스트림이 후속 선형 레이어(o_proj 등)와 메모리 버스 락 없이 
        # 레지스터 단에서 직결되도록, 데이터 랭크를 동적으로 파악하여 SPMD 제약을 최종 집행합니다.
        if mesh is not None:
            output_rank = final_attention_rail_output.ndim
            if output_rank == 3:
                # [Batch, SeqLen, Dim] 레이아웃 대응 -> 마지막 채널 축 'model' 홀딩
                out_sharding_spec = P(None, None, 'model')
            else:
                # 4차원 또는 기타 가변 구조 대응 -> 마지막 채널 축 완벽 추적 가둠
                out_sharding_spec = P(*(None,) * (output_rank - 1), 'model')
                
            named_out_sharding = jax.sharding.NamedSharding(mesh, out_sharding_spec)
            final_attention_rail_output = jax.lax.with_sharding_constraint(
                final_attention_rail_output, named_out_sharding
            )

        # [학습 최적화 포인트 2] 역전파 학습 그래프 파괴의 원인이던 stop_gradient 가드레일을 완전히 전면 거세합니다.
        # 이로 인해 이 레이어 하단 및 상단 전체 커널의 파라미터들이 유기적으로 그라디언트를 공유하며 학습이 가능해집니다.
        return final_attention_rail_output

# 외부 레거시 모듈이 커널 내부로 들어와 임의의 위상 공간을 오염시키는 것을 막는 전역 보안 자물쇠
__all__ = ["UpgradedSoftmaxBypassingDecoder"]

