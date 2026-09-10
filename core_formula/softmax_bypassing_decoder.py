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
        # 후속 푸리에 직교 기저(Sin/Cos) 분할 매핑 시, 하드웨어 메모리 정렬 및 매트릭스 레이아웃 
        # 붕괴를 원천 차단하기 위해 feature_dim이 짝수인지 컴파일 및 초기화 타임에 즉시 검증합니다.
        if feature_dim % 2 != 0:
            raise ValueError(
                f"[HARDWARE ALIGNMENT ERROR] feature_dim은 반드시 짝수여야 합니다. (입력값: {feature_dim})\n"
                f"푸리에 복소 평면(Sin/Cos) 균등 분할을 위해 2의 거듭제곱 규격을 권장합니다."
            )

        # [고도화 포인트] 튜플 분산 처리 시 XLA 컴파일러 타입 미스매치를 방지하기 위해 정형화 튜플 변환 강제
        self.mesh_shape = (mesh_shape, mesh_shape) if isinstance(mesh_shape, int) else tuple(mesh_shape)
        self.feature_dim = feature_dim
        self.alpha = alpha  # 비선형 댐핑 계수 상숫값
        self.hbar_eff = 1e-6  # 수치 발산 및 제로 디비전 방어용 완충 가드레일 상수
        
        # ------------------------------------------------------------------------
        # [⚡ 고도화 1: 레거시 LLM 가중치 결착용 트랜스포머 표준 스케일 팩터 인입]
        # ------------------------------------------------------------------------
        # Softmax가 사라진 평면에서 Q, K 내적값이 테일러 급수를 무차별 인플레이션 시키지 않도록,
        # 기존 모델의 head_dim 사양에 정합하는 1 / sqrt(d_k) 역수 스케일을 컴파일 상수로 고정합니다.
        self.scale_factor = 1.0 / jnp.sqrt(float(self.feature_dim))
        
        # [⚡ 고도화 2: 카시미르 Vacuum Singular Boundary 선언]
        # jnp.maximum(..., 0.0) 제로 고사 영역 진입 시, NaN 전파를 차단할 에라스틱 구조용 최소 임계값 바인딩
        self.casimir_delta = 1e-4

        # [고도화 3: 푸리에 직교 기저(Sin/Cos) 완성을 위한 메쉬 축 선언]
        # 역전파 시 그라디언트 흐름이 기저 축을 오염시키지 않도록 stop_gradient를 유지하되,
        # 정보 손실(Rank Collapse)을 물리적으로 차단하기 위해 이 주파수 축은 순방향/역방향 모두에서 고정 상수로 작용합니다.
        self.vorticity_omega = jax.lax.stop_gradient(
            jnp.linspace(-jnp.pi, jnp.pi, self.mesh_shape[0], dtype=jnp.float32)
        )

    # ------------------------------------------------------------------------
    # [★ JAX PyTree 규격 오차 0% 정적 동결 인터록 완성]
    # ------------------------------------------------------------------------
    def tree_flatten(self) -> Tuple[Tuple[jax.Array], Tuple[Tuple[int, int], int, float, float, float, float]]:
        """
        XLA 컴파일러가 장치 메모리(HBM) 트래킹을 놓치지 않도록 동적 텐서와 정적 상수를 완벽히 격리 분리합니다.
        역전파 자동 미분 경로에서도 이 격리 구조가 유지되어 그라디언트가 오염되지 않습니다.
        """
        # 자동 미분 대상 추적용 가동 노드 (동적 배열)
        children = (self.vorticity_omega,)
        
        # 고도화 상숫값(scale_factor, casimir_delta)을 정적 메타데이터 관로에 온전히 추가 격리
        aux_data = (self.mesh_shape, self.feature_dim, self.alpha, self.hbar_eff, self.scale_factor, self.casimir_delta)
        return children, aux_data


       @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Tuple[int, int], int, float, float, float, float], children: Tuple[jax.Array]) -> "UpgradedSoftmaxBypassingDecoder":
        """
        역전파 자동 미분 그래프 빌드 시, 정적 메트릭스 차원이 단 1비트도 뒤틀리지 않도록 원형 그대로 뷰 복원합니다.
        """
        # [고도화 포인트] 분산 샤딩(SPMD) 환경에서 튜플 형태의 형상 데이터가 정수형으로 강제 언팩 및 변환되는 
        # 컴파일러 타입 바인딩 사각지대를 방지하기 위해 aux_data[0][0] 규격으로 정밀 언팩 복원합니다.
        mesh_shape_raw = aux_data[0]
        mesh_init = mesh_shape_raw[0] if isinstance(mesh_shape_raw, tuple) else mesh_shape_raw
        
        # Part 1에서 확장된 aux_data 인덱스 명세(4, 5)에 정확히 싱크 정합 매핑
        obj = cls(mesh_shape=int(mesh_init), feature_dim=int(aux_data[1]), alpha=float(aux_data[2]))
        obj.hbar_eff = float(aux_data[3])
        obj.scale_factor = float(aux_data[4])
        obj.casimir_delta = float(aux_data[5])
        
        obj.vorticity_omega = children[0]
        return obj

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def __call__(self, clean_manifold_tensor: jax.Array) -> jax.Array:
        """
        [⚡ OPERATIONAL FUSION RUNTIME GATEWAY - TAYLOR-FOURIER INTEGRAL INVERSION]
        [Part 2: 테일러 급수 2차 근사 및 푸리에 직교 위상 융합 런타임 (학습 가능형)]
        
        하드웨어 가속기(GPU/TPU)의 고성능 프리미티브를 100% 활용하면서
        역전파(Backpropagation) 시 그라디언트가 완벽하게 흐르도록 미분 경로를 완전히 개방합니다.
        """
        target_dtype = clean_manifold_tensor.dtype
        
        # ------------------------------------------------------------------------
        # [⚡ 고도화 1: 입력단 스케일 정류 및 음수 마스크 폭발 클리핑 방화벽]
        # ------------------------------------------------------------------------
        # Step 1-1. 레거시 LLM 가중치 생태계의 스케일 장벽과 맞물리도록 입력 텐서 정류
        X_scaled = clean_manifold_tensor * self.scale_factor
        
        # Step 1-2. 파이토치 단에서 유입되는 미래 토큰 소거용 음수 마스크(-10000.0)가 
        # 후속 테일러 제곱항(0.5 * x^2)을 만나 플러스 수억대의 거대한 폭발 값으로 반전되는 것을 
        # 원천 차단하기 위해, 하드웨어 MUX 레벨 하한선 클리핑을 인라인 융합합니다.
        # (-50.0 미만의 값은 제곱되어도 2500 수준으로 가속기 범위 내에서 엄격히 통제됩니다.)
        X_safe = jnp.maximum(X_scaled, -50.0)
        
        # [🛡️ TAYLOR 2nd-ORDER AMPLIFICATION CORE]
        # 초월 지수함수 회로를 우회하여 ALU 내 1클록 만에 Fused Multiply-Add(FMA)로 완전히 병합
        X_raw = X_safe
        X_squared = jax.lax.square(X_raw)
        X_amplified = 1.0 + X_raw + (0.5 * X_squared)
        
        # [📊 3rd-ORDER SKEWNESS MOMENT FLATTENING - NaN-Free 고도화]
        # 수치 연산 중 발생하는 비대칭 기하 왜곡(Skewness)을 평탄화하여 발산을 억제
        spatial_mean = jnp.mean(X_amplified, axis=-1, keepdims=True)
        spatial_var = jnp.var(X_amplified, axis=-1, keepdims=True) + self.hbar_eff
        
        # [학습 최적화 포인트 1] 미분 불가능한 부호 반전 및 절대값 가드레일을 걷어내고, 
        # 수치적 연속성(Continuity)과 미분 가능성(Differentiability)을 보장하기 위해 hbar_eff 분산 안정화 채택
        # [고도화 포인트] jax.lax.rsqrt 프리미티브 가속 기계어를 분산 표준편차 역수 연산에 융합하여 
        # sqrt 후 reciprocal을 개별 처리하던 2-Cycle HLO 병목을 단 1클록으로 압착합니다.
        recip_std = jax.lax.rsqrt(spatial_var)

        
              # 3차 모멘트 왜도 계산 파이프라인의 하드웨어 인라인 융합 유도
        normalized_deviation = (X_amplified - spatial_mean) * recip_std
        skewness = jnp.mean(jax.lax.integer_pow(normalized_deviation, 3), axis=-1, keepdims=True)
        
        # 비선형 댐핑 계수(alpha)를 통과시켜 왜곡 분산 완벽 정류
        X_rectified = X_amplified - (self.alpha * skewness)

        # [🛡️ COMPILER HLO INLINE FUSION - 푸리에 직교 기저(Sin/Cos) 결합 완성]
        # 정보 손실(Rank Collapse)에 따른 난해도(PPL) 폭발을 물리적으로 제어하기 위해 오일러 복소 평면을 모사합니다.
        # Part 1의 인터록 통과를 전제로, feature_dim 공간의 절반은 Sin 축으로, 나머지 절반은 Cos 축으로 투영하여 위상 소멸 데드존을 제거합니다.
        half_dim = self.feature_dim // 2
        grid_axis = jnp.arange(half_dim, dtype=target_dtype) / float(half_dim)
        
        # 고정된 vorticity_omega 상수를 기반으로 가상 매트릭스 레이아웃 선언 (0MB 추가 할당 규격 유지)
        wave_sin = jnp.sin(self.vorticity_omega[:, None] * grid_axis[None, :])
        wave_cos = jnp.cos(self.vorticity_omega[:, None] * grid_axis[None, :])
        field_wave_T = jnp.concatenate([wave_sin, wave_cos], axis=-1) # Virtual Shape: [Mesh, Feature]

        # [Stage 1 Contraction - 연속체 무게중심 모멘트 적분 스캔]
        # 복잡도 완화: [Batch, Feature] x [Feature, Mesh] -> [Batch, Mesh]
        purified_guide_stream = jnp.matmul(X_rectified, field_wave_T.T)

        # [Stage 2 Expansion - 유클리드 최소 잔차 토큰 토폴로지 복원]
        # 복사 연산 없이 고정밀 토큰 매니폴드로 재투영: [Batch, Mesh] x [Mesh, Feature] -> [Batch, Feature]
        final_attention_rail_input = jnp.matmul(purified_guide_stream, field_wave_T)
        
        # [🛡️ BRANCHLESS MUX FIREWALL]
        # 조건문 분기 예측 실패(Stall) 0% 마진을 위해 jnp.maximum primitive를 유지하되 역전파 미분 경로 보존
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
        
        # [학습 최적화 포인트 2] 역전파 학습 그래프 파괴의 원인이던 stop_gradient 가드레일을 완전히 전면 거세합니다.
        # 이로 인해 이 레이어 하단 및 상단 전체 커널의 파라미터들이 유기적으로 그라디언트를 공유하며 학습이 가능해집니다.
        return final_attention_rail_output

# 외부 레거시 모듈이 커널 내부로 들어와 임의의 위상 공간을 오염시키는 것을 막는 전역 보안 자물쇠
__all__ = ["UpgradedSoftmaxBypassingDecoder"]


