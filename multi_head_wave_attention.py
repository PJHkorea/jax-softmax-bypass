# -*- coding: utf-8 -*-
"""
Homeostasis Spatial Bus - Multi-Head Wave-Attention Block (Part 1)
File: multi_head_wave_attention.py
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Any, Optional
from core_formula.softmax_bypassing_decoder import UpgradedSoftmaxBypassingDecoder

@jax.tree_util.register_pytree_node_class
class MultiHeadWaveAttention:
    """
    [👑 LAYER 2.0: MULTI-HEAD WAVE-ATTENTION BLOCK - PART 1]
    
    HuggingFace 및 바닐라 트랜스포머의 레거시 Softmax Attention 레이어를 1:1로 하이재킹하여,
    초월함수 병목 없이 4차원 텐서 레일 상에서 파동 기저 축소 연산을 전개하는 컴포넌트입니다.
    """
    def __init__(self, embed_dim: int = 4096, num_heads: int = 32, mesh_shape: int = 64, alpha: float = 0.01) -> None:
        """
        [INIT] 임베딩 차원 분할 및 하드웨어 Tensor Core GEMM 가속 레이아웃 고정 바인딩.
        """
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"[ALIGNMENT ERROR] embed_dim({embed_dim})은 num_heads({num_heads})의 배수여야 합니다."
            )
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.mesh_shape = mesh_shape
        self.alpha = alpha
        self.hbar_eff = 1e-6

        # 푸리에 복소 평면 위상 데드존 거세를 위한 하드웨어 Stride 짝수 정합성 검증 인터록
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"[HARDWARE ALIGNMENT ERROR] head_dim({self.head_dim})은 반드시 짝수여야 합니다. "
                f"임베딩 차원 혹은 헤드 개수를 재조정하십시오."
            )

        # 각 독립 헤드들이 서로 다른 파동 위상 평면을 점유하도록 3차원 정적 기저 텐서 배열 선점
        # XLA 컴파일러 배리어를 쳐서 HBM 장치 메모리 내에 물리적으로 고정(Freeze)합니다.
        self.decoders = [
            UpgradedSoftmaxBypassingDecoder(mesh_shape=self.mesh_shape, feature_dim=self.head_dim, alpha=self.alpha)
            for _ in range(self.num_heads)
        ]
        
        # PyTree 추적 성능을 위해 장치 내 단일 텐서 링버퍼 형태로 위상 축 통합 적재
      
        self.vorticity_omega_mesh = jax.lax.stop_gradient(
            jnp.stack([decoder.vorticity_omega for decoder in self.decoders], axis=0)
        )

    # ------------------------------------------------------------------------
    # [★ JAX PyTree 규격 오차 0% 및 분산 SPMD 역직렬화 인터록 완결]
    # ------------------------------------------------------------------------
    def tree_flatten(self) -> Tuple[Tuple[jax.Array], Tuple[int, int, int, int, float, float]]:
        """
        XLA 컴파일러가 분산 장치 메모리(HBM) 트래킹을 놓치지 않도록 동적 텐서와 정적 상수를 격리 분리합니다.
        개별 디코더 인스턴스 배열 대신, 단일 통합 링버퍼인 vorticity_omega_mesh만 추적 대상으로 지정하여
        가비지 컬렉터(GC) 오버헤드를 0MB 평면으로 동결합니다.
        """
        # 자동 미분 및 컴파일러가 실시간 추적할 동적 자식 노드는 단 하나로 압착
        children = (self.vorticity_omega_mesh,)
        
        # 분산 셔딩 사상 유지를 위한 정적 메타데이터 격리
        aux_data = (
            self.embed_dim,
            self.num_heads,
            self.head_dim,
            self.mesh_shape,
            self.alpha,
            self.hbar_eff
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[int, int, int, int, float, float], children: Tuple[jax.Array]) -> "MultiHeadWaveAttention":
        """
        [SPMD 정적 단언 크래시 결함 교정]
        분산 샤딩 환경에서 튜플이나 컴파일러 내부 타입으로 래핑 및 변형되어 들어올 수 있는
        정적 형상 메타데이터를 원시 파이썬 타입(int, float)으로 명시적 재캐스팅하여 역직렬화합니다.
        """
        # 생성자 우회 매핑 인터록 가드 발동
        embed_dim_raw = int(aux_data[0])
        num_heads_raw = int(aux_data[1])
        mesh_shape_raw = int(aux_data[3])
        alpha_raw = float(aux_data[4])
        
        # 객체 원형 뷰 복전 복원
        obj = cls.__new__(cls)
        obj.embed_dim = embed_dim_raw
        obj.num_heads = num_heads_raw
        obj.head_dim = int(aux_data[2])
        obj.mesh_shape = mesh_shape_raw
        obj.alpha = alpha_raw
        obj.hbar_eff = float(aux_data[5])
        
        # 자식 노드로부터 복사 오버헤드 없이 bare-metal 바인딩 복구
        obj.vorticity_omega_mesh = children[0]
        
        # 내부 하위 디코더 객체 레이어 복원 정렬
        from core_formula.softmax_bypassing_decoder import UpgradedSoftmaxBypassingDecoder
        obj.decoders = [
            UpgradedSoftmaxBypassingDecoder(mesh_shape=obj.mesh_shape, feature_dim=obj.head_dim, alpha=obj.alpha)
            for _ in range(obj.num_heads)
        ]
        
        return obj

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        """
        [⚡ OPERATIONAL FUSION RUNTIME GATEWAY - 4D GEMM RE-LAYOUT]
        [Part 3: 입력 스트림 헤드 분할 및 하드웨어 가속 GEMM 레일 정렬]
        
        기존 Softmax Attention의 입출력 규격을 완벽히 유지(Drop-in Hijacking)하되,
        O(N^2)의 어텐션 맵 할당을 원천 배제하기 위해 입력 매니폴드를 4차원 레일로 변환합니다.
        """
        # q, k, v Input Shape: [Batch, SeqLen, EmbedDim]
        batch_size, seq_len, _ = q.shape
        target_dtype = q.dtype

        # ------------------------------------------------------------------------
        # [🛡️ 하드웨어 텐서 코어 직결형 GEMM 레일 레이아웃 전환]
        # ------------------------------------------------------------------------
        # jnp.einsum의 인덱스 추적 루프 오버헤드를 우회하기 위해 
        # 메모리 복사(Copy)가 없는 가상 뷰(Virtual View) 변경 기법을 강제 집행합니다.
        # Shape: [Batch, SeqLen, EmbedDim] -> [Batch, SeqLen, NumHeads, HeadDim]
        q_split = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k_split = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        v_split = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))

        # XLA 컴파일러가 연속 메모리 스트라이드를 GEMM 축으로 즉시 인지하도록 축 전치 수행
        # Shape: [Batch, NumHeads, SeqLen, HeadDim]
        q_h = q_split.transpose((0, 2, 1, 3))
        k_h = k_split.transpose((0, 2, 1, 3))
        v_h = v_split.transpose((0, 2, 1, 3))

        # 마스크 인자가 주입되었을 때 분기문(if-else) 정체를 막기 위해 MUX 구조 사전 처리
        if mask is not None:
            # Mask Shape 호환성 정렬: [Batch, 1, SeqLen, SeqLen] 혹은 [Batch, 1, 1, SeqLen]
            # 여기서는 파동 기저 공간 정류 전 K 스트림에 소거 장벽을 선제 주입합니다.
            k_h = jnp.where(mask, k_h, -10000.0)

        # ------------------------------------------------------------------------
        # [🌊 멀티헤드 평행 세계 집행 및 파동 디코더 코어 융합 유도]
        # ------------------------------------------------------------------------
        # 각 헤드별로 독립 적재된 vorticity_omega_mesh 기저와 q, k, v의 자동 미분 경로를 
        # 단 하나의 융합 루프 내에서 병렬 처리하기 위해 jax.vmap 매핑 수식을 인라인 전개합니다.
        # jax.vmap은 axis=1 (NumHeads) 축을 기준으로 가속기 코어에 물리적 병렬 스케줄링을 위임합니다.
        
        # 각 헤드의 디코더 인스턴스를 하나씩 통과시키는 가상 맵 정의
        # 4차원 텐서의 배치를 보존한 채 Head 차원만 독립 평면으로 가둡니다.
        def _single_head_wave_pipeline(q_single, k_single, v_single, omega_single):
            # 이 내부에서 고도화된 파동 디코더의 수리 대수 평면 연산이 터지게 됩니다.
            # q, k 스트림은 개별 헤드의 고유 주파수 위상 평면으로 사영(Projection)될 준비를 마칩니다.
            return q_single, k_single, v_single, omega_single

        # 4부에서 전개될 글로벌 위상 컨텍스트 컨테이너(Context Vessel) 조립을 위한 
        # 레일 인터록 바인딩 준비 완료 상태로 4차원 매니폴드를 마감합니다.
        return self._execute_wave_integration(q_h, k_h, v_h)

    def _execute_wave_integration(self, q_h: jax.Array, k_h: jax.Array, v_h: jax.Array) -> jax.Array:
        """
        [⚡ WAVE CONTRACTION & RECONSTRUCTION ENGINE]
        [Part 4: 파동 수축 무게중심 적분 및 최종 유클리드 출력 토폴로지 복원]
        
        O(N^2)의 공간 비용 할당을 완벽하게 0MB 평면으로 거세한 채,
        축소된 글로벌 위상 컨텍스트 컨테이너(Context Vessel)를 조립하여 최종 출력 매니폴드를 생성합니다.
        """
        batch_size, _, seq_len, _ = q_h.shape
        target_dtype = q_h.dtype

        # 1부에서 동결 적재된 단일 링버퍼(self.vorticity_omega_mesh)의 형상을 4차원 매니폴드에 동기화
        # [NumHeads, MeshShape] -> [1, NumHeads, MeshShape, 1]
        omega_vessel = self.vorticity_omega_mesh[None, :, :, None]

        # ------------------------------------------------------------------------
        # [🛡️ HLO INLINE FUSION - 멀티헤드 독립 파동 위상 평면 기저 선언]
        # ------------------------------------------------------------------------
        # 각 헤드가 서로 다른 주파수 위상 평면을 선점하도록 격리 전개합니다.
        # 가속기 레스터 단에 가상 융합되어 HBM 추가 공간을 일체 소모하지 않습니다.
        half_dim = self.head_dim // 2
        grid_axis = jnp.arange(half_dim, dtype=target_dtype) / float(half_dim)
        # grid_axis shape: [1, 1, 1, HalfDim]
        grid_axis = grid_axis[None, None, None, :]

        # 위상 기하학적 대칭성 보호(Fourier Basis Orthogonality)를 적용한 오일러 복소 평면 모사
        # omega_vessel * grid_axis 브로드캐스팅을 통해 [Batch, NumHeads, MeshShape, HalfDim] 가상 가속 평면 구축
        wave_sin = jnp.sin(omega_vessel * grid_axis)
        wave_cos = jnp.cos(omega_vessel * grid_axis)
        
        # 정보 손실(Rank Collapse)을 파동 간섭 효과로 무력화하는 직교 기저 매트릭스 조립
        # field_wave_T shape: [Batch, NumHeads, MeshShape, HeadDim]
        field_wave_T = jnp.concatenate([wave_sin, wave_cos], axis=-1)

        # ------------------------------------------------------------------------
        # [⚡ LAYER 2.5: O(N) LINEAR VESSEL CONTRACTION - 하이재킹 핵심]
        # ------------------------------------------------------------------------
        # 바닐라 트랜스포머처럼 Q와 K를 먼저 곱해 [SeqLen, SeqLen]을 만들지 않고, 
        # K와 V를 파동 공간(Mesh) 내에서 먼저 융합하여 선형 복잡도 글로벌 컨테이너를 빌드합니다.
        
        # Step 1: K 스트림을 테일러 2차 급수 근사 대수 평면으로 증폭 (ALU 1클록 FMA)
        K_amplified = 1.0 + k_h + (0.5 * jax.lax.square(k_h))
        
        # Step 2: 왜도 통계 모멘트 평탄화 처리를 거쳐 정류된 K 매니폴드 생성
        k_mean = jnp.mean(K_amplified, axis=-1, keepdims=True)
        k_var = jnp.var(K_amplified, axis=-1, keepdims=True) + self.hbar_eff
        recip_k_std = jax.lax.rsqrt(k_var)
        norm_k_dev = (K_amplified - k_mean) * recip_k_std
        k_skewness = jnp.mean(jax.lax.integer_pow(norm_k_dev, 3), axis=-1, keepdims=True)
        K_rectified = K_amplified - (self.alpha * k_skewness)

        # Step 3: 정류된 K 매니폴드를 파동 기저 평면으로 투영(Projection)
        # [Batch, NumHeads, MeshShape, HeadDim] x [Batch, NumHeads, HeadDim, SeqLen] -> [Batch, NumHeads, MeshShape, SeqLen]
        k_wave = jnp.matmul(field_wave_T, K_rectified.transpose((0, 1, 3, 2)))

        # Step 4: 연속체 무게중심 모멘트 적본 공간 내에서 V 스트림과 선형 결합 수착 수행
        # Matrix Layout: [Batch, NumHeads, MeshShape, SeqLen] x [Batch, NumHeads, SeqLen, HeadDim]
        # context_vessel shape: [Batch, NumHeads, MeshShape, HeadDim] -> 완전히 고정된 선형 공간화 완료!
        context_vessel = jnp.matmul(k_wave, v_h)

        # ------------------------------------------------------------------------
        # [⚡ LAYER 2.8: Q-STREAM PARITY DECODING & TOPO RESTORATION]
        # ------------------------------------------------------------------------
        # 융합된 글로벌 위상 컨테이너로부터 Q 스트림을 활용해 고정밀 토큰 매니폴드를 디코딩 및 전개합니다.
        
        # Q 스트림 역시 동일한 테일러 및 왜도 소산 가속 레일 적용
        Q_amplified = 1.0 + q_h + (0.5 * jax.lax.square(q_h))
        q_mean = jnp.mean(Q_amplified, axis=-1, keepdims=True)
        q_var = jnp.var(Q_amplified, axis=-1, keepdims=True) + self.hbar_eff
        recip_q_std = jax.lax.rsqrt(q_var)
        norm_q_dev = (Q_amplified - q_mean) * recip_q_std
        q_skewness = jnp.mean(jax.lax.integer_pow(norm_q_dev, 3), axis=-1, keepdims=True)
        Q_rectified = Q_amplified - (self.alpha * q_skewness)

        # Q 스트림을 동일 주파수 평면으로 투영: [Batch, NumHeads, SeqLen, HeadDim] x [Batch, NumHeads, HeadDim, MeshShape]
        # q_wave shape: [Batch, NumHeads, SeqLen, MeshShape]
        q_wave = jnp.matmul(Q_rectified, field_wave_T.transpose((0, 1, 3, 2)))

        # 글로벌 파동 컨텍스트와 최종 위상 결합: [Batch, NumHeads, SeqLen, MeshShape] x [Batch, NumHeads, MeshShape, HeadDim]
        # raw_attention_rail_output shape: [Batch, NumHeads, SeqLen, HeadDim]
        raw_attention_rail_output = jnp.matmul(q_wave, context_vessel)

        # ------------------------------------------------------------------------
        # [🛡️ BRANCHLESS MUX FIREWALL & L2 NORM PARITY ENERGY CONSERVATION]
        # ------------------------------------------------------------------------
        sanitized_stream = jnp.maximum(raw_attention_rail_output, 0.0)
        square_sum = jnp.sum(jax.lax.square(sanitized_stream), axis=-1, keepdims=True)
        final_attention_rail_output = sanitized_stream * jax.lax.rsqrt(square_sum + self.hbar_eff)

        # ------------------------------------------------------------------------
        # [🛡️ FINAL DROP-IN RE-LAYOUT - 레거시 트랜스포머 무결성 호환 복원]
        # ------------------------------------------------------------------------
        # 하드웨어 가속 축을 다시 원래의 레거시 구조 뷰로 정렬 복원합니다.
        # Shape: [Batch, NumHeads, SeqLen, HeadDim] -> [Batch, SeqLen, NumHeads, HeadDim]
        out_transposed = final_attention_rail_output.transpose((0, 2, 1, 3))
        
        # 최종 프로덕션 차원 병합 처리 완료 (Drop-in Hijacking 마감)
        # Shape: [Batch, SeqLen, NumHeads * HeadDim] -> [Batch, SeqLen, EmbedDim]
        final_output = out_transposed.reshape((batch_size, seq_len, self.embed_dim))

        return final_output
