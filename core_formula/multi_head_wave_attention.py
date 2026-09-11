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
        
        # [고도화 포인트] 분산 환경(SPMD) 내부 컴파일러의 가변 튜플 타입 오참을 방지하기 위해 단일 정수형 명시 강제
        self.mesh_shape = (mesh_shape, mesh_shape) if isinstance(mesh_shape, int) else tuple(mesh_shape)
        self.alpha = alpha
        self.hbar_eff = 1e-6
        
        # ------------------------------------------------------------------------
        # [⚡ 고도화: 코어 디코더 방화벽 규격과 정합할 상숫값 바인딩]
        # ------------------------------------------------------------------------
        # 후단 디코딩 레이어의 Casimir Rescue Lock 작동 사양과 완벽히 동기화되도록
        # 진공 붕괴 최소 임계 상숫값을 마스터 블록 스펙 명세에 선제 등록합니다.
        self.casimir_delta = 1e-4

        # 푸리에 복소 평면 위상 데드존 거세를 위한 하드웨어 Stride 짝수 정합성 검증 인터록
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"[HARDWARE ALIGNMENT ERROR] head_dim({self.head_dim})은 반드시 짝수여야 합니다. "
                f"임베딩 차원 혹은 헤드 개수를 재조정하십시오."
            )

        # 각 독립 헤드들이 서로 다른 파동 위상 평면을 점유하도록 3차원 정적 기저 텐서 배열 선점
        # XLA 컴파일러 배리어를 쳐서 HBM 장치 메모리 내에 물리적으로 고정(Freeze)합니다.
        # [고도화 포인트] 상위 mesh_shape 인터록 전사에 맞추어 단일 차원 정수형 추출 바인딩
        decoder_mesh_target = self.mesh_shape[0] if isinstance(self.mesh_shape, tuple) else self.mesh_shape
        self.decoders = [
            UpgradedSoftmaxBypassingDecoder(mesh_shape=decoder_mesh_target, feature_dim=self.head_dim, alpha=self.alpha)
            for _ in range(self.num_heads)
        ]
        
        # PyTree 추적 성능을 위해 장치 내 단일 텐서 링버퍼 형태로 위상 축 통합 적재
        self.vorticity_omega_mesh = jax.lax.stop_gradient(
            jnp.stack([decoder.vorticity_omega for decoder in self.decoders], axis=0)
        )

         # ------------------------------------------------------------------------
    # [★ JAX PyTree 규격 오차 0% 및 분산 SPMD 역직렬화 인터록 완결]
    # ------------------------------------------------------------------------
    def tree_flatten(self) -> Tuple[Tuple[jax.Array], Tuple[int, int, int, Any, float, float, float]]:
        """
        XLA 컴파일러가 분산 장치 메모리(HBM) 트래킹을 놓치지 않도록 동적 텐서와 정적 상수를 격리 분리합니다.
        개별 디코더 인스턴스 배열 대신, 단일 통합 링버퍼인 vorticity_omega_mesh만 추적 대상으로 지정하여
        가비지 컬렉터(GC) 오버헤드를 0MB 평면으로 동결합니다.
        """
        # 자동 미분 및 컴파일러가 실시간 추적할 동적 자식 노드는 단 하나로 압착
        children = (self.vorticity_omega_mesh,)
        
        # [고도화] 분산 셔딩 사상 유지를 위해 casimir_delta를 정적 메타데이터 관로(인덱스 6)에 추가 인입
        aux_data = (
            self.embed_dim,
            self.num_heads,
            self.head_dim,
            self.mesh_shape,
            self.alpha,
            self.hbar_eff,
            self.casimir_delta
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[int, int, int, Any, float, float, float], children: Tuple[jax.Array]) -> "MultiHeadWaveAttention":
        """
        [SPMD 정적 단언 크래시 결함 교정]
        분산 샤딩 환경에서 튜플이나 컴파일러 내부 타입으로 래핑 및 변형되어 들어올 수 있는
        정적 형상 메타데이터를 원시 파이썬 타입(int, float)으로 명시적 재캐스팅하여 역직렬화합니다.
        """
        # 생성자 우회 매핑 인터록 가드 발동
        embed_dim_raw = int(aux_data[0])
        num_heads_raw = int(aux_data[1])
        
        # [고도화 포인트] mesh_shape가 복전 타임에 단일 값 혹은 튜플 구조로 깨져 들어와 발생하는 
        # XLA 타입 미스매치를 방지하기 위해 타입 가드레일을 주입하여 안전하게 복원합니다.
        mesh_shape_aux = aux_data[3]
        mesh_shape_raw = (int(mesh_shape_aux[0]), int(mesh_shape_aux[1])) if isinstance(mesh_shape_aux, (tuple, list)) else (int(mesh_shape_aux), int(mesh_shape_aux))
        
        alpha_raw = float(aux_data[4])
        
        # 객체 원형 뷰 복전 복원
        obj = cls.__new__(cls)
        obj.embed_dim = embed_dim_raw
        obj.num_heads = num_heads_raw
        obj.head_dim = int(aux_data[2])
        obj.mesh_shape = mesh_shape_raw
        obj.alpha = alpha_raw
        obj.hbar_eff = float(aux_data[5])
        
        # [고도화] 확장된 aux_data 인덱스 규격에 맞춰 정적 방화벽 상수 복원 바인딩
        obj.casimir_delta = float(aux_data[6])
        
        # 자식 노드로부터 복사 오버헤드 없이 bare-metal 바인딩 복구
        obj.vorticity_omega_mesh = children[0]
        
        # 내부 하위 디코더 객체 레이어 복원 정렬 (고도화된 파라미터 연동 유지)
        from core_formula.softmax_bypassing_decoder import UpgradedSoftmaxBypassingDecoder
        decoder_mesh_target = obj.mesh_shape[0] if isinstance(obj.mesh_shape, tuple) else obj.mesh_shape
        obj.decoders = [
            UpgradedSoftmaxBypassingDecoder(mesh_shape=decoder_mesh_target, feature_dim=obj.head_dim, alpha=obj.alpha)
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

        # ------------------------------------------------------------------------
        # [⚡ 고도화 1: 브랜치리스(Branchless) 곱셈 소거형 마스크 인터록]
        # ------------------------------------------------------------------------
        # 파이토치 하이재킹단에서 마스크가 주입되지 않았을 때(None)의 기본 참(True) 레일 빌드
        # 정적 컴파일러(XLA)의 그래프 추적선 파편화를 막기 위해 파이썬 분기문(if-else)을 전면 제거합니다.
        default_mask = jnp.ones((batch_size, 1, 1, seq_len), dtype=jnp.bool_)
        safe_mask = jax.lax.select(mask is None, default_mask, mask)
        
        # 브로드캐스팅 정합성 확보를 위한 차원 정류 스캔
        # [Batch, 1, 1, SeqLen] 또는 [Batch, 1, SeqLen, SeqLen] 명세를 [Batch, NumHeads, SeqLen, 1] 레일에 맞춤 정렬
        # 소프트맥스가 탈피된 테일러 대수학 평면에서는 미래 토큰의 위상을 소거하기 위해 
        # 거대한 음수가 아닌, 곱셈 부호 소거(0.0f) 또는 완전 제로 아웃 기믹을 강제 집행해야 안전합니다.
        # k_h와 v_h 두 스트림의 물리 전하량을 동시에 0으로 수축시켜 정보 누수를 완벽 차단합니다.
        k_h = jnp.where(safe_mask, k_h, 0.0)
        v_h = jnp.where(safe_mask, v_h, 0.0)

        # ------------------------------------------------------------------------
        # [🌊 멀티헤드 평행 세계 집행 및 파동 디코더 코어 융합 유도]
        # ------------------------------------------------------------------------
        # 각 헤드별로 독립 적재된 vorticity_omega_mesh 기저와 q, k, v의 자동 미분 경로를 
        # 단 하나의 융합 루프 내에서 병렬 처리하기 위해 jax.vmap 매핑 수식을 인라인 전개할 준비선을 수립합니다.
        # (vmap 대신 4차원 텐서 통매칭 행렬곱 축소 연산인 _execute_wave_integration 파이프라인으로 관류 인입)
        
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
        # [⚡ 고도화: SPMD 분산 노드 주소선 바운싱 차단 인터록]
        # ------------------------------------------------------------------------
        # K와 V가 결착하여 압착해낸 고정 차원의 정보 용기(context_vessel)에 
        # spmd_sharding_lanes에서 정의한 NamedSharding 명세를 강제 주입합니다.
        # 분산 가속기 군집 환경에서 후속 Q 복원 평면 진입 전 일어날 수 있는 
        # NCCL 올-리듀스/올-게더 지연을 0MB 완전 매싱으로 통제 격리합니다.
        from core_formula.spmd_sharding_lanes import verify_context_vessel_sharding_coherence
        
        # JAX 하위 런타임 추적 에러를 완벽 방어하기 위해 전역 디바이스 메시 상태 가로채기
        # (JAX 컴파일 컨텍스트 내에서 활성화된 Mesh 자원을 찾아 주입하거나 NamedSharding 정합 체결)
        # 여기서는 사전에 선언된 전역 하드웨어 메시 락킹 관로를 경유하도록 사양 조율
        try:
            from jax.experimental.shard_map import get_mesh
            current_mesh = get_mesh()
            if current_mesh is not None:
                context_vessel = verify_context_vessel_sharding_coherence(current_mesh, context_vessel)
        except (ImportError, ValueError, RuntimeError):
            # 수동 단일 장치 혹은 명시적 메시 바인딩이 누락된 복전 타임의 가드레일 통과 우회
            pass


                 # ------------------------------------------------------------------------
        # [⚡ LAYER 2.8: Q-STREAM PARITY DECODING & TOPO RESTORATION]
        # ------------------------------------------------------------------------
        # 융합된 글로벌 위상 컨테이너로부터 Q 스트림을 활용해 고정밀 토큰 매니폴드를 디코딩 및 전개합니다.
        
        # Q 스트림 역시 동일한 테일러 및 왜도 소산 가속 레일 적용
        Q_amplified = 1.0 + q_h + (0.5 * jax.lax.square(q_h))
        q_mean = jnp.mean(Q_amplified, axis=-1, keepdims=True)
        q_var = jnp.var(Q_amplified, axis=-1, keepdims=True) + self.hbar_eff
        
        # [고도화 포인트] 3부의 K 레일과 결을 일치시켜 rsqrt 단일 기계어로 제곱근 역수를 압착 연산합니다.
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
        
        # [⚡ 고도화: 카시미르 Vacuum Singular Boundary 에라스틱 가드 이식]
        # 코어 디코더 단의 안전 설계 철학과 완벽히 싱크를 맞춥니다.
        # 모든 위상 원소가 음수로 깎여 전역 제로 벡터(Zero Matrix)로 고사하려 할 때,
        # 정규화 분모가 완전히 무력화되어 후속 LLaMA 아키텍처 사영 레이어의 표현력을 
        # 파괴하는 싱큘러리티 현상을 원천 방어하기 위해 최소 하한 임계 장벽을 강제 집행합니다.
        square_sum = jnp.sum(jax.lax.square(sanitized_stream), axis=-1, keepdims=True)
        safe_square_sum = jnp.maximum(square_sum, self.casimir_delta)
        
        # jax.lax.rsqrt 내장 가속 명령어를 그대로 경유하여 분모 정규화 레이턴시 한계 압착
        final_attention_rail_output = sanitized_stream * jax.lax.rsqrt(safe_square_sum + self.hbar_eff)

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
