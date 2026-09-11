# -*- coding: utf-8 -*-
"""
Homeostasis Spatial Bus - Multi-Head Wave-Attention Block (Part 1)
File: core_formula/multi_head_wave_attention.py
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Any, Optional

# [아키텍처 정류] core_formula/ 내부 통합 이동에 따른 청정 상대 경로 변환
from .softmax_bypassing_decoder import UpgradedSoftmaxBypassingDecoder

# 우리가 격리 완공 및 2차 고도화한 실리콘 정류기 무기고 인입
from bypass_rectifiers import LocalHomeostaticRectifier, TorusTopologyRotaryEmbedding

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
        self.casimir_delta = 1e-4

        # 푸리에 복소 평면 위상 데드존 거세를 위한 하드웨어 Stride 짝수 정합성 검증 인터록
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"[HARDWARE ALIGNMENT ERROR] head_dim({self.head_dim})은 반드시 짝수여야 합니다. "
                f"임베딩 차원 혹은 헤드 개수를 재조정하십시오."
            )

        # ------------------------------------------------------------------------
        # [🌟 도킹 고도화 1: 3대 정류기 무기를 멀티헤드 마스터 제어선에 결착]
        # ------------------------------------------------------------------------
        # 가속기 내부 연산 루프(Q_h, K_h) 진입 전후의 전역 감축 락을 영구 분쇄하기 위해
        # 국소 항상성 정류기 커널을 헤드 차원(head_dim) 사양에 정합하여 결착합니다.
        self.rectifier = LocalHomeostaticRectifier(
            head_dim=self.head_dim,
            alpha=self.alpha,
            casimir_delta=self.casimir_delta
        )
        
        # 문맥 길이가 늘어남에 따라 회전 각도가 무한 발산하는 레거시 결함을 제거하고
        # 파동 위상을 닫힌 도넛 표면에 감금하기 위해 토러스 위치 임베딩 커널을 인스턴스에 탑재합니다.
        self.torus_rope = TorusTopologyRotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=131072,  # 실리콘 최대 주행 한계선 설정
            base=10000.0,
            torus_radius=1.0     # 토러스 곡률 주 반경 동결
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
    def tree_flatten(self) -> Tuple[Tuple[jax.Array], Tuple[Any, ...]]:
        """
        XLA 컴파일러가 분산 장치 메모리(HBM) 트래킹을 놓치지 않도록 동적 텐서와 정적 상수를 격리 분리합니다.
        새로 내장된 국소 정류기(rectifier)와 토러스 RoPE(torus_rope) 커널 인스턴스를 
        정적 메타데이터 관로 끝에 추가 인입하여 인프라 토폴로지의 직렬화 무결성을 수호합니다.
        """
        # 자동 미분 및 컴파일러가 실시간 추적할 동적 자식 노드는 단 하나로 압착 유지 (GC 오버헤드 0MB 동결)
        children = (self.vorticity_omega_mesh,)
        
        # [고도화] 새로 장착된 정적 정류 무기 2가지를 관로 인덱스 7, 8번에 순차 하이재킹 결착
        aux_data = (
            self.embed_dim,
            self.num_heads,
            self.head_dim,
            self.mesh_shape,
            self.alpha,
            self.hbar_eff,
            self.casimir_delta,
            self.rectifier,   # 🌟 인덱스 7: 국소 정류기 객체 고정 사상
            self.torus_rope   # 🌟 인덱스 8: 토러스 RoPE 객체 고정 사상
        )
        return children, aux_data


      @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[jax.Array]) -> "MultiHeadWaveAttention":
        """
        [SPMD 정적 단언 크래시 결함 교정]
        역전파 자동 미분 그래프 재조립 시, 새롭게 도킹된 국소 정류기 및 토러스 RoPE 인스턴스까지
        단 1비트의 유실 없이 원형 구조 그대로 정밀 언팩 복원합니다.
        """
        # 생성자 우회 매핑 인터록 가드 발동
        embed_dim_raw = int(aux_data[0])
        num_heads_raw = int(aux_data[1])
        
        # [고도화 포인트] XLA 타입 미스매치를 방지하기 위한 구조 가드레일 유지
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
        
        # 확장된 aux_data 인덱스 규격에 맞춰 정적 방화벽 상수 복원 바인딩
        obj.casimir_delta = float(aux_data[6])
        
        # [🌟 도킹 복원 인터록]: 직렬화 관로(인덱스 7, 8)에서 정류기 및 토러스 RoPE 인스턴스를 추출하여 원형 복제 바인딩
        obj.rectifier = aux_data[7]
        obj.torus_rope = aux_data[8]
        
        # 자식 노드로부터 복사 오버헤드 없이 bare-metal 바인딩 복구
        obj.vorticity_omega_mesh = children[0]
        
        # [아키텍처 정류]: 내부 하위 디코더 객체 레이어 복원 시 상대 경로 정합 사상 유지
        from .softmax_bypassing_decoder import UpgradedSoftmaxBypassingDecoder
        decoder_mesh_target = obj.mesh_shape[0] if isinstance(obj.mesh_shape, tuple) else obj.mesh_shape
        obj.decoders = [
            UpgradedSoftmaxBypassingDecoder(mesh_shape=decoder_mesh_target, feature_dim=obj.head_dim, alpha=obj.alpha)
            for _ in range(obj.num_heads)
        ]
        
        return obj



      @partial(jax.jit, static_argnums=(0,))
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, mask: Optional[jax.Array] = None, mesh: jax.sharding.Mesh = None) -> jax.Array:
        """
        [⚡ OPERATIONAL FUSION RUNTIME GATEWAY - 4D GEMM RE-LAYOUT]
        [Part 3: 입력 스트림 헤드 분할 및 토러스 RoPE 감금 인터록 정렬]
        
        기존 Softmax Attention의 입출력 규격을 완벽히 유지하되,
        마스크 소거 직후 q_h와 k_h 레일에 토러스 위치 위상 감금 제약을 원자적으로 집행합니다.
        """
        # q, k, v Input Shape: [Batch, SeqLen, EmbedDim]
        batch_size, seq_len, _ = q.shape
        target_dtype = q.dtype

        # ------------------------------------------------------------------------
        # [🛡️ 하드웨어 텐서 코어 직결형 GEMM 레일 레이아웃 전환]
        # ------------------------------------------------------------------------
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
        default_mask = jnp.ones((batch_size, 1, 1, seq_len), dtype=jnp.bool_)
        safe_mask = jax.lax.select(mask is None, default_mask, mask)
        
        # k_h와 v_h 두 스트림의 물리 전하량을 동시에 0으로 수축시켜 정보 누수를 완벽 차단합니다.
        k_h = jnp.where(safe_mask, k_h, 0.0)
        v_h = jnp.where(safe_mask, v_h, 0.0)

        # ------------------------------------------------------------------------
        # [🌟 도킹 고도화 2: 레지스터 프리 토러스 위치 임베딩(RoPE) 제약 집행]
        # ------------------------------------------------------------------------
        # 0선 정적 인덱스 레일(seq_idx)을 온플라이로 빌드하여 토러스 매니폴드 사영을 집행합니다.
        # 파동 공간 투영 전, 각도가 무한히 발산하며 복소 위상을 파괴하는 현상을 실리콘 레벨에서 봉쇄합니다.
        seq_idx = jnp.arange(seq_len, dtype=target_dtype)
        
        # Q 스트림과 K 스트림에 각각 독립적인 닫힌계 위상 회전을 원자적으로 관류시킵니다.
        q_h = self.torus_rope(stream=q_h, seq_idx=seq_idx, mesh=mesh)
        k_h = self.torus_rope(stream=k_h, seq_idx=seq_idx, mesh=mesh)

        # ------------------------------------------------------------------------
        # [🌊 멀티헤드 평행 세계 집행 및 파동 디코더 코어 융합 유도]
        # ------------------------------------------------------------------------
        # 토러스 위상이 록킹된 청정 복소 레일들을 최종 적산 및 복원 파이프라인으로 관류 인입합니다.
        return self._execute_wave_integration(q_h, k_h, v_h, mesh=mesh)




        def _execute_wave_integration(self, q_h: jax.Array, k_h: jax.Array, v_h: jax.Array, mesh: jax.sharding.Mesh = None) -> jax.Array:
        """
        [⚡ WAVE CONTRACTION & RECONSTRUCTION ENGINE]
        [Part 4: 파동 수축 무게중심 적분 및 국소 정류기 융합 컨테이너 빌드]
        
        기존의 jnp.mean/jnp.var 전역 감축 락을 완전히 분쇄 거세한 채,
        호너법 FMA와 국소 왜도 제동만으로 청정 context_vessel을 0MB 평면에 조립합니다.
        """
        batch_size, _, seq_len, _ = q_h.shape
        target_dtype = q_h.dtype

        # 1부에서 동결 적재된 단일 링버퍼(self.vorticity_omega_mesh)의 형상을 4차원 매니폴드에 동기화
        omega_vessel = self.vorticity_omega_mesh[None, :, :, None]

        # ------------------------------------------------------------------------
        # [🛡️ HLO INLINE FUSION - 멀티헤드 독립 파동 위상 평면 기저 선언]
        # ------------------------------------------------------------------------
        half_dim = self.head_dim // 2
        grid_axis = jnp.arange(half_dim, dtype=target_dtype) / float(half_dim)
        grid_axis = grid_axis[None, None, None, :]

        # 위상 기하학적 대칭성 보호(Fourier Basis Orthogonality)를 적용한 오일러 복소 평면 모사
        wave_sin = jnp.sin(omega_vessel * grid_axis)
        wave_cos = jnp.cos(omega_vessel * grid_axis)
        
        # 정보 손실(Rank Collapse)을 파동 간섭 효과로 무력화하는 직교 기저 매트릭스 조립
        field_wave_T = jnp.concatenate([wave_sin, wave_cos], axis=-1)

        # ------------------------------------------------------------------------
        # [⚡ LAYER 2.5: O(N) LINEAR VESSEL CONTRACTION - 하이재킹 핵심]
        # ------------------------------------------------------------------------
        # 바닐라 트랜스포머처럼 Q와 K를 먼저 곱해 [SeqLen, SeqLen]을 만들지 않고, 
        # K와 V를 파동 공간(Mesh) 내에서 먼저 융합하여 선형 복잡도 글로벌 컨테이너를 빌드합니다.
        
        # [🌟 도킹 고도화 1: 호너법(Horner's Method) 기반 K 스트림 대수 평면 1클록 인라인 FMA 융합]
        # x * (1 + x + 0.5 * x^2) 구조를 호너법으로 치환하여 가속기 SRAM 내부 임시 버퍼 적재 병목을 원천 숙청합니다.
        K_amplified = k_h * (1.0 + k_h * (1.0 + 0.5 * k_h))
        
        # [🌟 도킹 고도화 2: 중복 전역 감축 락 거세형 국소 항상성 정류기 실행 (K 레일)]
        # 기존의 jnp.mean/jnp.var 전역 감축 연산을 전면 철폐하고, 
        # 분산 메시 제약선이 인입된 최고 고도화 정류 커널로 원소별 레지스터 단 초고속 평탄화를 관류 집행합니다.
        K_rectified = self.rectifier(
            x=K_amplified, 
            gamma=None,  # 어텐션 내부 연산이므로 백본 가중치 주입은 패스 (0ns 즉시 정류 처리)
            use_gemma_offset=False, 
            mesh=mesh
        )

        # Step 3: 정류된 K 매니폴드를 파동 기저 평면으로 투영(Projection)
        # field_wave_T: [B, H, M, D], K_rectified.T: [B, H, D, L] -> k_wave: [B, H, M, L]
        k_wave = jnp.matmul(field_wave_T, K_rectified.transpose((0, 1, 3, 2)))

        # Step 4: 연속체 무게중심 모멘트 적본 공간 내에서 V 스트림과 선형 결합 수착 수행
        # k_wave: [B, H, M, L], v_h: [B, H, L, D] -> context_vessel: [B, H, M, D]
        context_vessel = jnp.matmul(k_wave, v_h)

               # ------------------------------------------------------------------------
        # [⚡ 고도화: SPMD 분산 노드 주소선 바운싱 차단 인터록]
        # ------------------------------------------------------------------------
        # K와 V가 결착하여 압착해낸 고정 차원의 정보 용기(context_vessel)에 
        # spmd_sharding_lanes에서 정의한 NamedSharding 명세를 강제 주입합니다.
        # [아키텍처 정류]: core_formula/ 하부 통합 이동에 따른 청정 상대 경로 변환 인입
        from .spmd_sharding_lanes import verify_context_vessel_sharding_coherence
        
        # 1순위: 상위 제어선으로부터 명시적으로 명해진 전역 하드웨어 mesh 록킹 관로를 최우선 적용
        if mesh is not None:
            context_vessel = verify_context_vessel_sharding_coherence(mesh, context_vessel)
        else:
            # 2순위 백업 가드: 명시적 메시 인입이 누락되었을 경우에만 가상 머신(VM) 컨텍스트를 역추적 우회
            try:
                from jax.experimental.shard_map import get_mesh
                current_mesh = get_mesh()
                if current_mesh is not None:
                    context_vessel = verify_context_vessel_sharding_coherence(current_mesh, context_vessel)
            except (ImportError, ValueError, RuntimeError):
                # 수동 단일 장치 주행 시의 예외 처리 가드레일 통과
                pass


               # ------------------------------------------------------------------------
        # [⚡ LAYER 2.8: Q-STREAM PARITY DECODING & TOPO RESTORATION]
        # ------------------------------------------------------------------------
        # 융합된 글로벌 위상 컨테이너로부터 Q 스트림을 활용해 고정밀 토큰 매니폴드를 디코딩 및 전개합니다.
        
        # [🌟 도킹 고도화 1: 호너법(Horner's Method) 기반 Q 스트림 대수 평면 1클록 인라인 FMA 융합]
        # K 레일과 결을 완전 일치시켜 임시 변수 공간(Q_squared, Q_amplified)의 레지스터 스필을 완전 숙청합니다.
        Q_amplified = q_h * (1.0 + q_h * (1.0 + 0.5 * q_h))
        
        # [🌟 도킹 고도화 2: 중복 전역 감축 락 거세형 국소 항상성 정류기 실행 (Q 레일)]
        # jnp.mean/jnp.var 전역 감축 연산을 전면 철폐하고, SRAM 레지스터 단에서 자가 중화하도록 관류 집행합니다.
        Q_rectified = self.rectifier(
            x=Q_amplified,
            gamma=None,  # 독립 정류 모드로 0ns 즉시 사출 처리
            use_gemma_offset=False,
            mesh=mesh
        )

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

        # ------------------------------------------------------------------------
        # [🌟 도킹 고도화 3: 사출구 최종 텐서 레일 분산 샤딩 헌법 집행]
        # ------------------------------------------------------------------------
        # 최종 프로덕션 복원 출력물인 final_output이 후단 FFN(SwiGLU) 레이어로 진입할 때
        # 가속기 노드 메모리 균열을 방지하도록 3차원 레일 상에 정적 포인터 고정 제약을 선포합니다.
        if mesh is not None:
            # final_output shape: [Batch, SeqLen, EmbedDim] -> 마지막 채널 축을 'model' 파티션으로 록킹
            out_sharding_spec = P(None, None, 'model')
            named_out_sharding = jax.sharding.NamedSharding(mesh, out_sharding_spec)
            final_output = jax.lax.with_sharding_constraint(final_output, named_out_sharding)

        return final_output

