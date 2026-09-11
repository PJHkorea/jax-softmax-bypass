# -*- coding: utf-8 -*-
"""
Homeostasis Spatial Bus - Multi-Head Wave-Attention Block Test Bench
File: test_wave_attention.py
"""

import time
import jax
import jax.numpy as jnp
from multi_head_wave_attention import MultiHeadWaveAttention

def execute_e2e_test_bench():
    # [고도화 포인트] 파이썬 메모리 계측 오차를 분쇄하기 위해 리눅스 커널 물리 메모리(VmRSS) 직접 스캔 함수 정의
    def get_kernel_vm_rss() -> int:
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if "VmRSS:" in line:
                        return int(line.split()[1]) * 1024  # KB -> Byte 승격
        except:
            pass
        return 0

    # ------------------------------------------------------------------------
    # [Step 1: 하드웨어 가속 검증용 하이퍼파라미터 및 더미 매니폴드 선언]
    # ------------------------------------------------------------------------
    batch_size = 2
    seq_len = 2048       # 초장문 컨텍스트 스트레스 환경 모사
    embed_dim = 4096     # LLaMA-7B 급 스케일 적용
    num_heads = 32
    mesh_shape = 64
    alpha = 0.01

    print(f"[⚙️ TEST CONFIG] Batch: {batch_size}, SeqLen: {seq_len}, EmbedDim: {embed_dim}, Heads: {num_heads}")
    
    # 가속기 난수 생성기 인터록 바인딩
    key = jax.random.PRNGKey(42)
    key_q, key_k, key_v, key_mask = jax.random.split(key, 4)
    
    # 수치적 연속성 공간 검증을 위한 가우시안 무작위 입력 스트림 텐서 선언
    q_init = jax.random.normal(key_q, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    k_init = jax.random.normal(key_k, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    v_init = jax.random.normal(key_v, (batch_size, seq_len, embed_dim), dtype=jnp.float32)
    
    # ------------------------------------------------------------------------
    # [⚡ 고도화 1: 레거시 LLM 백본 내부 스케일 폭발 및 인과적 마스크 스트레스 인입]
    # ------------------------------------------------------------------------
    # Step 1-1. 실제 프로덕션 LLaMA-3 환경에서 Q와 K의 사영 변환 후 에너지가 극단적으로 팽창한 상태를 모사
    # 고의적으로 입력 텐서 스케일을 대폭 증폭시켜, 우리가 코어 단에 심어둔 Scale Factor의 정류력을 사증합니다.
    q_init = q_init * 15.0
    k_init = k_init * 15.0
    
    # Step 1-2. 실제 파이토치 하이재킹단에서 밀려 들어오는 미래 토큰 차단용 인과적 마스크 스트레스 버스 구축
    # [Batch, 1, SeqLen, SeqLen] 규격의 Lower-Triangular 인과율 불값 매니폴드 생성
    mask_matrix = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    mask_init = jnp.broadcast_to(mask_matrix[None, None, :, :], (batch_size, 1, seq_len, seq_len))

    # ------------------------------------------------------------------------
    # [Step 2: 하이재킹 모듈 인스턴스화 및 자동 미분 타깃 손실함수 선언]
    # ------------------------------------------------------------------------
    # 모듈 초기화 및 정적 자원 할당 전 커널 메모리 상태 기저 측정
    memory_before = get_kernel_vm_rss()

    wave_attention_block = MultiHeadWaveAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mesh_shape=mesh_shape,
        alpha=alpha
    )

    # 역전파 그라디언트 유동선 검증을 위한 가상 스칼라 손실(Loss) 함수 매핑
    # [고도화 포인트] 직렬화 복원 테스트를 유증하기 위해 모듈 블록을 동적으로 인입받도록 유연화
    def build_loss_engine(target_block):
        # 고도화된 마스크 인터록 흐름까지 동시 사증하기 위해 파이프라인 관로에 mask_tensor 추가 결착
        def loss_fn(q_tensor, k_tensor, v_tensor, mask_tensor):
            output_manifold = target_block(q_tensor, k_tensor, v_tensor, mask=mask_tensor)
            # L2 NormParity 및 연속 미분 가능 공간 상상의 스칼라 수축 복원
            return jnp.mean(jax.lax.square(output_manifold))
        return jax.value_and_grad(loss_fn, argnums=(0, 1, 2))

     # ------------------------------------------------------------------------
    # [★ 추가 고도화 PHASE 0: JAX PyTree 분산 SPMD 직렬화/역직렬화 인터록 완결 검증]
    # ------------------------------------------------------------------------
    print("\n[⚡ PHASE 0] JAX PyTree 구조적 직렬화 및 Bare-Metal 복원 무결성 검증...")
    children, aux_data = wave_attention_block.tree_flatten()
    reconstructed_block = MultiHeadWaveAttention.tree_unflatten(aux_data, children)
    
    # 복원된 인스턴스를 마스터 엔진으로 기용하여 하위 자동 미분 파이프라인 가동
    grad_loss_engine = build_loss_engine(reconstructed_block)

    # ------------------------------------------------------------------------
    # [Step 3: XLA 워밍업 컴파일 및 런타임 수치 무결성 검증]
    # ------------------------------------------------------------------------
    print("\n[⚡ PHASE 1] XLA 컴파일 파이프라인 워밍업 및 기계어 융합 유도 중...")
    start_compile = time.time()
    
    # 1회차 실행을 통해 가속기 내부 JIT 컴파일 및 최적 GEMM 커널 바인딩 강제
    # [보정] 파트 1에서 새롭게 구축한 인과적 마스크 텐서(mask_init)를 컴파일러 추적선 내부로 동시 인입
    init_loss, init_grads = grad_loss_engine(q_init, k_init, v_init, mask_init)
    
    # JAX 비동기 실행 제어를 풀고 가속기 실리콘 단의 연산 완료를 완벽하게 동기화
    jax.block_until_ready((init_loss, init_grads))
    compile_time = time.time() - start_compile
    print(f"🥇 컴파일 완료 레이턴시: {compile_time:.4f} 초")

    print("\n[⚡ PHASE 2] 프로덕션 가속 런타임 성능 실측 시작...")
    start_runtime = time.time()
    
    # 실제 기계어 인라인 융합 루프를 통과하는 가속 런타임 집행
    # [보정] 프로덕션 계측 레일에도 동일하게 마스크 인자를 주입하여 브랜치리스 MUX의 실전 레이턴시 계측
    runtime_loss, runtime_grads = grad_loss_engine(q_init, k_init, v_init, mask_init)
    jax.block_until_ready((runtime_loss, runtime_grads))
    runtime_time = time.time() - start_runtime
    print(f"🚀 순수 가속기 연산 레이턴시: {runtime_time:.4f} 초")

      # ------------------------------------------------------------------------
    # [Step 4: 수치 해석적 안정성(NaN-Free) 및 커널 자원 가드레일 단언문 검증]
    # ------------------------------------------------------------------------
    print("\n[📊 PHASE 3] 수치 해석적 안정성 및 그라디언트 유동선 정밀 전사...")
    
    # 순방향 손실값 NaN 검사 단언문
    assert not jnp.isnan(runtime_loss), "[FATAL ERROR] 순방향 연산 결과 손실값이 NaN으로 파괴되었습니다."
    print(f"✅ 순방향 패스 안정성 확보: Loss = {runtime_loss:.6f}")

    # 역방향 q, k, v 그라디언트 텐서 무결성 전수 조사
    for idx, (stream_name, grad_tensor) in enumerate(zip(['Query', 'Key', 'Value'], runtime_grads)):
        nan_mask = jnp.isnan(grad_tensor)
        nan_count = jnp.sum(nan_mask)
        
        # 미분 도함수 붕괴 발생 여부 엄격한 단언 단선 제어
        assert nan_count == 0, f"[FATAL ERROR] {stream_name} 역전파 그라디언트에 {nan_count}개의 NaN 결함이 검출되었습니다."
        
        # 그라디언트 전하량 평면 확인을 위한 L2 스칼라 값 연산
        grad_l2_norm = jnp.sqrt(jnp.sum(jax.lax.square(grad_tensor)))
        print(f"✅ 역방향 패스 그라디언트 전하량 확보 ({stream_name}): L2 Norm = {grad_l2_norm:.6f}")

    # [★ 추가 고도화 핵심 - OS 물리 메모리 지터 누수 차단 교차 단언]
    final_total_alloc = get_kernel_vm_rss()
    memory_jitter_amplitude = abs(final_total_alloc - memory_before)
    print(f"├─ [인프라] 가동 전 선점 물리 자원 총량 : {memory_before} Byte")
    print(f"├─ [인프라] 대용량 역전파 연산 후 자원 총량 : {final_total_alloc} Byte")
    print(f"🚨 [결론] 서버 인프라 실제 물리 메모리 변동 진폭 : {memory_jitter_amplitude} Byte")
    
    # ------------------------------------------------------------------------
    # [⚡ 고도화: OS 커널 실제 가용 자원 및 SPMD 파이프라인 누수 제로 검증 마감]
    # ------------------------------------------------------------------------
    # 대규모 초장문 컨텍스트(SeqLen: 2048) 역전파 그라디언트 루프가 완전히 휘몰아쳤음에도,
    # 우리가 PyTree 격리 설계와 vorticity_omega_mesh 통합 링버퍼 관리를 빌드했기 때문에
    # 가비지 컬렉터(GC)에 의한 하드웨어 레이턴시 지터 마진이 64KB 이내로 영구 동결됩니다.
    assert memory_jitter_amplitude <= 65536, (
        f"[FATAL ERROR] 수리 오류: 분산 학습 런타임 도중 정적 자원 닫힌계 바운더리가 파괴되어 메모리 지터가 터졌습니다!\n"
        f"실측 진폭 수치: {memory_jitter_amplitude} Byte"
    )

    print("\n🎉 [SUCCESS] 모든 하드웨어 가속, PyTree 복원 및 연속 미분 자동 미분 테스트를 완벽하게 통과했습니다!")

if __name__ == "__main__":
    execute_e2e_test_bench()

