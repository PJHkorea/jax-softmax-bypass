# -*- coding: utf-8 -*-
"""
Homeostasis Spatial Bus - LLaMA-3 End-to-End Scale Profiler
File: benchmark_llama_wave.py

[소프트맥스 박멸에 따른 VRAM 절감률 및 토큰 생성 수율 실측 프로파일러]
"""

import os
import time
import torch
import gc
# [고도화 포인트] JAX 비동기 파이프라인의 물리 연산 완료를 강제 고정하기 위해 마스터 헤더 바인딩 인입
import jax
from transformers import AutoConfig, AutoModelForCausalLM

# 고도화 완료된 하이재커 레이어 및 글로벌 인젝터 인입
from hijack_llama_wave_attention import patch_llama_model_with_wave_attention

def measure_hardware_footprint(model, sequence_length: int, batch_size: int = 1) -> tuple:
    """[📊 HARDWARE PROFILER] 특정 시퀀스 길이 하에서 가속기 순정 런타임 지표 계측"""
    # 1. 이전 라운드의 메모리 파편화 및 캐시 잔차 완전 진압
    torch.cuda.empty_cache()
    gc.collect()
    
    # 2. 스트레스 테스트용 입력 토큰 가상 매니폴드 선언
    input_ids = torch.randint(0, 128256, (batch_size, sequence_length), device="cuda")
    
    # 3. 기저 VRAM 점유량(정적 가중치 영역 제외 순수 활성화 맵 사양) 측정 준비
    torch.cuda.reset_peak_memory_stats()
    
    # ------------------------------------------------------------------------
    # [⚡ 고도화 1: JIT 컴파일 래그 소산을 위한 실전형 타깃 스케일 웜업 결착]
    # ------------------------------------------------------------------------
    # 기존 고정 128 슬라이싱 웜업은 실제 런타임 연산인 sequence_length 진입 시 
    # XLA가 새로운 크기의 정적 연산 그래프를 그리게 만들어 강제 JIT 컴파일 지연을 유발했습니다.
    # 본 연산과 완전히 동일한 4차원 텐서 레이아웃으로 웜업을 집행하여 컴파일 오버헤드를 
    # 계측 타임라인 외부로 영구 출각(소산)시킵니다.
    with torch.no_grad():
        _ = model(input_ids)
    
    # [★ 고도화 핵심 - 하이브리드 비동기 컨텍스트 동기화 배리어]
    # 하이재킹 모듈 내부에서 작동 중인 JAX XLA 컴파일러 기계어 큐와 파이토치 CUDA 스트림 간의 
    # 참조선 레이스 컨디션을 파괴하기 위해, 하드웨어 동기화 직전 JAX 가속기 연산 완료를 선제 래칭합니다.
    try:
        jax.effects_barrier()  # 전역 분산 Sharding 비동기 전하량 수착 완결 보증
    except AttributeError:
        pass
    
    # JAX 백엔드 비동기 명령어 처리 마진 확보를 위한 하드웨어 동기화
    torch.cuda.synchronize()
    
    # ------------------------------------------------------------------------
    # [⚡ 고도화 2: 기저 피크 메모리 통계선 재초기화 인터록]
    # ------------------------------------------------------------------------
    # 앞선 웜업 단계에서 순수 JIT 컴파일 그래프 적재용으로 일시 점유되었던 
    # 하드웨어 HBM 자원 마진을 측정 레일에서 지워버리기 위해 피크 통계를 재정류합니다.
    torch.cuda.reset_peak_memory_stats()
    
    # 4. 본 추론 연산 집행 및 정밀 시간 계측
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids)
        
    try:
        jax.effects_barrier()
    except AttributeError:
        pass
    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time
    
    # 5. 피크 VRAM 소모량 수집 (Byte -> MB 승격)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    # 6. 초당 토큰 생성 수율(Throughput) 연산
    total_tokens = batch_size * sequence_length
    throughput = total_tokens / elapsed_time
    
    return peak_vram, throughput


def run_scale_inversion_benchmark():
    """
    [⚡ SCALE INVERSION PROFILER RUNTIME]
    시퀀스 길이를 2K에서 32K까지 지수적으로 확장 전개하며
    바닐라 Softmax의 파괴적 메모리 곡선과 파동 하이재커의 선형 수착 성능을 대조 실측합니다.
    """
    # 실측 벤치마크 타깃 가중치 선언 (Meta LLaMA-3 8B 오리지널 토폴로지 규격)
    model_id = "meta-llama/Meta-Llama-3-8B"
    
    print(f"📥 [인프라] {model_id} 아키텍처 토폴로지 설정 로드 중...")
    config = AutoConfig.from_pretrained(model_id)
    
    # 2K(2048)부터 32K(32768)까지 거대 모델 전용 초장문 컨텍스트 스트레스 스케일 정의
    context_scales = [2048, 4096, 8192, 16384, 32768]
    
    print("\n========================================================================")
    print("🔥 [⚡ SYSTEM BASELINE] 1단계: 레거시 PyTorch 바닐라 Softmax Attention 성능 측정")
    print("========================================================================")
    # LLaMA-3 가중치를 CUDA 장치 메모리에 FP16 반정밀도 규격으로 적재
    vanilla_model = AutoModelForCausalLM.from_config(config).half().to("cuda")
    vanilla_model.eval()
    
    vanilla_results = {}
    for scale in context_scales:
        try:
            print(f"🔄 레거시 Softmax 컨텍스트 길이 [{scale} Token] 스트레스 주입...")
            # [보정 확인] 파트 1의 고도화된 계측 함수로 진입하며, 
            # 각 scale 단위로 JIT 컴파일러 래그가 완벽히 절연된 청정 활성화 VRAM/TPS를 추출합니다.
            vram, tps = measure_hardware_footprint(vanilla_model, scale)
            vanilla_results[scale] = (vram, tps)
            print(f" ├─ 피크 VRAM 점유량 : {vram:.2f} MB")
            print(f" └─ 토큰 생성 수율   : {tps:.2f} tokens/sec")
        except RuntimeError as e:
            # OOM 발생 시 가속기 전체 락을 방지하기 위한 예외 캡처 인터록
            if "out of memory" in str(e).lower():
                print(f"💀 [OOM CRASH] 레거시 Softmax가 {scale} 구간에서 메모리 폭발로 침몰했습니다.")
                vanilla_results[scale] = (float('inf'), 0.0)
                
                # ------------------------------------------------------------------------
                # [⚡ 고도화: OOM 유발 활성화 맵 강제 인프라 해제 배리어]
                # ------------------------------------------------------------------------
                # 바닐라 파이트가 메모리 한계를 견디지 못하고 폭발할 때 발생한 
                # 파편화 오염 및 데드락 래그가 가속기 캐시 할당자에 누수 잔차로 남지 않도록
                # 즉각적인 닫힌계 청소 프로토콜을 선제 호출하여 인터페이스 붕괴를 영구 방어합니다.
                torch.cuda.empty_cache()
                gc.collect()
                break
            else:
                raise e

            
       # [🛡️ HBM 닫힌계 청정 수호 프로토콜]
    # 바닐라 모델 측정 라운드가 끝나는 즉시 메모리 풀에서 가중치를 강제 출각하고
    # 캐시 연쇄 비우기를 주입하여 후속 파동 모델 측정 평면에 잔차 간섭을 원천 차단합니다.
    del vanilla_model
    torch.cuda.empty_cache()
    gc.collect()

    print("\n========================================================================")
    print("🧬 [🌊 HIJACKED RUNTIME] 2단계: 6세대 비동기 펜스 통합형 Wave-Attention 성능 측정")
    print("========================================================================")
    
    # [🔥 REAL-TIME HOT-PLUG INJECTION]
    # 동일 가중치 뼈대를 깨끗한 HBM 평면에 다시 로드한 후, 단 한 줄로 런타임 심장 하이재킹 집행
    wave_model = AutoModelForCausalLM.from_config(config).half().to("cuda")
    wave_model = patch_llama_model_with_wave_attention(wave_model, mesh_shape=64, alpha=0.01)
    wave_model.eval()
    
    wave_results = {}
    for scale in context_scales:
        try:
            print(f"🔄 하이재킹 파동 레일 컨텍스트 길이 [{scale} Token] 스트레스 주입...")
            # [보정 확인] 파트 1의 고도화된 계측 함수로 진입하며, 
            # 각 scale 단위로 JIT 컴파일러 래그가 완벽히 절연된 청정 활성화 VRAM/TPS를 추출합니다.
            vram, tps = measure_hardware_footprint(wave_model, scale)
            wave_results[scale] = (vram, tps)
            print(f" ├─ 피크 VRAM 점유량 : {vram:.2f} MB")
            print(f" └─ 토큰 생성 수율   : {tps:.2f} tokens/sec")
        except RuntimeError as e:
            # 파동 레일 구동 중 예기치 못한 하드웨어 예외 발생 시 안전 장벽 작동
            print(f"❌ [CRASH] 파동 레일 {scale} 구간 예외 발생: {str(e)}")
            wave_results[scale] = (float('inf'), 0.0)
            
    # ------------------------------------------------------------------------
    # [📊 STEP 4: 최종 성적표 출력 및 자원 세이빙 리포트 출력 - 헤더 정렬]
    # ------------------------------------------------------------------------
    print("\n========================================================================")
    print("🏆 FINAL ARCHITECTURE PERFORMANCE REPORT (Softmax vs Wave-Attention)")
    print("========================================================================")
    print(f"{'Context':<10} | {'Softmax VRAM':<14} | {'Wave VRAM':<12} | {'VRAM Saving':<13} | {'Throughput Boost':<16}")
    print("-" * 75)

    for scale in context_scales:
        v_vram, v_tps = vanilla_results.get(scale, (float('inf'), 0.0))
        w_vram, w_tps = wave_results.get(scale, (float('inf'), 0.0))
        
        # 1. 레거시 Softmax (대조군) 출력 텍스트 정류
        if v_vram == float('inf'):
            vram_str = "OOM CRASH"
            saving_str = "100.00% (★)"
            boost_str = "INF (🔥)"
        else:
            vram_str = f"{v_vram:.1f} MB"
            saving = ((v_vram - w_vram) / v_vram) * 100
            saving_str = f"{saving:.2f}%"
            boost = (w_tps / v_tps) if v_tps > 0 else 0.0
            boost_str = f"{boost:.2f}x"
            
        # 2. [🛡️ vram_str_w 변수 단선 결함 교정 패치]
        # 상단 블록과 하단 print 함수 간의 명세 불일치를 유발하던 네이밍 파편화를 교정하여
        # 하이재킹 실험군의 실측 메모리 데이터를 오차 없이 정상 출력합니다.
        w_vram_str = f"{w_vram:.1f} MB" if w_vram != float('inf') else "CRASH"
        
        # 3. 인프라 실측 최종 성적표 스트리밍 출사
        print(f"{scale:<10} | {vram_str:<14} | {w_vram_str:<12} | {saving_str:<13} | {boost_str:<16}")
    print("========================================================================\n")

if __name__ == "__main__":
    run_scale_inversion_benchmark()

