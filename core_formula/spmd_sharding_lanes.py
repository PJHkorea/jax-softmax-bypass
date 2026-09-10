# -*- coding: utf-8 -*-
"""
Homeostasis Spatial Bus - JAX XLA SPMD Sharding Orchestrator
File: core_formula/spmd_sharding_lanes.py
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

def establish_global_hardware_sharding_lanes(num_data_replicas: int = 4, num_model_partitions: int = 8):
    """
    [⚡ GLOBAL HARDWARE MESH MATRIX SETUP]
    물리 가속기 디바이스 어레이를 탐지하여 데이터 병렬선과 모델 평면 축으로 2중 격리 배정합니다.
    """
    # 1. 시스템 내 가용 물리 가속기(GPU/TPU) 소켓 전수 가로채기
    devices = jax.devices()
    total_devices = len(devices)
    required_devices = num_data_replicas * num_model_partitions
    
    if total_devices < required_devices:
        raise RuntimeError(
            f"[SHARDING CLUSTER ERROR] 가용 가속기 자원 부족! (요구량: {required_devices}, 실측치: {total_devices})\n"
            f"물리 노드 토폴로지 연결 상태를 전단 체크하십시오."
        )
        
    # 2. Bare-Metal 레이아웃 사상에 맞춰 디바이스 어레이 물리 재배치
    hardware_grid = mesh_utils.create_device_mesh((num_data_replicas, num_model_partitions))
    global_mesh = Mesh(hardware_grid, ('data', 'model'))
    
    print(f"🛰 [CLUSTER] SPMD 실리콘 메시 헌법 선포 완료 | Topology: {num_data_replicas}x{num_model_partitions}")
    return global_mesh

def apply_wave_attention_sharding_rules(global_mesh: Mesh, q: jax.Array, k: jax.Array, v: jax.Array) -> tuple:
    """
    [⚡ MULTI-DIMENSIONAL TENSOR SHARDING INJECTION]
    입력 4차원 매니폴드가 하드웨어 버스를 탈 때, 단 1바이트의 주소 찢어짐이나
    불필요한 All-Gather 통신 오버헤드가 터지지 않도록 전역 샤딩 사물쇠를 체결합니다.
    """
    # Q, K, V 마스터 4차원 레일 레이아웃 스펙 규격: [Batch, NumHeads, SeqLen, HeadDim]
    # 'data' 축으로 Batch 분산, 'model' 축으로 NumHeads를 쪼개어 가속기 SRAM 내부로 다이렉트 이식
    # SeqLen과 HeadDim은 물리적으로 쪼개지 않고 단일 가속기 코어 연산 장치(SFU) 내에 밀착 고정
    attention_input_spec = P('data', 'model', None, None)
    
    # XLA 전역 네임스페이스 통제 명세서 합성
    sharding_rule = NamedSharding(global_mesh, attention_input_spec)
    
    # ------------------------------------------------------------------------
    # [⚡ 고도화: jax.lax.with_sharding_constraint 컴파일러 제약식 유도]
    # ------------------------------------------------------------------------
    # 단순히 jax.device_put으로 밀어 넣는 방식은 순방향에서는 안전할지 몰라도,
    # 역전파(Backpropagation) 자동 미분 도함수 그래프가 생성될 때 
    # XLA 컴파일러가 장치 분산 트래킹 배리어를 놓쳐 ConcretizationTypeError 크래시를 유발할 수 있습니다.
    # 주소선 포인터를 가로챔과 동시에 물리 샤딩 제약식을 강제 주입하여 컴파일 최적화 라인을 동결합니다.
    q_sharded = jax.lax.with_sharding_constraint(q, sharding_rule)
    k_sharded = jax.lax.with_sharding_constraint(k, sharding_rule)
    v_sharded = jax.lax.with_sharding_constraint(v, sharding_rule)
    
    return q_sharded, k_sharded, v_sharded


def verify_context_vessel_sharding_coherence(global_mesh: Mesh, context_vessel: jax.Array) -> jax.Array:
    """
    [⚡ O(1) CONTEXT VESSEL MEMORY LOCK]
    K와 V가 수착된 파동 공간 글로벌 컨테이너가 분산 장치 간에 불필요하게 복사되거나
    바운싱 노이즈를 일으키지 않도록 메모리 배치 상태를 물리적으로 하드락킹합니다.
    """
    # context_vessel shape: [Batch, NumHeads, MeshShape, HeadDim]
    # 배치의 분산과 헤드의 격리가 연속적으로 완전히 유지됨을 단언합니다.
    vessel_spec = P('data', 'model', None, None)
    vessel_sharding = NamedSharding(global_mesh, vessel_spec)
    
    # ------------------------------------------------------------------------
    # [⚡ 고도화: 역전파 경로 전하량 보존용 HLO Sharding Constraint 강제 결착]
    # ------------------------------------------------------------------------
    # 파동 디코더 코어에서 Q 스트림과 매칭되어 유클리드 토폴로지를 디코딩해내기 전,
    # context_vessel이 분산 노드 메모리 풀 사이에서 원치 않게 재배치(Resharding)되거나
    # 복사본 파편을 형성하는 Latency 병목을 원천 봉쇄합니다.
    # jax.device_put 대신 컴파일타임 제약식 펜스를 쳐서 O(1) 고정 스펙을 완벽하게 록킹합니다.
    vessel_locked = jax.lax.with_sharding_constraint(context_vessel, vessel_sharding)
    
    return vessel_locked
