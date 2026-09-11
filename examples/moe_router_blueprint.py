"""
File: examples/moe_router_blueprint.py
"""
import jax
import jax.numpy as jnp
from typing import Tuple, List
from bypass_rectifiers.taylor_glu import HomeostaticTaylorGluCore

class StaticMoEWaveRouter:
    """
    [UNIVERSAL STATIC MoE ROUTER GATEWAY]
    Symmetrically pads dynamic token-to-expert allocations into a frozen, 
    static tensor layout to enforce 1-Cycle FMA execution queues.
    """
    def __init__(self, num_experts: int = 8, expert_capacity: int = 64, hidden_dim: int = 4096):
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.hidden_dim = hidden_dim
        
        # 기존 core_formula 오염 없이, 고성능 TaylorGLU 코어들을 부품처럼 리스트로 관리
        self.experts = [
            HomeostaticTaylorGluCore(hidden_dim=hidden_dim) for _ in range(num_experts)
        ]

    def route_and_dispatch(
        self, 
        hidden_states: jax.Array, 
        gating_logits: jax.Array,
        mesh: jax.sharding.Mesh = None
    ) -> jax.Array:
        """
        [CORE MOE PARALLEL DISPATCHER]
        Shape pipeline: [NumTokens, Dim] -> [NumExperts, ExpertCapacity, Dim] -> [NumTokens, Dim]
        """
        num_tokens, dim = hidden_states.shape
        
        # 1. 게이팅 확률 계산 및 Top-2 전문가 추출 (기존 뼈대 연산)
        routing_weights = jax.nn.softmax(gating_logits, axis=-1)
        scores, expert_indices = jax.lax.top_k(routing_weights, k=2) # [NumTokens, 2]
        
        # 가중치 정규화 (Top-2 전문가 스코어 합이 1이 되도록 조정)
        scores = scores / jnp.sum(scores, axis=-1, keepdims=True)

        # ------------------------------------------------------------------------
        # [COMPILER FENCE: HARDWARE FREQUENCY INDEXING FOR STATIC EXPANSION]
        # ------------------------------------------------------------------------
        # 각 토큰이 선택한 전문가 내에서 몇 번째 자리에 위치할지(인덱스)를 동적 루프 없이 계산합니다.
        # jnp.equal을 통해 토큰-전문가 매칭 2D 마스크를 만들고, 누적 합(cumsum)으로 슬롯 위치를 정적 락온합니다.
        token_indices = jnp.arange(num_tokens)[:, None]  # [NumTokens, 1]
        
        # 각 토큰별 Top-2 선택을 원-핫 형태의 매트릭스로 변환: [NumExperts, NumTokens]
        expert_mask_1 = (expert_indices[:, 0:1] == jnp.arange(self.num_experts)).T
        expert_mask_2 = (expert_indices[:, 1:2] == jnp.arange(self.num_experts)).T
        combined_expert_mask = expert_mask_1 | expert_mask_2  # [NumExperts, NumTokens]

        # 각 전문가별로 몇 번째 토큰으로 들어오는지 인라인 포지션 계산 (CumSum 활용)
        position_in_expert = jnp.cumsum(combined_expert_mask, axis=1) - 1
        
        # [Capacity Firewall] 지정된 expert_capacity를 초과하는 토큰은 마스킹하여 드롭(Token Dropping)
        valid_slot_mask = position_in_expert < self.expert_capacity
        final_gate_mask = combined_expert_mask & valid_slot_mask  # 최종 정적 활성화 마스크

        # 2. [Zero-Recompilation Gate] 정적 3D 텐서 그릇 pre-allocation: [NumExperts, Capacity, Dim]
        # 어떤 전문가가 토큰을 하나도 받지 못하더라도, XLA 컴파일러는 항상 이 고정된 텐서 형태만 보게 됩니다.
        static_expert_vessel = jnp.zeros((self.num_experts, self.expert_capacity, dim), dtype=hidden_states.dtype)

        # 각 전문가의 정적 위치 주소 매핑 생성
        expert_ids = jnp.arange(self.num_experts)[:, None]  # [NumExperts, 1]
        
        # 3. Scatter 연산을 통해 무작위로 흐르는 토큰들을 고정된 전문가 그릇에 물리적으로 정렬 분배
        # 정적 마스크 조건에 맞는 토큰들만 정적 그릇의 지정된 슬롯([Expert_ID, Slot_Position])에 박아 넣습니다.
        for exp_id in range(self.num_experts):
            # 현재 전문가에 할당된 토큰들의 인덱스 필터링 (분기문 없는 jnp.where 사용)
            token_select_mask = final_gate_mask[exp_id]
            src_indices = jnp.where(token_select_mask, jnp.arange(num_tokens), num_tokens - 1)
            target_slots = jnp.where(token_select_mask, position_in_expert[exp_id], 0)
            
            # Gather & 정적 그릇 채우기
            extracted_tokens = jnp.take(hidden_states, src_indices, axis=0)
            # 조건에 맞지 않는 빈 자리는 0(Padding)으로 유지한 채, 유효 토큰들만 제자리에 꽂음
            static_expert_vessel = static_expert_vessel.at[exp_id, target_slots].set(extracted_tokens)

        # 4. [COMPILER RUNTIME] 완성된 정적 그릇들을 기존 TaylorGLU 연산 레일에 통과
        # 연산 경로가 고정되어 있으므로, 8개의 전문가 FFN 블록이 동기화 락 없이 가속기 레일에서 질주합니다.
        expert_outputs = []
        for exp_id in range(self.num_experts):
            # 입력 스트림과 게이트 스트림이 분리되는 구조를 가정하여 내부 FMA 유닛 구동
            current_input = static_expert_vessel[exp_id]
            # 기존 bypass_rectifiers/taylor_glu.py 핵심 수식 원형 그대로 재사용
            out_processed = self.experts[exp_id](gate_stream=current_input, up_stream=current_input, mesh=mesh)
            expert_outputs.append(out_processed)
            
        expert_outputs = jnp.stack(expert_outputs, axis=0)  # [NumExperts, ExpertCapacity, Dim]

        # 5. [RESTORATION HIGHWAY] 각 가속기 그릇에서 연산된 결과를 원래 토큰 스트림 형태로 복원(Gather)
        # Top-2 스코어(가중치)를 결합하여 최종 아웃풋 스트림을 선형 결합 구조로 조립합니다.
        combined_output = jnp.zeros_like(hidden_states)
        
        # 첫 번째 가중 전문가 결과 누적
        target_slots_1 = position_in_expert[expert_indices[:, 0], jnp.arange(num_tokens)]
        gate_valid_1 = final_gate_mask[expert_indices[:, 0], jnp.arange(num_tokens)]
        exp_out_1 = expert_outputs[expert_indices[:, 0], target_slots_1]
        combined_output += jnp.where(gate_valid_1[:, None], exp_out_1 * scores[:, 0:1], 0.0)

        # 두 번째 가중 전문가 결과 누적
        target_slots_2 = position_in_expert[expert_indices[:, 1], jnp.arange(num_tokens)]
        gate_valid_2 = final_gate_mask[expert_indices[:, 1], jnp.arange(num_tokens)]
        exp_out_2 = expert_outputs[expert_indices[:, 1], target_slots_2]
        combined_output += jnp.where(gate_valid_2[:, None], exp_out_2 * scores[:, 1:2], 0.0)

        return combined_output
