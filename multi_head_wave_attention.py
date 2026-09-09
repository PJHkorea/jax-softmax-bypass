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
