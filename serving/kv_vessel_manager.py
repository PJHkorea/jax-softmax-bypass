# -*- coding: utf-8 -*-
"""File: serving/kv_vessel_manager.py"""
import jax
import jax.numpy as jnp

class WaveKVCacheContainer:
    """VRAM 폭발을 영구 차단하기 위해 고정 차원 평면 상에 상태 전하량을 적산하는 캐시 용기."""
    def __init__(self, batch_size: int, num_heads: int, mesh_shape: int, head_dim: int, dtype=jnp.float16):
        # 문맥 길이가 10만이 되든 100만이 되든 이 텐서의 크기는 영구히 변하지 않습니다.
        self.vessel = jnp.zeros((batch_size, num_heads, mesh_shape, head_dim), dtype=dtype)
        
    def update_vessel(self, new_k_wave: jax.Array, new_v_h: jax.Array) -> jax.Array:
        """새로 유입된 싱글 토큰의 파동 간섭 변량을 글로벌 용기에 누적 가산합니다."""
        # new_k_wave: [B, H, M, 1], new_v_h: [B, H, 1, D] -> delta: [B, H, M, D]
        delta_vessel = jnp.matmul(new_k_wave, new_v_h)
        self.vessel = self.vessel + delta_vessel
        return self.vessel
