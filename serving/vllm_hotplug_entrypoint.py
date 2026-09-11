# -*- coding: utf-8 -*-
"""File: serving/vllm_hotplug_entrypoint.py"""
from fastapi import FastAPI
from wave_attention_hijacker_core import patch_llama_model_with_wave_attention
from serving.cluster_bootstrap import auto_bootstrap_hardware_mesh

def initialize_wave_serving_engine(pytorch_vllm_model):
    """vLLM 내부 가중치 로드 직후 파동 하이재커를 심장에 결착합니다."""
    # 1. 분산 실리콘 메시 전역 활성화
    global_mesh = auto_bootstrap_hardware_mesh()
    
    # 2. 레거시 어텐션을 복사 없이 파동 레일로 강제 대체 (0ns 몽키 패치)
    hijacked_serving_model = patch_llama_model_with_wave_attention(
        pytorch_vllm_model, 
        mesh_shape=64, 
        alpha=0.01
    )
    
    # 3. 전역 메시 주소선을 모델 인스턴스에 영구 바인딩
    hijacked_serving_model.global_hardware_mesh = global_mesh
    return hijacked_serving_model
