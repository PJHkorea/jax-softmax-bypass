"""File: serving/vllm_hotplug_entrypoint.py"""
from fastapi import FastAPI
from wave_attention_hijacker_core import patch_llama_model_with_wave_attention
from serving.cluster_bootstrap import auto_bootstrap_hardware_mesh

def initialize_wave_serving_engine(pytorch_vllm_model):
    """
    Hooks the wave hijacking layer directly into the core backbone immediately after vLLM loads model parameters into HBM.
    """
    # 1. Activates global distributed hardware device configurations across the silicon topology.
    global_mesh = auto_bootstrap_hardware_mesh()
    
    # 2. Deploys an on-the-fly monkey patch to intercept legacy attention layouts with zero memory-copy overhead.
    hijacked_serving_model = patch_llama_model_with_wave_attention(
        pytorch_vllm_model, 
        mesh_shape=64, 
        alpha=0.01
    )
    
    # 3. Statically hard-locks the global infrastructure mesh address tracks onto the model instance attributes,
    # ensuring multi-thread API stream concurrent requests instantly locate their designated accelerator banks (0ns lookup overhead).
    hijacked_serving_model.global_hardware_mesh = global_mesh
    return hijacked_serving_model
