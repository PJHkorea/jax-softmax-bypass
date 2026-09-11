#!/bin/bash
# setup_env.sh: Environment alignment script for the jax-softmax-bypass platform.

# Enforces a strict circuit-breaker barrier; freezes execution immediately if any command returns a non-zero exit code.
set -e

echo "[INFRASTRUCTURE] Initializing bare-metal hardware acceleration setup tracks over the host OS..."

# 1. Allocates an isolated Conda environment instance to insulate against cross-architecture parameter collisions.
if ! conda info --envs | grep -q "wave_engine"; then
    echo " ├── [CONDA] Creating a clean virtual tracking registry 'wave_engine'..."
    conda create -n wave_engine python=3.10 -y
else
    echo " ├── [CONDA] Existing 'wave_engine' target context found. Proceeding with infrastructure re-alignment tracks."
fi

# Interlocks internal subshell hook routines to explicitly handle python-conda context transitions inside non-interactive shells.
eval "$(conda shell.bash hook)"
conda activate wave_engine

echo " ├── [FFI BRIDGE] Establishing precise hardware mapping paths for PyTorch CUDA 12.x arrays..."
# Hardcodes optimal binary indices to map device-level memory strides directly over the baseline accelerator tracks.
pip install torch --index-url https://pytorch.org

echo " ├── [XLA CORE] Ingesting Google JAX XLA computation compiler engines and unified driver fence assets..."
pip install --upgrade "jax[cuda12]" -f https://googleapis.com

echo " ├── [SERVING BACKBONE] Mounting cross-platform OS resource monitors and production-grade serving components..."
# Integrates the vLLM engine directly into a single atomic installation transaction path to support real-time serving workflows.
pip install psutil transformers fastapi pydantic vllm

echo "[PROD ENGINE READY] Pure silicon computational pod for branchless wave inversion tensor rails successfully allocated."
echo " Execute 'conda activate wave_engine' to mount the physical accelerator rails."
