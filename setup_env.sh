#!/bin/bash
# setup_env.sh: jax-softmax-bypass 청정 실리콘 개발 환경 정류 스크립트

# 스크립트 주행 도중 단 1개의 명령어라도 에러(Non-zero)를 뱉으면 즉시 파이프라인을 동결 가드합니다.
set -e

echo "🌌 [인프라] 호스트 OS 베어메탈 가속 가드레일 수립 시작..."

# 1. 아키텍처 충돌 방지를 위한 Conda 독립 가상 환경 개통
if ! conda info --envs | grep -q "wave_engine"; then
    echo " ├─ [CONDA] 신규 가상 환경 'wave_engine' 생성 중..."
    conda create -n wave_engine python=3.10 -y
else
    echo " ├─ [CONDA] 이미 'wave_engine' 환경이 존재합니다. 인프라 재정류 주입을 시작합니다."
fi

# 서브쉘 환경에서 콘다 명령어 메커니즘을 완벽히 인식시키기 위한 내부 셸 훅 인터록 체결
eval "$(conda shell.bash hook)"
conda activate wave_engine

echo " ├─ [FFI BRIDGE] 파이토치 CUDA 12.x 하드웨어 주소 레일 정밀 바인딩..."
# 🌟 고도화: 일반 pytorch.org 랜딩 주소가 아닌, 하드웨어 명령어 매핑을 유도할 cu124 바이너리 인덱스로 직결 개통
pip install torch --index-url https://pytorch.org

echo " ├─ [XLA CORE] 구글 JAX XLA 가속 엔진 및 전전후 드라이버 펜스 통합본 인입..."
pip install --upgrade "jax[cuda12]" -f https://googleapis.com

echo " ├─ [SERVING BACKBONE] 크로스 플랫폼 OS 커널 자원 가드레일 및 상용 서빙 백본 종속성 완공..."
# 🌟 고도화: 실실론 서빙 주행을 위한 vllm 무기까지 단선 없이 한 통문으로 융합 탑재
pip install psutil transformers fastapi pydantic vllm

echo "🎉 [SUCCESS] 4대 브랜치리스 통합 체인을 구동할 청정 전산 파드가 영구 수립되었습니다."
print("👉 'conda activate wave_engine' 명령어를 터미널에 입력하여 가속 레일에 탑승하십시오.")
