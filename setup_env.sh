#!/bin/bash
# setup_env.sh: jax-softmax-bypass 청정 실리콘 개발 환경 정류 스크립트

echo "🌌 [인프라] 호스트 OS 베어메탈 가속 가드레일 수립 시작..."

# 1. 아키텍처 충돌 방지를 위한 Conda 독립 가상 환경 개통
conda create -n wave_engine python=3.10 -y
source activate wave_engine

# 2. 파이토치 CUDA 12.x 하드웨어 주소 레일 바인딩
pip install torch --index-url https://pytorch.org

# 3. 구글 JAX XLA 가속 엔진 및 전전후 드라이버 펜스 통합본 인입
pip install --upgrade "jax[cuda12]" -f https://googleapis.com

# 4. 크로스 플랫폼 OS 커널 자원 가드레일 및 상용 서빙 백본 종속성 완공
pip install psutil transformers fastapi pydantic

echo "🎉 [SUCCESS] 4대 브랜치리스 통합 체인을 구동할 청정 전산 파드가 영구 수립되었습니다."
