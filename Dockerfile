FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# uv 설치 및 가상 환경 생성
RUN pip install uv
RUN uv venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# PyTorch + torchvision (GPU 버전)
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Transformers 및 관련 라이브러리 설치
RUN uv pip install transformers accelerate

# FlagEmbedding (BGE M3) 설치
RUN uv pip install -U FlagEmbedding
RUN uv pip install transformers==4.44.2

# 작업 디렉토리 생성
WORKDIR /app

# 벤치마크 스크립트 복사
COPY resnet50_benchmark.py .

# 실행 커맨드
CMD ["sh", "-c", "python resnet50_benchmark.py && exit 0"]