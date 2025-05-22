FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 기본 패키지 설치
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# PyTorch + torchvision (GPU 버전)
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 작업 디렉토리 생성
WORKDIR /app

# 벤치마크 스크립트 복사
COPY resnet50_benchmark.py .

# 실행 커맨드
CMD ["python3", "resnet50_benchmark.py"]