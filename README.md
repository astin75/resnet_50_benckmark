# resnet_50_benckmark

# 빌드
docker build -t resnet50-benchmark .

# 실행 (GPU 사용)
docker run --rm --gpus all resnet50-benchmark