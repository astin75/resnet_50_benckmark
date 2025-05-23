# resnet_50_benckmark

# 빌드
docker build -t resnet50-benchmark .

# 실행 (GPU 사용)
docker run --rm --gpus all resnet50-benchmark

# docker push
docker tag resnet50-benchmark astin75/resnet50-benchmark:latest
docker push astin75/resnet50-benchmark:latest