import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
import time
import gc
import subprocess

def check_cuda():
    """CUDA 사용 가능 여부 확인"""
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다. 프로그램을 종료합니다.")
        exit(1)
    print(f"사용 중인 디바이스: {torch.cuda.get_device_name(0)}")

def print_gpu_memory_status(test_name):
    """nvidia-smi를 사용하여 GPU 메모리 상태 출력"""
    print(f"\n===== {test_name} 시작 전 GPU 상태 =====")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi 실행 중 오류 발생: {e}")
    except FileNotFoundError:
        print("nvidia-smi를 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있는지 확인하세요.")
    print("=" * 60)

def get_max_batch_size_fp32(model, processor, image_size=(3, 224, 224), device='cuda', dtype=torch.float32):
    """FP32에 대한 최대 배치 사이즈 찾기"""
    batch_size = 1
    max_batch_size = 1
    
    while True:
        try:
            # Transformers 모델용 입력 생성
            dummy_input = torch.randn(batch_size, *image_size, device=device, dtype=dtype)
            with torch.no_grad():
                outputs = model(pixel_values=dummy_input)
            max_batch_size = batch_size
            batch_size *= 2
            print(f"배치 사이즈 테스트: {batch_size}")
            
            # 메모리 정리
            del dummy_input, outputs
            gc.collect()
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                break
            else:
                raise e
    
    return max_batch_size

def get_max_batch_size_fp16(model, processor, image_size=(3, 224, 224), device='cuda'):
    """FP16 AMP에 대한 최대 배치 사이즈 찾기"""
    batch_size = 1
    max_batch_size = 1
    
    while True:
        try:
            dummy_input = torch.randn(batch_size, *image_size, device=device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=dummy_input)
            max_batch_size = batch_size
            batch_size *= 2
            print(f"배치 사이즈 테스트 (FP16): {batch_size}")
            
            # 메모리 정리
            del dummy_input, outputs
            gc.collect()
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                break
            else:
                raise e
    
    return max_batch_size

def benchmark_fp32(model, processor, batch_size, image_size=(3, 224, 224), num_iters=50, device='cuda', dtype=torch.float32):
    """FP32 벤치마크 실행"""
    dummy_input = torch.randn(batch_size, *image_size, device=device, dtype=dtype)
    model.eval()
    
    with torch.no_grad():
        # 워밍업
        for _ in range(10):
            outputs = model(pixel_values=dummy_input)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_iters):
            outputs = model(pixel_values=dummy_input)
        
        torch.cuda.synchronize()
        end = time.time()
    
    avg_time = (end - start) / num_iters
    throughput = batch_size / avg_time
    return avg_time, throughput

def benchmark_fp16(model, processor, batch_size, image_size=(3, 224, 224), num_iters=50, device='cuda'):
    """FP16 AMP 벤치마크 실행"""
    dummy_input = torch.randn(batch_size, *image_size, device=device)
    model.eval()
    
    with torch.no_grad():
        # 워밍업
        for _ in range(10):
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=dummy_input)
        
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(num_iters):
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=dummy_input)
        
        torch.cuda.synchronize()
        end = time.time()
    
    avg_time = (end - start) / num_iters
    throughput = batch_size / avg_time
    return avg_time, throughput

def run_fp32_test(device='cuda'):
    """FP32 테스트 실행"""
    print_gpu_memory_status("FP32 테스트")
    print(f"\n===== FP32 테스트 =====")
    
    # Transformers 모델 및 프로세서 로드
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        torch_dtype=torch.float32
    ).to(device)
    
    # 최대 배치 사이즈 찾기
    max_batch = get_max_batch_size_fp32(model, processor, device=device, dtype=torch.float32)
    print(f"[FP32] 최대 배치 사이즈: {max_batch}")
    
    # 벤치마크 실행
    avg_time, throughput = benchmark_fp32(model, processor, max_batch, device=device, dtype=torch.float32)
    print(f"[FP32] 평균 추론 시간: {avg_time:.6f} 초")
    print(f"[FP32] 처리량: {throughput:.2f} images/sec")
    
    # 메모리 정리
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

def run_fp16_test(device='cuda'):
    """FP16 AMP 테스트 실행"""
    print_gpu_memory_status("FP16 테스트")
    print(f"\n===== FP16 (AMP) 테스트 =====")
    print("주의: FP16은 torch.cuda.amp를 사용한 자동 혼합 정밀도입니다")
    
    # Transformers 모델 및 프로세서 로드
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained(
        "microsoft/resnet-50",
        torch_dtype=torch.float16
    ).to(device)
    
    # 최대 배치 사이즈 찾기
    max_batch = get_max_batch_size_fp16(model, processor, device=device)
    print(f"[FP16] 최대 배치 사이즈: {max_batch}")
    
    # 벤치마크 실행
    avg_time, throughput = benchmark_fp16(model, processor, max_batch, device=device)
    print(f"[FP16] 평균 추론 시간: {avg_time:.6f} 초")
    print(f"[FP16] 처리량: {throughput:.2f} images/sec")
    
    # 메모리 정리
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

def main():
    """메인 실행 함수"""
    # CUDNN 벤치마크 모드 활성화
    torch.backends.cudnn.benchmark = True
    
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_cuda()
    
    # 테스트 실행
    run_fp32_test(device)
    run_fp16_test(device)

if __name__ == "__main__":
    main()