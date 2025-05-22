import torch
import torchvision.models as models
import time
import gc

def check_cuda():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit(1)
    print(f"Using device: {torch.cuda.get_device_name(0)}")

def get_max_batch_size(model, image_size=(3, 224, 224), device='cuda', dtype=torch.float32):
    batch_size = 1
    max_batch_size = 1
    while True:
        try:
            dummy_input = torch.randn(batch_size, *image_size, device=device, dtype=dtype)
            with torch.no_grad():
                _ = model(dummy_input)
            max_batch_size = batch_size
            batch_size *= 2
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                break
            else:
                raise e
    return max_batch_size

def benchmark(model, batch_size, image_size=(3, 224, 224), num_iters=50, device='cuda', dtype=torch.float32):
    dummy_input = torch.randn(batch_size, *image_size, device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        # warm-up
        for _ in range(10):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()
    avg_time = (end - start) / num_iters
    throughput = batch_size / avg_time
    return avg_time, throughput

def run_test(dtype, precision_name):
    print(f"\n===== Testing {precision_name} =====")
    model = models.resnet50(pretrained=False).to(device).to(dtype)
    max_batch = get_max_batch_size(model, device=device, dtype=dtype)
    print(f"[{precision_name}] Max batch size: {max_batch}")
    avg_time, throughput = benchmark(model, max_batch, device=device, dtype=dtype)
    print(f"[{precision_name}] Avg Inference Time: {avg_time:.6f} sec")
    print(f"[{precision_name}] Throughput: {throughput:.2f} images/sec")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_cuda()

    # FP32
    run_test(dtype=torch.float32, precision_name="FP32")

    # FP16 (AMP)
    print("\nNote: FP16 uses torch.cuda.amp for automatic mixed precision")
    model_fp16 = models.resnet50(pretrained=False).to(device)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    model_fp16.eval()

    # Find max batch size for AMP
    batch_size = 1
    max_batch_size = 1
    while True:
        try:
            dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
            with torch.cuda.amp.autocast():
                _ = model_fp16(dummy_input)
            max_batch_size = batch_size
            batch_size *= 2
            gc.collect()
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                break
            else:
                raise e

    print(f"[FP16] Max batch size: {max_batch_size}")
    dummy_input = torch.randn(max_batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        # warm-up
        for _ in range(10):
            with torch.cuda.amp.autocast():
                _ = model_fp16(dummy_input)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(50):
            with torch.cuda.amp.autocast():
                _ = model_fp16(dummy_input)
        torch.cuda.synchronize()
        end = time.time()

    avg_time = (end - start) / 50
    throughput = max_batch_size / avg_time
    print(f"[FP16] Avg Inference Time: {avg_time:.6f} sec")
    print(f"[FP16] Throughput: {throughput:.2f} images/sec")