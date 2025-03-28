import numpy as np
import cupy as cp
import timeit
from numba import njit, prange
import argparse

# ---------------------------
# CPU Implementations
# ---------------------------
@njit(parallel=True, fastmath=True)
def numba_matmul(A, B):
    m, n = A.shape
    _, p = B.shape
    C = np.zeros((m, p))
    for i in prange(m):
        for k in range(n):
            a = A[i, k]
            for j in range(p):
                C[i, j] += a * B[k, j]
    return C

def numpy_matmul(A, B):
    return A @ B

# ---------------------------
# GPU Implementations
# ---------------------------
def gpu_matmul_fp32(A, B, stream):
    with stream:
        return A @ B

def gpu_matmul_tensorcore(A, B, stream):
    with stream:  # Use TensorCores with mixed precision
        A_fp16 = A.astype(cp.float16)
        B_fp16 = B.astype(cp.float16)
        result_fp16 = cp.matmul(A_fp16, B_fp16)
        return result_fp16.astype(cp.float32)

# ---------------------------
# Benchmark Utilities
# ---------------------------
def create_matrices(size, dtype, device='cpu'):
    if device == 'gpu':
        return (cp.random.rand(size, size).astype(dtype),
                cp.random.rand(size, size).astype(dtype))
    return (np.random.rand(size, size).astype(dtype),
            np.random.rand(size, size).astype(dtype))

def benchmark(func, A, B, device='cpu', warmup=3, repeats=5):
    times = []
    
    if device == 'gpu':
        stream = cp.cuda.Stream()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        # Warmup
        for _ in range(warmup):
            _ = func(A, B, stream)
            stream.synchronize()
        
        # Timing
        for _ in range(repeats):
            start_event.record(stream)
            _ = func(A, B, stream)
            end_event.record(stream)
            stream.synchronize()
            times.append(cp.cuda.get_elapsed_time(start_event, end_event) * 1e-3)
    else:
        # Warmup
        for _ in range(warmup):
            _ = func(A, B)
        
        # Timing
        timer = timeit.Timer(lambda: func(A, B))
        times = timer.repeat(number=1, repeat=repeats)
    
    return min(times), np.median(times), max(times)

# ---------------------------
# Main Benchmark
# ---------------------------
def main(size=2048):
    print(f"Benchmarking {size}x{size} matrices on RTX 4050")
    print("Initializing matrices...")
    
    # CPU matrices
    A_cpu, B_cpu = create_matrices(size, np.float32, 'cpu')
    
    # GPU matrices (directly initialized on device)
    A_gpu_fp32, B_gpu_fp32 = create_matrices(size, cp.float32, 'gpu')
    A_gpu_tc, B_gpu_tc = create_matrices(size, cp.float32, 'gpu')
    
    implementations = [
        ('NumPy', numpy_matmul, (A_cpu, B_cpu), 'cpu'),
        ('Numba', numba_matmul, (A_cpu, B_cpu), 'cpu'),
        ('CuPy FP32', gpu_matmul_fp32, (A_gpu_fp32, B_gpu_fp32), 'gpu'),
        ('CuPy TensorCore', gpu_matmul_tensorcore, (A_gpu_tc, B_gpu_tc), 'gpu')
    ]
    
    print("\n{:<20} {:<12} {:<12} {:<12}".format(
        "Method", "Best(sec)", "Median(sec)", "Max(sec)"))
    
    results = []
    for name, func, mats, device in implementations:
        best, med, worst = benchmark(func, *mats, device=device)
        results.append((name, best, med, worst))
    
    # Sort by best time
    results.sort(key=lambda x: x[1])
    
    for name, best, med, worst in results:
        print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f}".format(name, best, med, worst))
    
    # Calculate speedup
    numpy_time = next(r[1] for r in results if r[0] == "NumPy")
    print("\nSpeedup vs NumPy:")
    for name, best, *_ in results:
        if name != "NumPy":
            print(f"{name:<20} {numpy_time/best:.1f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=2048)
    args = parser.parse_args()
    
    # Verify CUDA device
    print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"CUDA Compute Capability: {cp.cuda.runtime.getDeviceProperties(0)['major']}.{cp.cuda.runtime.getDeviceProperties(0)['minor']}")
    print(f"Total GPU Memory: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']/1e9:.1f} GB\n")
    
    main(args.size)
