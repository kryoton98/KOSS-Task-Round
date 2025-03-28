import numpy as np
import cupy as cp
import timeit
import matplotlib.pyplot as plt

# ---------------------------
# CPU Implementations
# ---------------------------
def naive_matmul(A, B):
    m, n = A.shape
    _, p = B.shape
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

def numpy_matmul(A, B):
    return A @ B

# ---------------------------
# GPU Implementations
# ---------------------------
def gpu_matmul_fp32(A_gpu, B_gpu):
    return A_gpu @ B_gpu

def gpu_matmul_tensorcore(A_gpu, B_gpu):
    # Use tensor cores by using float16 and mixed precision
    with cp.cuda.Device(0):
        return cp.matmul(A_gpu.astype(cp.float16), 
                         B_gpu.astype(cp.float16)).astype(cp.float32)

# ---------------------------
# Benchmark Utilities
# ---------------------------
def benchmark(func, A, B, device='cpu', warmup=3, repeats=5):
    times = []
    
    if device == 'gpu':
        stream = cp.cuda.Stream()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        # Warmup
        for _ in range(warmup):
            _ = func(A, B)
            stream.synchronize()
        
        # Timing
        for _ in range(repeats):
            start_event.record(stream)
            _ = func(A, B)
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

def create_matrices(size, dtype=np.float32, device='cpu'):
    if device == 'gpu':
        return (cp.random.rand(size, size).astype(dtype),
                cp.random.rand(size, size).astype(dtype))
    return (np.random.rand(size, size).astype(dtype),
            np.random.rand(size, size).astype(dtype))

# ---------------------------
# Visualization Utilities
# ---------------------------
def plot_scaling(results_cpu, results_gpu_fp32, results_gpu_tc):
    sizes = [r[0] for r in results_cpu]
    
    cpu_times = [r[1] for r in results_cpu]
    gpu_fp32_times = [r[1] for r in results_gpu_fp32]
    gpu_tc_times = [r[1] for r in results_gpu_tc]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, label="NumPy (CPU)", marker="o")
    plt.plot(sizes, gpu_fp32_times, label="CuPy FP32 (GPU)", marker="o")
    plt.plot(sizes, gpu_tc_times, label="CuPy TensorCore (GPU)", marker="o")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time Scaling Across Matrix Sizes")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_speedup(results_cpu, results_gpu_fp32, results_gpu_tc):
    sizes = [r[0] for r in results_cpu]
    
    cpu_times = [r[1] for r in results_cpu]
    gpu_fp32_speedup = [cpu_times[i] / r[1] for i, r in enumerate(results_gpu_fp32)]
    gpu_tc_speedup = [cpu_times[i] / r[1] for i, r in enumerate(results_gpu_tc)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, gpu_fp32_speedup, label="CuPy FP32 Speedup", marker="o")
    plt.plot(sizes, gpu_tc_speedup, label="CuPy TensorCore Speedup", marker="o")
    
    plt.xscale("log")
    plt.xlabel("Matrix Size")
    plt.ylabel("Speedup Factor vs NumPy (CPU)")
    plt.title("Speedup Across Matrix Sizes")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------------------
# Main Analysis Script
# ---------------------------
def main():
    sizes = [512, 1024, 2048]  # Matrix sizes to test
    dtype_cpu = np.float32      # Data type for CPU matrices
    dtype_gpu = cp.float32      # Data type for GPU matrices
    
    results_cpu = []
    results_gpu_fp32 = []
    results_gpu_tc = []
    
    print("\nBenchmarking Matrix Multiplication:")
    
    for size in sizes:
        print(f"\nTesting {size}x{size} matrices...")
        
        # Create matrices on CPU and GPU
        A_cpu, B_cpu = create_matrices(size=size, dtype=dtype_cpu)
        A_gpu_fp32, B_gpu_fp32 = create_matrices(size=size, dtype=dtype_gpu, device='gpu')
        
        # Benchmark CPU methods
        numpy_time_best = benchmark(numpy_matmul,
                                    A=A_cpu,
                                    B=B_cpu,
                                    device='cpu')[0]
        
        # Benchmark GPU methods (FP32 and TensorCore)
        gpu_fp32_time_best = benchmark(gpu_matmul_fp32,
                                       A=A_gpu_fp32,
                                       B=B_gpu_fp32,
                                       device='gpu')[0]
        
        gpu_tc_time_best = benchmark(gpu_matmul_tensorcore,
                                     A=A_gpu_fp32,
                                     B=B_gpu_fp32,
                                     device='gpu')[0]
        
        # Store results
        results_cpu.append((size, numpy_time_best))
        results_gpu_fp32.append((size, gpu_fp32_time_best))
        results_gpu_tc.append((size, gpu_tc_time_best))
        
        print(f"NumPy (CPU): {numpy_time_best:.6f}s")
        print(f"CuPy FP32 (GPU): {gpu_fp32_time_best:.6f}s")
        print(f"CuPy TensorCore (GPU): {gpu_tc_time_best:.6f}s")
    
    # Plot scaling and speedups
    plot_scaling(results_cpu=results_cpu,
                 results_gpu_fp32=results_gpu_fp32,
                 results_gpu_tc=results_gpu_tc)
    
    plot_speedup(results_cpu=results_cpu,
                 results_gpu_fp32=results_gpu_fp32,
                 results_gpu_tc=results_gpu_tc)

if __name__ == "__main__":
    main()
