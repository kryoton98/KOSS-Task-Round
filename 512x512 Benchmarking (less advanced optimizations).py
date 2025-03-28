import numpy as np
import timeit
from numba import njit, prange
import sys

# ---------------------------
# 1. Naive Python Implementation
# ---------------------------
def matmul_naive(A, B):
    m, n = A.shape
    _, p = B.shape
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]
    return C

# ---------------------------
# 2. Numba-optimized Versions
# ---------------------------
@njit
def matmul_numba(A, B):
    m, n = A.shape
    _, p = B.shape
    C = np.zeros((m, p))
    for i in range(m):
        for k in range(n):
            a = A[i,k]
            for j in range(p):
                C[i,j] += a * B[k,j]
    return C

@njit(parallel=True, fastmath=True)
def matmul_numba_parallel(A, B):
    m, n = A.shape
    _, p = B.shape
    C = np.zeros((m, p))
    for i in prange(m):
        for k in range(n):
            a = A[i,k]
            for j in range(p):
                C[i,j] += a * B[k,j]
    return C

# ---------------------------
# 3. Blocked Matrix Multiplication
# ---------------------------
def matmul_blocked(A, B, block_size=64):
    m, n = A.shape
    _, p = B.shape
    C = np.zeros((m, p))
    
    for i in range(0, m, block_size):
        for j in range(0, p, block_size):
            for k in range(0, n, block_size):
                ii = min(i+block_size, m)
                jj = min(j+block_size, p)
                kk = min(k+block_size, n)
                
                C[i:ii, j:jj] += A[i:ii, k:kk] @ B[k:kk, j:jj]
    return C

# ---------------------------
# 4. Strassen's Algorithm
# ---------------------------
def strassen(A, B, leaf_size=128):
    n = A.shape[0]
    
    if n <= leaf_size:
        return A @ B
    
    n2 = n // 2
    A11, A12 = A[:n2, :n2], A[:n2, n2:]
    A21, A22 = A[n2:, :n2], A[n2:, n2:]
    
    B11, B12 = B[:n2, :n2], B[:n2, n2:]
    B21, B22 = B[n2:, :n2], B[n2:, n2:]
    
    P1 = strassen(A11, B12 - B22)
    P2 = strassen(A11 + A12, B22)
    P3 = strassen(A21 + A22, B11)
    P4 = strassen(A22, B21 - B11)
    P5 = strassen(A11 + A22, B11 + B22)
    P6 = strassen(A12 - A22, B21 + B22)
    P7 = strassen(A11 - A21, B11 + B12)
    
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P1 + P5 - P3 - P7
    
    return np.vstack((np.hstack((C11, C12)),
                     np.hstack((C21, C22))))

# ---------------------------
# 5. GPU Implementation (CuPy)
# ---------------------------
try:
    import cupy as cp
    def matmul_gpu(A, B):
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        C_gpu = A_gpu @ B_gpu
        return cp.asnumpy(C_gpu)
except ImportError:
    print("CuPy not installed, GPU support disabled")

# ---------------------------
# Verification & Benchmarking
# ---------------------------
def verify_implementation(impl_fn, A, B):
    C_numpy = A @ B
    C_impl = impl_fn(A, B)
    return np.allclose(C_numpy, C_impl, rtol=1e-5, atol=1e-8)

def benchmark(impl_fn, A, B, num_runs=5):
    timer = timeit.Timer(lambda: impl_fn(A, B))
    times = timer.repeat(number=1, repeat=num_runs)
    return min(times), max(times), np.mean(times)

if __name__ == "__main__":
    np.random.seed(42)
    size = 512  # Start with 512x512 matrices
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    implementations = {
        "NumPy": lambda a,b: a @ b,
        "Naive Python": matmul_naive,
        "Numba": matmul_numba,
        "Numba Parallel": matmul_numba_parallel,
        "Blocked (64)": lambda a,b: matmul_blocked(a,b,64),
        "Strassen": lambda a,b: strassen(a,b,128),
    }
    
    if 'matmul_gpu' in globals():
        implementations["CuPy GPU"] = matmul_gpu

    # Warm-up runs for JIT compilation
    if size >= 256:
        matmul_numba(A.copy(), B.copy())
        matmul_numba_parallel(A.copy(), B.copy())

    print(f"Benchmarking {size}x{size} matrix multiplication:")
    print("{:<15} {:<10} {:<10} {:<10} {:<10}".format(
        "Method", "Min(sec)", "Max(sec)", "Avg(sec)", "Valid"))
    
    results = []
    for name, fn in implementations.items():
        try:
            valid = verify_implementation(fn, A, B)
            min_t, max_t, avg_t = benchmark(fn, A, B)
            results.append((name, min_t, max_t, avg_t, valid))
        except Exception as e:
            print(f"Error in {name}: {str(e)}")
    
    # Sort results by minimum time
    results.sort(key=lambda x: x[1])
    
    for name, min_t, max_t, avg_t, valid in results:
        print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}".format(
            name, min_t, max_t, avg_t, str(valid)))

    # Print performance comparison
    numpy_time = next(r[1] for r in results if r[0] == "NumPy")
    print("\nSpeedup vs NumPy:")
    for name, min_t, *_ in results:
        if name != "NumPy":
            print(f"{name:<15} {numpy_time/min_t:.1f}x")
