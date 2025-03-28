# 🚀 **KOSS Task Round: Optimizing Matrix Multiplication**

## 📌 **Overview**
This repository contains the complete code and resources for benchmarking matrix multiplication on both CPU and GPU, using various implementations and optimizations. The project demonstrates the performance benefits of GPU acceleration using **CuPy** with **FP32** and **TensorCore** support, alongside CPU-based approaches with **NumPy**, **Numba**, and custom CUDA kernels.


---

## ⚙️ **Features**
- **CPU Implementations:**  
  - Naïve Python multiplication  
  - NumPy-based matrix multiplication  
  - Numba parallel-accelerated multiplication  

- **GPU Implementations:**  
  - **CuPy FP32:** Optimized with standard floating-point precision  
  - **CuPy TensorCore:** Faster, mixed-precision matrix multiplication  
  - Custom CUDA kernel for small matrices with shared memory optimization  

- **Performance Benchmarking:**  
  - Execution time, throughput, and speedup measurements  
  - Comparison against NumPy baseline performance  

- **Visualizations:**  
  - Logarithmic scale performance graphs  
  - Speedup and time comparisons  

---

## 🚦 **How to Run**

1. **Install dependencies:**  
Ensure you have the required libraries installed. Use the following command:  
```bash
pip install numpy numba cupy matplotlib
```

## 🔧 **Customization**
- **Modify matrix sizes** and block dimensions in the scripts for different benchmarks.  
- **Tune the CUDA kernel parameters** to optimize performance for specific hardware.  
- **Experiment with different matrix sizes** to observe scaling behavior.  

## ✅ **Contribute**
Feel free to contribute by suggesting further optimizations, adding new algorithms, or improving the visualizations. 🚀

## 📚 **References**
- [CuPy Documentation](https://docs.cupy.dev/)  
- [Numba Documentation](https://numba.readthedocs.io/)  
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
