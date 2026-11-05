/*
 * Day 27: CUDA 库 — 站在巨人的肩膀上
 * ========================================
 * 学习目标:
 *   1. 了解 NVIDIA 提供的高性能 CUDA 库
 *   2. 使用 cuBLAS 进行矩阵运算
 *   3. 了解 cuDNN (深度学习)、cuFFT (傅里叶变换)、cuSPARSE (稀疏矩阵)
 *   4. 理解何时用库 vs 手写 kernel
 *
 * 主要 CUDA 库:
 *   - cuBLAS:  线性代数 (BLAS 3 级), 矩阵乘法等
 *   - cuDNN:   深度学习原语 (卷积, 池化, 激活函数)
 *   - cuFFT:   快速傅里叶变换
 *   - cuSPARSE: 稀疏矩阵运算
 *   - cuSOLVER: 线性方程组求解, 特征值
 *   - cuRAND:  随机数生成
 *   - Thrust:  C++ STL 风格的并行算法
 *
 * 编译: nvcc day27_cublas_cudnn.cu -o day27 -lcublas
 * 运行: ./day27
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define TILE 32
#define NAIVE_BLOCK 16

// 手写矩阵乘法 (用于对比) — 朴素版本用 16x16 block, 提高占用率
__global__ void matMulNaive(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

// Tiled 矩阵乘法
__global__ void matMulTiled(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        sA[threadIdx.y][threadIdx.x] = (row < M && t * TILE + threadIdx.x < K) ?
            A[row * K + t * TILE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (t * TILE + threadIdx.y < K && col < N) ?
            B[(t * TILE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; i++)
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads(); 
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// d_test 与 d_ref 拷回 host 对比, 打印 maxAbsDiff 与相对误差
void checkResult(const char *name, const float *d_ref, const float *d_test, int n) {
    float *h_ref = (float *)malloc((size_t)n * sizeof(float));
    float *h_tst = (float *)malloc((size_t)n * sizeof(float));
    cudaMemcpy(h_ref, d_ref, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tst, d_test,(size_t)n * sizeof(float), cudaMemcpyDeviceToHost);

    float maxErr = 0.0f, maxVal = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(h_ref[i] - h_tst[i]);
        if (e > maxErr) maxErr = e;
        if (fabsf(h_ref[i]) > maxVal) maxVal = fabsf(h_ref[i]);
    }
    float rel = (maxVal > 0.0f) ? (maxErr / maxVal) : 0.0f;
    const char *mark = (rel < 1e-2f) ? "✓" : "✗";
    printf("  %-18s maxAbsDiff=%.3e  相对误差=%.3e  %s\n",
           name, maxErr, rel, mark);
    
    free(h_ref); free(h_tst);
}

int main(){
    printf("============================================\n");
    printf("   Day 27: CUDA 库 (cuBLAS)\n");
    printf("============================================\n\n");

    int M = 1024, N = 1024, K = 1024;
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);
    printf("矩阵: %dx%d * %dx%d\n\n", M, K, K, N);

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMalloc(&d_C_ref, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // --- 先跑一次 naive 生成参考结果 d_C_ref (验证基准) ---
    {
        dim3 block(NAIVE_BLOCK, NAIVE_BLOCK);
        dim3 grid((N + NAIVE_BLOCK - 1) / NAIVE_BLOCK,
                  (M + NAIVE_BLOCK - 1) / NAIVE_BLOCK);
        matMulNaive<<<grid, block>>>(d_A, d_B, d_C_ref, M, N, K);
        cudaDeviceSynchronize();
    }

    float ms_naive = 0, ms_tiled = 0, ms_cublas = 0;
    float gf_naive = 0, gf_tiled = 0, gf_cublas = 0;

    // --- 朴素 GPU ---
    printf("--- 朴素矩阵乘法 ---\n");
    {
        dim3 block(NAIVE_BLOCK, NAIVE_BLOCK);
        dim3 grid((N + NAIVE_BLOCK - 1) / NAIVE_BLOCK,
                  (M + NAIVE_BLOCK - 1) / NAIVE_BLOCK);
        
        // 预热
        matMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 10; i++)
            matMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        
        ms_naive = ms / 10.0f;
        gf_naive = 2.0f * M * N * K / (ms_naive * 1e6f);
        printf("  耗时: %.3f ms, 性能: %.2f GFLOPS\n", ms_naive, gf_naive);
        checkResult("naive vs ref", d_C_ref, d_C, M * N);
        printf("\n");
    }

    // --- Tiled GPU ---
    printf("--- Tiled 矩阵乘法 ---\n");
    {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

        // 预热
        matMulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 10; i++)
            matMulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        ms_tiled = ms /10.0f;
        gf_tiled = 2.0f * M * N * K / (ms_tiled * 1e6f);
        printf("  耗时: %.3f ms, 性能: %.2f GFLOPS\n", ms_tiled, gf_tiled);
        checkResult("tiled vs ref", d_C_ref, d_C, M * N);
        printf("\n");
    }

    // --- cuBLAS ---
    // 注意: cuBLAS 使用列主序 (column-major), 与 C 的行主序不同
    // 技巧: C = A * B (行主序) 等价于 C^T = B^T * A^T (列主序)
    // 所以 cuBLAS 中调用 B*A 的转置即可
    printf("--- cuBLAS 矩阵乘法 (sgemm) ---\n");
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        // 默认 CUBLAS_DEFAULT_MATH 会在 Ampere 及之后启用 TF32,
        // 精度略低 (相对误差约 1e-3) 但速度显著提升。
        // 若想严格 FP32: cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

        float alpha = 1.0f, beta = 0.0f;

        //预热
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 10; i++)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        ms_cublas = ms / 10.0f;
        gf_cublas = 2.0f * M * N * K / (ms_cublas * 1e6f);
        printf("  耗时: %.3f ms, 性能: %.2f GFLOPS\n", ms_cublas, gf_cublas);
        checkResult("cuBLAS vs ref", d_C_ref, d_C, M * N);

        cublasDestroy(handle);
    }
    printf("\n");

    // --- 汇总 ---
    printf("================ 结果汇总 ================\n");
    printf("矩阵: %dx%d * %dx%d  (2*M*N*K = %.2f GFLOP)\n",
           M, K, K, N, 2.0f * M * N * K / 1e9f);
    printf("%-10s %12s %16s %16s\n",
           "实现", "耗时(ms)", "性能(GFLOPS)", "相对 tiled");
    printf("%-10s %12.3f %16.2f %16s\n",
           "朴素", ms_naive, gf_naive, "-");
    printf("%-10s %12.3f %16.2f %15.2fx\n",
           "Tiled", ms_tiled, gf_tiled,
           ms_tiled > 0 ? ms_naive / ms_tiled : 0.0f);
    printf("%-10s %12.3f %16.2f %15.2fx\n",
           "cuBLAS", ms_cublas, gf_cublas,
           ms_cublas > 0 ? ms_tiled / ms_cublas : 0.0f);
    printf("\n");

    printf("--- CUDA 库全景 ---\n");
    printf("  ┌──────────────┬───────────────────────────────┐\n");
    printf("  │ 库            │ 功能                          │\n");
    printf("  ├──────────────┼───────────────────────────────┤\n");
    printf("  │ cuBLAS        │ 线性代数 (矩阵乘法, 向量运算) │\n");
    printf("  │ cuDNN         │ 深度学习 (卷积, RNN, 激活)    │\n");
    printf("  │ cuFFT         │ 快速傅里叶变换                │\n");
    printf("  │ cuSPARSE      │ 稀疏矩阵运算                  │\n");
    printf("  │ cuSOLVER      │ 线性方程组, 特征值            │\n");
    printf("  │ cuRAND        │ 随机数生成                    │\n");
    printf("  │ Thrust        │ C++ STL 风格并行算法          │\n");
    printf("  │ CUB           │ 低级并行原语 (sort, scan...)  │\n");
    printf("  └──────────────┴───────────────────────────────┘\n");

    printf("\n--- 何时用库 vs 手写 ---\n");
    printf("  ✓ 用库: 标准操作 (矩阵乘法, FFT, 排序)\n");
    printf("  ✓ 用库: 追求极致性能 (库通常已深度优化)\n");
    printf("  ✓ 手写: 非标准操作或特殊数据布局\n");
    printf("  ✓ 手写: 融合多个操作 (减少内存读写)\n");
    printf("  ✓ 手写: 学习目的 (理解底层原理)\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("\n✓ Day 27 完成! 学会站在巨人的肩膀上。\n");
    return 0;
}