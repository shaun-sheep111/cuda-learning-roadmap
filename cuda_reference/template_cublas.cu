/*
 * CUDA 编程范式模板 ② cuBLAS 调用流程
 * ================================================
 * 演示用官方库 cuBLAS 做矩阵乘法 C = A*B 的完整流程：
 *   cudaMalloc/Memcpy 准备数据 → cublasCreate 句柄
 *   → cublasSgemm 计算 → 拷回 → cublasDestroy
 * 重点：cuBLAS 是【列主序 column-major】，这是最大的坑。
 *
 * 编译: nvcc template_cublas.cu -o cb -arch=sm_89 -lcublas
 * 运行: ./cb
 */
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x)  do{ cudaError_t e=(x); if(e){ \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} }while(0)
#define CUBLAS_CHECK(x) do{ cublasStatus_t s=(x); if(s){ \
    printf("cuBLAS %s:%d status=%d\n",__FILE__,__LINE__,(int)s); return 1;} }while(0)

int main() {
    // 计算 2x2: C = A * B
    // cuBLAS 列主序：下面的数组按【列】排列
    const int M = 2, N = 2, K = 2;
    float hA[] = {1,3, 2,4};   // 列主序 => 矩阵 [[1,2],[3,4]]
    float hB[] = {5,7, 6,8};   // 列主序 => 矩阵 [[5,6],[7,8]]
    float hC[4] = {0};

    // 准备设备数据（和手写一样用 cudaMalloc/Memcpy）
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));

    // 创建 cuBLAS 句柄（整个程序创建一次即可）
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // C = alpha*A*B + beta*C
    const float alpha = 1.0f, beta = 0.0f;
    // 参数: op(A), op(B), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc
    // 列主序下 lda=M, ldb=K, ldc=M
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K, &alpha,
                             dA, M, dB, K, &beta, dC, M));

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));

    // 期望 [[1,2],[3,4]]*[[5,6],[7,8]] = [[19,22],[43,50]]
    printf("cuBLAS 结果 (列主序读出):\n");
    printf("  [%.0f %.0f]\n  [%.0f %.0f]\n", hC[0], hC[2], hC[1], hC[3]);
    printf("期望:\n  [19 22]\n  [43 50]\n");

    // 清理
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    printf("✓ cuBLAS 调用流程演示完成\n");
    return 0;
}
