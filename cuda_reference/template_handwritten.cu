/*
 * CUDA 编程范式模板 ① 手写 kernel 完整流程
 * ================================================
 * 这是一个「可直接编译运行」的最小完整示例，演示 7 步骨架：
 *   ① cudaMalloc  ② cudaMemcpy H2D  ③ kernel<<<>>>  ④ 同步/计时
 *   ⑤ cudaMemcpy D2H  ⑥ cudaFree  ⑦ 错误检查
 * 以 inclusive scan（前缀和）朴素版为例。
 *
 * 编译: nvcc template_handwritten.cu -o hw -arch=sm_89
 * 运行: ./hw
 */
#include <cstdio>
#include <cuda_runtime.h>

// ⑦ 错误检查宏：每个 CUDA 调用都包一下
#define CUDA_CHECK(call) do {                                   \
    cudaError_t e = (call);                                     \
    if (e != cudaSuccess) {                                     \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
               cudaGetErrorString(e));                          \
        return 1;                                               \
    }                                                           \
} while(0)

// 核函数：每个线程算 out[idx] = in[0..idx] 之和（朴素，仅示范范式）
__global__ void scanInclusiveNaive(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程编号
    if (idx < n) {                                      // 边界检查
        float sum = 0.0f;
        for (int i = 0; i <= idx; i++) sum += in[i];
        out[idx] = sum;
    }
}

int main() {
    const int n = 1024;
    const size_t bytes = n * sizeof(float);

    // 主机数据
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;         // 全 1，前缀和应为 1,2,3,...

    // ① 分配设备内存
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // ② 拷入 H2D
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // 计时准备
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 启动配置
    int block = 256;
    int grid  = (n + block - 1) / block;                // 向上取整

    // ④ 计时起点
    CUDA_CHECK(cudaEventRecord(start));
    // ③ 启动核函数（异步）
    scanInclusiveNaive<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());                     // 查启动是否出错
    CUDA_CHECK(cudaEventRecord(stop));                  // 计时终点
    CUDA_CHECK(cudaEventSynchronize(stop));             // 等 GPU 真正算完

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); // 得到毫秒数

    // ⑤ 拷回 D2H
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // 验证
    bool ok = true;
    for (int i = 0; i < n; i++) if (h_out[i] != (float)(i + 1)) { ok = false; break; }
    printf("结果验证: %s   (out[0]=%.0f out[n-1]=%.0f)\n",
           ok ? "OK" : "FAIL", h_out[0], h_out[n-1]);
    printf("kernel 耗时: %.4f ms\n", ms);

    // ⑥⑦ 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out);

    printf("✓ 手写 CUDA 完整流程演示完成\n");
    return 0;
}
