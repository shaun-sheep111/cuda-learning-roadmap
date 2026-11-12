// =============================================================================
// day34_instruction_throughput.cu — 指令吞吐量
// =============================================================================
// 学习目标:
//   1. 理解不同算术指令的吞吐量差异
//   2. 掌握 #pragma unroll 循环展开优化
//   3. 学会用位运算替代整数除法
//   4. 对比 __sinf (快速数学函数) vs sinf (标准库函数)
//   5. 理解 native vs non-native 指令差异
//
// 核心概念:
//   指令吞吐量: 每个 SM 每周期可执行的指令数
//   - 整数加法/位运算: 128 ops/cycle/SM (full throughput)
//   - 整数乘法: 64 ops/cycle/SM
//   - 整数除法/取模: 低吞吐量 (20+ cycles latency)
//   - 浮点加法/乘法: 64 ops/cycle/SM (FP32)
//   - 浮点除法: 非原生，由多个指令模拟 (低吞吐量)
//   - rsqrtf: 原生指令 (1 MUFU op)
//   - sqrtf: 由 rsqrtf + 乘法组合
//   - __sinf/__cosf: 快速近似 (1-2 MUFU ops)
//   - sinf/cosf: 标准精度 (多条指令)
//
// 对应官方指南:
//   - PG Ch8.4 (Maximize Instruction Throughput)
//   - BPG Ch12 (Instruction-Level Optimization)
//
// 编译运行:
//   nvcc -O2 -std=c++14 -arch=sm_89 day34_instruction_throughput.cu -o day34
//   ./day34
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

const int N = 32 * 1024 * 1024;  // 32M 元素
const int BLOCK_SIZE = 256;
const int WARMUP = 1;
const int REPS = 5;

// ============================================================================
// Kernels（保持不变）
// ============================================================================
__global__ void fadd_kernel(float *out, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void fmul_kernel(float *out, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void fdiv_kernel(float *out, const float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / b[idx];
}

__global__ void sqrt_kernel(float *out, const float *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrtf(a[idx]);
}

__global__ void rsqrt_kernel(float *out, const float *a, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = rsqrtf(a[idx]);
}

__global__ void sin_standard_kernel(float *out, const float *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sinf(a[idx]);
}

__global__ void sin_fast_kernel(float *out, const float *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __sinf(a[idx]);
}

__global__ void idiv_kernel(int *out, const int *a, const int *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / b[idx];
}

__global__ void ishift_kernel(int *out, const int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] >> 3;
}

__global__ void loop_no_unroll_kernel(float *out, const float *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {
        sum += a[(idx + i) % n] * (float)(i + 1);
        }
        out[idx] = sum;
    }
}

__global__ void loop_unroll_kernel(float *out, const float *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float s0 = a[(idx + 0) % n] * 1.0f;
        float s1 = a[(idx + 1) % n] * 2.0f;
        float s2 = a[(idx + 2) % n] * 3.0f;
        float s3 = a[(idx + 3) % n] * 4.0f;
        float s4 = a[(idx + 4) % n] * 5.0f;
        float s5 = a[(idx + 5) % n] * 6.0f;
        float s6 = a[(idx + 6) % n] * 7.0f;
        float s7 = a[(idx + 7) % n] * 8.0f;
        out[idx] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    }
}

// ============================================================================
// 修复后的计时函数：
//   - 接收 lambda 闭包，内部直接写 <<<>>> launch 表达式
//   - 避免把 __global__ 函数指针当作普通模板参数传递导致的编译失败
//   - 增加 warmup / launch 错误检查
// ============================================================================
template <typename F>
double time_it(F&& launch) {
    // warmup
    for (int i = 0; i < WARMUP; i++) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total = 0.0;
    for (int i = 0; i < REPS; i++) {
        CUDA_CHECK(cudaEventRecord(start, 0));
        launch();
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total / REPS;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("======== 指令吞吐量对比测试 ========\n");
    printf("元素数: %d (%.0f M)\n\n", N, N / 1e6);

    size_t fbytes = (size_t)N * sizeof(float);
    size_t ibytes = (size_t)N * sizeof(int);

    float *h_a = (float *)malloc(fbytes);
    float *h_b = (float *)malloc(fbytes);
    int   *h_ia = (int *)malloc(ibytes);
    int   *h_ib = (int *)malloc(ibytes);
    if (!h_a || !h_b || !h_ia || !h_ib) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; i++) {
        h_a[i]  = 1.0f + (float)(i % 1000) * 0.001f;
        h_b[i]  = 1.0f + (float)(i % 997)  * 0.001f;
        h_ia[i] = i + 1;
        h_ib[i] = (i % 7) + 1;
    }

    float *d_a, *d_b, *d_out;
    int   *d_ia, *d_ib, *d_iout;
    CUDA_CHECK(cudaMalloc(&d_a,    fbytes));
    CUDA_CHECK(cudaMalloc(&d_b,    fbytes));
    CUDA_CHECK(cudaMalloc(&d_out,  fbytes));
    CUDA_CHECK(cudaMalloc(&d_ia,   ibytes));
    CUDA_CHECK(cudaMalloc(&d_ib,   ibytes));
    CUDA_CHECK(cudaMalloc(&d_iout, ibytes));

    CUDA_CHECK(cudaMemcpy(d_a,  h_a,  fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,  h_b,  fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ia, h_ia, ibytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ib, h_ib, ibytes, cudaMemcpyHostToDevice));

    const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ---- 浮点算术指令对比 ----
    printf("======== 浮点算术指令吞吐量对比 ========\n");
    printf("+------------+------------+------------+------------+\n");
    printf("| 操作       | 时间(ms)   | 吞吐量评价 | 相对速度   |\n");
    printf("+------------+------------+------------+------------+\n");

    double t_add   = time_it([&](){ fadd_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, d_b, N); });
    double t_mul   = time_it([&](){ fmul_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, d_b, N); });
    double t_div   = time_it([&](){ fdiv_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, d_b, N); });
    double t_sqrt  = time_it([&](){ sqrt_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, N);     });
    double t_rsqrt = time_it([&](){ rsqrt_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, N);    });

    printf("| FP32 加法  | %10.3f | 高(原生)   | %10.2fx |\n", t_add,   1.0);
    printf("| FP32 乘法  | %10.3f | 高(原生)   | %10.2fx |\n", t_mul,   t_add / t_mul);
    printf("| FP32 除法  | %10.3f | 低(非原生) | %10.2fx |\n", t_div,   t_add / t_div);
    printf("| sqrtf      | %10.3f | 中(组合)   | %10.2fx |\n", t_sqrt,  t_add / t_sqrt);
    printf("| rsqrtf     | %10.3f | 高(MUFU)   | %10.2fx |\n", t_rsqrt, t_add / t_rsqrt);
    printf("+------------+------------+------------+------------+\n\n");

    // ---- 三角函数对比 ----
    printf("======== 三角函数对比 (sinf vs __sinf) ========\n");
    printf("+------------+------------+------------+------------+\n");
    printf("| 函数       | 时间(ms)   | 精度       | 相对速度   |\n");
    printf("+------------+------------+------------+------------+\n");

    double t_sin  = time_it([&](){ sin_standard_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, N); });
    double t_fsin = time_it([&](){ sin_fast_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, N);     });
    printf("| sinf       | %10.3f | 高(标准)   | %10.2fx |\n", t_sin,  1.0);
    printf("| __sinf     | %10.3f | 中(快速)   | %10.2fx |\n", t_fsin, t_sin / t_fsin);
    printf("+------------+------------+------------+------------+\n");
    printf("结论: __sinf 使用硬件 MUFU.SIN 指令，速度数倍于 sinf\n\n");

    // ---- 整数除法 vs 位运算 ----
    printf("======== 整数除法 vs 位运算 ========\n");
    printf("+------------+------------+------------+------------+\n");
    printf("| 操作       | 时间(ms)   | 吞吐量评价 | 相对速度   |\n");
    printf("+------------+------------+------------+------------+\n");

    double t_idiv   = time_it([&](){ idiv_kernel<<<grid, BLOCK_SIZE>>>(d_iout, d_ia, d_ib, N); });
    double t_ishift = time_it([&](){ ishift_kernel<<<grid, BLOCK_SIZE>>>(d_iout, d_ia, N);      });
    printf("| 整数除法   | %10.3f | 低(非原生) | %10.2fx |\n", t_idiv,   1.0);
    printf("| 位运算(>>) | %10.3f | 高(原生)   | %10.2fx |\n", t_ishift, t_idiv / t_ishift);
    printf("+------------+------------+------------+------------+\n");
    printf("结论: 整数除法是 GPU 上最慢的操作之一，应尽量用位运算替代\n\n");

    // ---- 循环展开对比 ----
    printf("======== 循环展开对比 ========\n");
    printf("+------------+------------+------------+------------+\n");
    printf("| 版本       | 时间(ms)   | 说明       | 相对速度   |\n");
    printf("+------------+------------+------------+------------+\n");

    double t_nounroll = time_it([&](){ loop_no_unroll_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, N); });
    double t_unroll   = time_it([&](){ loop_unroll_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, N);    });
    printf("| 不展开     | %10.3f | 有循环开销 | %10.2fx |\n", t_nounroll, 1.0);
    printf("| 手动展开   | %10.3f | 指令调度好 | %10.2fx |\n", t_unroll,   t_nounroll / t_unroll);
    printf("+------------+------------+------------+------------+\n");
    printf("结论: 循环展开减少分支开销，允许编译器更好地调度指令\n\n");

    printf("提示: 也可以使用 #pragma unroll 让编译器自动展开:\n");
    printf("  #pragma unroll 8\n");
    printf("  for (int i = 0; i < 8; i++) { ... }\n\n");

    printf("======== 总结 ========\n");
    printf("1. 浮点加法/乘法是原生指令，吞吐量最高\n");
    printf("2. 浮点除法是非原生指令，尽量避免在热循环中使用\n");
    printf("3. rsqrtf 是原生 MUFU 指令，比 sqrtf 快得多\n");
    printf("4. __sinf/__cosf 等快速函数比标准库版本快数倍\n");
    printf("5. 整数除法极慢，除以 2 的幂用右移替代\n");
    printf("6. 循环展开可减少分支开销，提高指令级并行\n");
    printf("7. 编译标志 --use_fast_math 可启用所有快速数学函数\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    cudaFree(d_ia); cudaFree(d_ib); cudaFree(d_iout);
    free(h_a); free(h_b); free(h_ia); free(h_ib);

    return 0;
}