// =============================================================================
// day35_control_flow.cu — 控制流优化
// =============================================================================
// 学习目标:
//   1. 理解 warp divergence (分支发散) 的成因和影响
//   2. 掌握减少分支发散的优化技术
//   3. 学会数据重排消除 divergence
//   4. 对比有/无 divergence 的性能差异
//
// 核心概念:
//   Warp Divergence:
//   - 同一 warp 内的 32 个线程必须执行相同的指令 (SIMT)
//   - 如果 if/else 导致线程走不同分支，warp 会串行执行所有分支
//   - 分支越多，性能损失越大
//
//   优化策略:
//   1. 数据重排: 使同一 warp 的数据走相同分支
//   2. 分支预测 (predication): 用条件传送替代分支
//   3. 循环展开: 减少分支判断次数
//   4. 避免 warp 内 divergence: 确保条件与 warp 对齐
//
// 对应官方指南:
//   - BPG Ch13 (Control Flow)
//
// 编译运行:
//   nvcc -O2 -std=c++14 -arch=sm_89 day35_control_flow.cu -o day35
//   ./day35
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

// ============================================================================
// 1. 有 warp divergence 的 kernel
// 条件: 偶数线程做加法，奇数线程做乘法
// 由于 threadIdx.x 连续交替，同一 warp 内一半线程走 if，一半走 else
// warp 会串行执行两个分支，性能减半
// ============================================================================
__global__ void divergent_kernel(float *out, const float *a,
                                  const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx % 2 == 0){
            out[idx] = a[idx] + b[idx];
        } else {
            out[idx] = a[idx] * b[idx];
        }
    }
}

// ============================================================================
// 2. 无 warp divergence — 分支预测 (predication)
// 用三元运算符替代 if/else，编译器使用条件传送指令
// 虽然两个运算都会执行，但避免了分支跳转的开销
// ============================================================================
__global__ void predicated_kernel(float *out, const float *a,
                                   const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
       // 编译器会生成条件选择指令 (SEL)，无分支
    float add_val = a[idx] + b[idx];
    float mul_val = a[idx] * b[idx];
    out[idx] = (idx % 2 == 0) ? add_val : mul_val; 
    }
}

// ============================================================================
// 3. 无 divergence — 数据重排（简化演示版）
// 将前一半数据做加法，后一半数据做乘法，使每个 block 内线程走相同分支。
//
// 注意:
//   - 这个 kernel 的输出语义与原 divergent_kernel 不同，仅用于性能对比，
//     不参与正确性验证。
//   - 本例 N/2 是 BLOCK_SIZE 的整数倍，因此每个 block 会完整落在同一分支，
//     实际不会有 block 内 divergence。
// ============================================================================
__global__ void rearranged_kernel(float *out, const float *a,
                                   const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int half = n / 2;
    if (idx < half) {
      // 前一半线程: 全做加法
      out[idx] = a[idx] + b[idx];
    } else {
      // 后一半线程: 全做乘法
      out[idx] = a[idx] * b[idx];
    }
  }
}

// ============================================================================
// 4. 严重 divergence — 4 路分支
// 模拟 switch-case 的 4 路分支，同一 warp 最多串行执行 4 个分支
// ============================================================================
__global__ void divergent_4way_kernel(float *out, const float *a,
                                       const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int branch = idx % 4;  // 4 路分支
    float val = 0.0f;      // 初始化，避免编译器警告
    switch (branch) {
      case 0: val = a[idx] + b[idx]; break;
      case 1: val = a[idx] - b[idx]; break;
      case 2: val = a[idx] * b[idx]; break;
      case 3: val = a[idx] / b[idx]; break;
    }
    out[idx] = val;
  }
}

// ============================================================================
// 5. 4 路分支 — predication 版本
// ============================================================================
__global__ void predicated_4way_kernel(float *out, const float *a,
                                        const float *b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int branch = idx % 4;
    // 所有分支都计算，用条件选择
    float v0 = a[idx] + b[idx];
    float v1 = a[idx] - b[idx];
    float v2 = a[idx] * b[idx];
    float v3 = a[idx] / b[idx];
    float val = v0;
    val = (branch == 1) ? v1 : val;
    val = (branch == 2) ? v2 : val;
    val = (branch == 3) ? v3 : val;
    out[idx] = val;
  }
}

// ============================================================================
// 6. 循环中有 divergence — 不均匀循环
// 每个线程的循环次数不同，导致 warp 内线程提前退出但需等待
// ============================================================================
__global__ void uneven_loop_kernel(float *out, const float *a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int loop_count = (idx % 32) + 1;  // 每个线程循环 1~32 次
    float sum = 0.0f;
    for (int i = 0; i < loop_count; i++) {
      int j = idx + i;
      // i 最大为 31，j 最多超过 n 一次，避免昂贵的 % n
      if (j >= n) j -= n;
      sum += a[j];
    }
    out[idx] = sum;
  }
}

// ============================================================================
// 7. 均匀循环 — 无 divergence
// 所有线程循环相同次数
// ============================================================================
__global__ void even_loop_kernel(float *out, const float *a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int loop_count = 32;  // 所有线程循环 32 次
    float sum = 0.0f;
    for (int i = 0; i < loop_count; i++) {
      int j = idx + i;
      if (j >= n) j -= n;
      sum += a[j];
    }
    out[idx] = sum;
  }
}

// ----------------------------------------------------------------------------
// 通用计时函数
// ----------------------------------------------------------------------------
template <typename K, typename... Args>
double time_it(K kernel, Args... args) {
  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Warm-up
  kernel<<<grid, BLOCK_SIZE>>>(args..., N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total = 0;
  for (int i = 0; i < 5; i++) {
    cudaEventRecord(start, 0);
    kernel<<<grid, BLOCK_SIZE>>>(args..., N);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    total += ms;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return total / 5;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
  printf("======== 控制流优化: Warp Divergence 对比 ========\n");
  printf("元素数: %d (%.0f M)\n\n", N, N / 1e6);

  size_t bytes = N * sizeof(float);
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);

  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f + (float)(i % 1000) * 0.001f;
    h_b[i] = 1.0f + (float)(i % 997) * 0.001f;
  }

  float *d_a, *d_b, *d_out;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  // ---- 2 路分支对比 ----
  printf("======== 2 路分支: Divergence vs Predication ========\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| 版本       | 时间(ms)   | 说明       | 相对速度   |\n");
  printf("+------------+------------+------------+------------+\n");

  double t_div = time_it(divergent_kernel, d_out, d_a, d_b);
  double t_pred = time_it(predicated_kernel, d_out, d_a, d_b);
  double t_rearr = time_it(rearranged_kernel, d_out, d_a, d_b);

  printf("| Divergent  | %10.3f | if/else发散| %10.2fx |\n", t_div, 1.0);
  printf("| Predicated | %10.3f | 条件传送   | %10.2fx |\n",
         t_pred, t_div / t_pred);
  printf("| Rearranged | %10.3f | 数据重排   | %10.2fx |\n",
         t_rearr, t_div / t_rearr);
  printf("+------------+------------+------------+------------+\n");
  printf("说明:\n");
  printf("  - Divergent: 同一 warp 内线程走不同分支，串行执行\n");
  printf("  - Predicated: 两个分支都执行，用 SEL 指令选择结果\n");
  printf("  - Rearranged: 数据分组使大部分 warp 走相同分支\n");
  printf("  - 注意: Rearranged 的输出语义与原 kernel 不同，仅用于性能演示\n\n");

  // ---- 4 路分支对比 ----
  printf("======== 4 路分支: Divergence vs Predication ========\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| 版本       | 时间(ms)   | 说明       | 相对速度   |\n");
  printf("+------------+------------+------------+------------+\n");

  double t_div4 = time_it(divergent_4way_kernel, d_out, d_a, d_b);
  double t_pred4 = time_it(predicated_4way_kernel, d_out, d_a, d_b);

  printf("| 4路Diverge | %10.3f | 4路串行    | %10.2fx |\n", t_div4, 1.0);
  printf("| 4路Predic  | %10.3f | 条件传送   | %10.2fx |\n",
         t_pred4, t_div4 / t_pred4);
  printf("+------------+------------+------------+------------+\n");
  printf("说明: 4 路分支时 divergence 损失更大 (最多 4 倍串行)\n\n");

  // ---- 循环 divergence 对比 ----
  printf("======== 循环 Divergence: 不均匀 vs 均匀循环 ========\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| 版本       | 时间(ms)   | 说明       | 相对速度   |\n");
  printf("+------------+------------+------------+------------+\n");

  double t_uneven = time_it(uneven_loop_kernel, d_out, d_a);
  double t_even = time_it(even_loop_kernel, d_out, d_a);

  printf("| 不均匀循环 | %10.3f | 1~32次循环 | %10.2fx |\n", t_uneven, 1.0);
  printf("| 均匀循环   | %10.3f | 全32次循环 | %10.2fx |\n",
         t_even, t_uneven / t_even);
  printf("+------------+------------+------------+------------+\n");
  printf("说明: 不均匀循环中，循环次数少的线程需等待最长的线程完成\n");
  printf("      虽然均匀循环总工作量更大，但无等待开销，可能更快\n\n");

  // ---- 验证正确性 ----
  printf("======== 正确性验证 ========\n");
  float *h_out = (float *)malloc(bytes);
  float *h_out2 = (float *)malloc(bytes);
  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // 验证 2 路: divergent vs predicated
  CUDA_CHECK(cudaMemset(d_out, 0, bytes));
  divergent_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, d_b, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemset(d_out, 0, bytes));
  predicated_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, d_b, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out2, d_out, bytes, cudaMemcpyDeviceToHost));

  float max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    max_err = fmaxf(max_err, fabsf(h_out[i] - h_out2[i]));
  }
  printf("2路 Divergent vs Predicated 最大误差: %e %s\n",
         max_err, max_err < 1e-5f ? "✓ 一致" : "✗ 不一致");

  // 验证 4 路: divergent_4way vs predicated_4way
  CUDA_CHECK(cudaMemset(d_out, 0, bytes));
  divergent_4way_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, d_b, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaMemset(d_out, 0, bytes));
  predicated_4way_kernel<<<grid, BLOCK_SIZE>>>(d_out, d_a, d_b, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(h_out2, d_out, bytes, cudaMemcpyDeviceToHost));

  max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    max_err = fmaxf(max_err, fabsf(h_out[i] - h_out2[i]));
  }
  printf("4路 Divergent vs Predicated 最大误差: %e %s\n\n",
         max_err, max_err < 1e-5f ? "✓ 一致" : "✗ 不一致");

  printf("======== 总结 ========\n");
  printf("1. Warp divergence 是 SIMT 架构的主要性能杀手\n");
  printf("2. 同一 warp 的 32 个线程走不同分支时会串行执行\n");
  printf("3. 分支预测 (predication) 用条件传送替代分支，消除跳转开销\n");
  printf("4. 数据重排使同一 warp 的线程走相同分支，是最有效的优化\n");
  printf("5. 不均匀循环也会导致 divergence，尽量使循环次数一致\n");
  printf("6. 少量 divergence 影响不大，严重 divergence (4路+) 应重点优化\n");

  // 清理
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
  free(h_a); free(h_b); free(h_out); free(h_out2);

  return 0;
}
