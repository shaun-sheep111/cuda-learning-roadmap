// =============================================================================
// day36_numerical_accuracy.cu — 数值精度
// =============================================================================
// 学习目标:
//   1. 理解单精度 (FP32) vs 双精度 (FP64) 的精度和性能差异
//   2. 理解浮点运算的非结合性: (a+b)+c ≠ a+(b+c)
//   3. 掌握 Kahan 求和算法减少浮点误差
//   4. 了解 IEEE 754 浮点标准合规性
//   5. 理解归约中不同求和顺序的精度差异
//
// 核心概念:
//   IEEE 754 浮点标准:
//   - FP32: 1 位符号 + 8 位指数 + 23 位尾数, 精度 ~7 位有效十进制
//   - FP64: 1 位符号 + 11 位指数 + 52 位尾数, 精度 ~16 位有效十进制
//   - 浮点运算不满足结合律: (a+b)+c ≠ a+(b+c)
//   - 大数吃小数: 相差很大的数相加，小数可能被完全丢失
//
//   Kahan 求和:
//   - 通过补偿被丢弃的低位来减少误差
//   - 误差补偿: c = (y - t) - sum, 其中 y = input - c, t = sum + y
//
//   性能差异:
//   - 消费级 GPU (RTX 4090): FP64 吞吐量 = FP32 的 1/64
//   - 专业卡 (A100): FP64 吞吐量 = FP32 的 1/2
//
// 对应官方指南:
//   - BPG Ch7.3 (Numerical Accuracy and Precision)
//
// 编译运行:
//   nvcc -O2 -std=c++14 -arch=sm_89 day36_numerical_accuracy.cu -o day36
//   ./day36
// =============================================================================

// =============================================================================
// day36_numerical_accuracy.cu — 数值精度（修复版）
// =============================================================================
// 学习目标:
//   1. 理解单精度 (FP32) vs 双精度 (FP64) 的精度和性能差异
//   2. 理解浮点运算的非结合性: (a+b)+c ≠ a+(b+c)
//   3. 掌握 Kahan 求和算法减少浮点误差
//   4. 了解 IEEE 754 浮点标准合规性
//   5. 理解归约中不同求和顺序的精度差异
//
// 修复说明:
//   - 修复 time_it 中 result 被重复累加的问题（计时与结果获取分离）
//   - 修复 Kahan kernel 树形归约丢失补偿的问题（用 cdata 共享数组）
//   - 修复 Kahan kernel 最终结果未加回补偿的问题
//   - 修复非结合性演示无法展示差异的问题（改用经典例子）
//   - 统一 CPU FP64 参考与 GPU FP64 的输入数值
//
// 编译运行:
//   nvcc -O2 -std=c++14 -arch=sm_89 day36_numerical_accuracy.cu -o day36
//   ./day36
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

const int N = 16 * 1024 * 1024;  // 16M 元素
const int BLOCK_SIZE = 256;
const int GRID_SIZE = 256;

// ============================================================================
// 1. 朴素单精度向量求和 kernel
// ============================================================================
__global__ void sum_naive_fp32_kernel(const float *a, float *result, int n) {
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float my_sum = 0.0f;
  for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
    my_sum += a[i];
  }
  sdata[tid] = my_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, sdata[0]);
  }
}

// ============================================================================
// 2. 朴素双精度向量求和 kernel
// ============================================================================
__global__ void sum_naive_fp64_kernel(const double *a, double *result, int n) {
  __shared__ double sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double my_sum = 0.0;
  for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
    my_sum += a[i];
  }
  sdata[tid] = my_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, sdata[0]);
  }
}

// ============================================================================
// 3. Kahan 求和 kernel (单精度) — 修复版
// 使用两个共享数组，保证树形归约时两个操作数的补偿都被正确处理
// ============================================================================
__global__ void sum_kahan_fp32_kernel(const float *a, float *result, int n) {
  __shared__ float sdata[BLOCK_SIZE];
  __shared__ float cdata[BLOCK_SIZE];   // 每个线程的补偿
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;
  float c = 0.0f;
  for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
    float y = a[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  sdata[tid] = sum;
  cdata[tid] = c;
  __syncthreads();

  // Kahan 树形归约: 先把右操作数的补偿应用，再做 Kahan 加法
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float x = sdata[tid + s] - cdata[tid + s];   // 应用右操作数补偿
      float y = x - cdata[tid];                    // 减去左操作数补偿
      float t = sdata[tid] + y;
      cdata[tid] = (t - sdata[tid]) - y;           // 更新补偿
      sdata[tid] = t;
    }
    __syncthreads();
  }

  if (tid == 0) {
    // 加回最终补偿
    atomicAdd(result, sdata[0] - cdata[0]);
  }
}

// ============================================================================
// 4. 演示浮点非结合性: (a+b)+c vs a+(b+c) — 修复版
// 经典例子: a=1e10, b=-1e10, c=1
//   (a+b)+c = (1e10-1e10) + 1 = 0 + 1 = 1
//   a+(b+c) = 1e10 + (-1e10+1) = 1e10 + (-1e10) = 0
//   (因为 -1e10+1 舍入到 -1e10)
// 差值 = 1
// ============================================================================
__global__ void associativity_demo_kernel(float *results) {
  float a = 1.0e10f;
  float b = -1.0e10f;
  float c = 1.0f;

  float left = (a + b) + c;   // = 1
  float right = a + (b + c);  // = 0

  results[0] = left;
  results[1] = right;
  results[2] = left - right;
}

// ============================================================================
// 5. CPU 参考函数
// ============================================================================
float sequential_sum_cpu(const float *a, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += a[i];
  }
  return sum;
}

double sequential_sum_fp64_cpu(const float *a, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += (double)a[i];
  }
  return sum;
}

float kahan_sum_cpu(const float *a, int n) {
  float sum = 0.0f;
  float c = 0.0f;
  for (int i = 0; i < n; i++) {
    float y = a[i] - c;
    float t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

// ============================================================================
// 通用计时函数（只用于计时，不关心结果）
// ============================================================================
template <typename K, typename... Args>
double time_it(K kernel, Args... args) {
  kernel<<<GRID_SIZE, BLOCK_SIZE>>>(args...);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total = 0;
  for (int i = 0; i < 5; i++) {
    cudaEventRecord(start, 0);
    kernel<<<GRID_SIZE, BLOCK_SIZE>>>(args...);
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
  printf("======== 数值精度: FP32 vs FP64 vs Kahan ========\n");
  printf("元素数: %d (%.0f M)\n\n", N, N / 1e6);

  // ---- 创建测试数据 ----
  // 场景: 16M 个 0.1 相加，理论结果 = 1.6M = 1.6e6
  size_t bytes_f32 = N * sizeof(float);
  size_t bytes_f64 = N * sizeof(double);

  float *h_a = (float *)malloc(bytes_f32);
  double *h_a64 = (double *)malloc(bytes_f64);

  for (int i = 0; i < N; i++) {
    h_a[i] = 0.1f;
    // 关键: 用 float 0.1 的值填充 double，确保所有测试输入数值一致
    h_a64[i] = (double)h_a[i];
  }

  // ---- CPU 参考计算 ----
  printf("======== CPU 参考计算 ========\n");
  double ref_fp64 = sequential_sum_fp64_cpu(h_a, N);
  float cpu_naive = sequential_sum_cpu(h_a, N);
  float cpu_kahan = kahan_sum_cpu(h_a, N);

  printf("FP64 参考 (高精度):    %.10f\n", ref_fp64);
  printf("FP32 顺序求和:         %.10f (误差: %.6e)\n",
         cpu_naive, fabs(cpu_naive - ref_fp64));
  printf("FP32 Kahan 求和:       %.10f (误差: %.6e)\n\n",
         cpu_kahan, fabs(cpu_kahan - ref_fp64));

  // ---- GPU 内存分配 ----
  float *d_a;
  double *d_a64;
  float *d_result_f32;
  double *d_result_f64;

  CUDA_CHECK(cudaMalloc(&d_a, bytes_f32));
  CUDA_CHECK(cudaMalloc(&d_a64, bytes_f64));
  CUDA_CHECK(cudaMalloc(&d_result_f32, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_result_f64, sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_f32, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_a64, h_a64, bytes_f64, cudaMemcpyHostToDevice));

  // ---- 计时（只测时间，不关心累加值） ----
  double t_fp32 = time_it(sum_naive_fp32_kernel, d_a, d_result_f32, N);
  double t_fp64 = time_it(sum_naive_fp64_kernel, d_a64, d_result_f64, N);
  double t_kahan = time_it(sum_kahan_fp32_kernel, d_a, d_result_f32, N);

  // ---- 单独运行一次获取正确结果 ----
  // (关键修复: time_it 中 kernel 会累加 6 次到 result，必须重置后重新计算)
  float gpu_fp32;
  CUDA_CHECK(cudaMemset(d_result_f32, 0, sizeof(float)));
  sum_naive_fp32_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_result_f32, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(&gpu_fp32, d_result_f32, sizeof(float),
                         cudaMemcpyDeviceToHost));

  double gpu_fp64;
  CUDA_CHECK(cudaMemset(d_result_f64, 0, sizeof(double)));
  sum_naive_fp64_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_a64, d_result_f64, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(&gpu_fp64, d_result_f64, sizeof(double),
                         cudaMemcpyDeviceToHost));

  float gpu_kahan;
  CUDA_CHECK(cudaMemset(d_result_f32, 0, sizeof(float)));
  sum_kahan_fp32_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_result_f32, N);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(&gpu_kahan, d_result_f32, sizeof(float),
                         cudaMemcpyDeviceToHost));

  // ---- 精度对比（统一用 ref_fp64 作为参考） ----
  printf("======== GPU 精度对比 ========\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| 方法       | 结果       | 绝对误差   | 相对误差%%  |\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| FP64 参考  | %10.4f | %10s | %10s |\n", ref_fp64, "-", "-");
  printf("| GPU FP32   | %10.4f | %.6e | %.6f%%  |\n",
         gpu_fp32, fabs(ref_fp64 - gpu_fp32),
         fabs(ref_fp64 - gpu_fp32) / ref_fp64 * 100.0);
  printf("| GPU FP64   | %10.4f | %.6e | %.6f%%  |\n",
         gpu_fp64, fabs(ref_fp64 - gpu_fp64),
         fabs(ref_fp64 - gpu_fp64) / ref_fp64 * 100.0);
  printf("| GPU Kahan  | %10.4f | %.6e | %.6f%%  |\n",
         gpu_kahan, fabs(ref_fp64 - gpu_kahan),
         fabs(ref_fp64 - gpu_kahan) / ref_fp64 * 100.0);
  printf("+------------+------------+------------+------------+\n\n");

  // ---- 性能对比 ----
  printf("======== 性能对比 (FP32 vs FP64 vs Kahan) ========\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| 方法       | 时间(ms)   | 相对速度   | 说明       |\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| GPU FP32   | %10.3f | %10.2fx | 基准       |\n",
         t_fp32, 1.0);
  printf("| GPU FP64   | %10.3f | %10.2fx | 慢(1/64)   |\n",
         t_fp64, t_fp32 / t_fp64);
  printf("| GPU Kahan  | %10.3f | %10.2fx | 精度高     |\n",
         t_kahan, t_fp32 / t_kahan);
  printf("+------------+------------+------------+------------+\n");
  printf("说明: RTX 4090 的 FP64 吞吐量为 FP32 的 1/64 (消费卡限制)\n");
  printf("      Kahan 求和用更多计算换取更高精度\n\n");

  // ---- 浮点非结合性演示 ----
  printf("======== 浮点非结合性演示 ========\n");
  float *d_assoc;
  CUDA_CHECK(cudaMalloc(&d_assoc, 3 * sizeof(float)));
  associativity_demo_kernel<<<1, 1>>>(d_assoc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float assoc_results[3];
  CUDA_CHECK(cudaMemcpy(assoc_results, d_assoc, 3 * sizeof(float),
                         cudaMemcpyDeviceToHost));

  float a_val = 1.0e10f, b_val = -1.0e10f, c_val = 1.0f;
  printf("a = %.1e, b = %.1e, c = %.1f\n", a_val, b_val, c_val);
  printf("(a + b) + c = %.6f\n", assoc_results[0]);
  printf("a + (b + c) = %.6f\n", assoc_results[1]);
  printf("差值 = %.6e\n\n", assoc_results[2]);
  printf("结论: 浮点运算不满足结合律，不同计算顺序可能得到不同结果\n");
  printf("      当 |a| >> |b|, |c| 时，b 和 c 在加法中可能被完全丢失\n\n");

  // ---- IEEE 754 合规性说明 ----
  printf("======== IEEE 754 合规性说明 ========\n");
  printf("GPU IEEE 754 合规性:\n");
  printf("  - FP32: 基本合规，但 fused multiply-add (FMA) 可改变结果\n");
  printf("  - FP64: 完全合规\n");
  printf("  - --ftz=true (默认): 非规格化数清零，影响极小值的精度\n");
  printf("  - --prec-div=false: 降低除法精度以提高速度\n");
  printf("  - --prec-sqrt=false: 降低平方根精度以提高速度\n");
  printf("  - --use_fast_math: 启用所有快速数学 (精度降低)\n\n");

  printf("======== 总结 ========\n");
  printf("1. FP32 精度约 7 位有效数字，FP64 约 16 位\n");
  printf("2. 浮点运算不满足结合律，归约顺序影响结果\n");
  printf("3. 大数吃小数: 相差大的数相加会丢失小数\n");
  printf("4. Kahan 求和用 O(n) 额外计算换取显著精度提升\n");
  printf("5. 消费卡 FP64 很慢 (1/64)，专业卡 FP64 快 (1/2)\n");
  printf("6. 需要高精度时优先考虑: 双精度 > Kahan > 排序后求和\n");
  printf("7. --use_fast_math 会降低精度，仅在精度要求不高时使用\n");

  // 清理
  cudaFree(d_a); cudaFree(d_a64);
  cudaFree(d_result_f32); cudaFree(d_result_f64);
  cudaFree(d_assoc);
  free(h_a); free(h_a64);

  return 0;
}