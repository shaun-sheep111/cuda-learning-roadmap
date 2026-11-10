// =============================================================================
// 学习目标:
//   1. 理解 APOD (Assess, Parallelize, Optimize, Deploy) 四阶段方法论
//   2. 实践从 CPU 到 GPU 的完整优化流程
//   3. 理解 Amdahl 定律和 Gustafson 定律
//
// 核心概念:
//   APOD 是 NVIDIA 推荐的 CUDA 应用开发迭代流程:
//   - Assess (评估): 分析 CPU 基准代码，找到性能热点
//   - Parallelize (并行化): 朴素 CUDA 移植，先正确再优化
//   - Optimize (优化): 逐步应用优化技术 (共享内存、合并访问等)
//   - Deploy (部署): 生产级代码，包含错误处理和回退机制
//
//   Amdahl 定律: 加速比受限于串行部分 S = 1 / (s + (1-s)/p)
//   Gustafson 定律: 随处理器增多可处理更大规模问题 S = p - s*(p-1)
//
// 对应官方指南:
//   - BPG Ch2.2 (APOD) + Ch4 (Application Profiling)
//
// 编译运行:
//   nvcc -O2 -std=c++14 -arch=sm_89 day32_apod_methodology.cu -o day32
//   ./day32
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

// 矩阵尺寸: M×N 矩阵乘以 N×1 向量
const int M = 4096;
const int N = 4096;

// ============================================================================
// Phase 0: Assess — CPU 基准代码
// 找到热点: 矩阵向量乘法是计算密集型操作
// ============================================================================
void matvec_cpu(const float *A, const float *x, float *y, int m, int n) {
  // A: m×n 行主序, x: n×1, y: m×1
  // y[i] = sum_j A[i*n + j] * x[j]
  for (int i = 0; i < m; i++) {
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
      sum += A[i * n + j] * x[j];
    }
    y[i] = sum;
  }
}

// ============================================================================
// Phase 1: Parallelize — 朴素 CUDA 移植
// 每个线程计算 y 的一个元素，直接全局内存访问
// 问题: A 的访问不合并 (每行跨 N 个元素), 无共享内存
// ============================================================================
__global__ void matvec_naive_kernel(const float *A, const float *x,
                                     float *y, int m, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m) {
    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
      sum += A[row * n + j] * x[j];  // 每次从全局内存读取
    }
    y[row] = sum;
  }
}

// ============================================================================
// Phase 2: Optimize — 使用共享内存缓存向量 x
// 关键优化: x 被所有线程重复读取，放入共享内存可大幅减少全局内存访问
// 每个 block 处理一段 x，分块加载到 shared memory
// ============================================================================
#define BLOCK_SIZE 256
#define TILE_SIZE 256  // 每次加载到共享内存的 x 的 tile 大小

__global__ void matvec_shared_kernel(const float *A, const float *x,
                                      float *y, int m, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float x_smem[TILE_SIZE];

  float sum = 0.0f;

  // 分 tile 遍历 x
  for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
    // 协作加载 x 的一个 tile 到共享内存
    int tile_idx = tile_start + threadIdx.x;
    if (tile_idx < n) {
      x_smem[threadIdx.x] = x[tile_idx];
    } else {
      x_smem[threadIdx.x] = .0f;
    }
    __syncthreads();  // 等待所有线程加载完成

    // 使用共享内存中的 x 进行计算
    if (row < m) {
      int tile_end = min(tile_start + TILE_SIZE, n);
      for (int j = tile_start; j < tile_end; j++) {
        sum += A[row * n + j] * x_smem[j - tile_start];
      }
    }
    __syncthreads();  // 确保计算完成后再加载下一个 tile
  }

  if (row < m) {
    y[row] = sum;
  }
}

// ============================================================================
// Phase 2+: Optimize — 使用向量化加载 (float4)
// 每个线程一次处理 4 个元素，减少指令数
// ============================================================================
__global__ void matvec_vec4_kernel(const float *A, const float *x,
                                    float *y, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float x_smem[TILE_SIZE];

  float sum = 0.0f;

  for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
    int tile_idx = tile_start + threadIdx.x;
    if (tile_idx < n) {
      x_smem[threadIdx.x] = x[tile_idx];
    } else {
      x_smem[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (row < m) {
      // 使用 float4 向量化加载 A 的行
      int tile_end = (tile_start + TILE_SIZE <= n) ? tile_start + TILE_SIZE : n;
      int j = tile_start;
      // 对齐部分用 float4
      for (; j + 3 < tile_end; j += 4) {
        float4 a_val = *reinterpret_cast<const float4 *>(&A[row * n + j]);
        sum += a_val.x * x_smem[j - tile_start];
        sum += a_val.y * x_smem[j + 1 - tile_start];
        sum += a_val.z * x_smem[j + 2 - tile_start];
        sum += a_val.w * x_smem[j + 3 - tile_start];
      }
      // 处理剩余部分
      for (; j < tile_end; j++) {
        sum += A[row * n + j] * x_smem[j - tile_start];
      }
    }
    __syncthreads();
  }

  if (row < m) {
    y[row] = sum;
  }
}

// ----------------------------------------------------------------------------
// 计时辅助函数
// 注意: CUDA 不允许通过普通函数指针启动 __global__ 函数，
//       因此这里改用模板 + lambda，把 launch 表达式封闭在调用方。
// ----------------------------------------------------------------------------
template <typename LaunchFunc>
double time_kernel(LaunchFunc launch, int runs = 5) {
  // 预热
  launch();
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total = 0.0;
  for (int i = 0; i < runs; i++) {
    cudaEventRecord(start, 0);
    launch();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    total += ms;
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return total / runs;
}

// ----------------------------------------------------------------------------
// Amdahl 定律: 计算理论加速比
// s = 串行比例, p = 处理器数
// S = 1 / (s + (1-s)/p)
// ----------------------------------------------------------------------------
double amdahl_speedup(double serial_fraction, int processors) {
  return 1.0 / (serial_fraction + (1.0 - serial_fraction) / processors);
}

// ----------------------------------------------------------------------------
// Gustafson 定律: 计算加速比 (固定时间，扩大问题规模)
// S = p - s*(p-1)
// ----------------------------------------------------------------------------
double gustafson_speedup(double serial_fraction, int processors) {
  return (double)processors - serial_fraction * (processors - 1);
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
  printf("======== APOD 方法论演示: 矩阵向量乘法 ========\n");
  printf("矩阵尺寸: %d x %d\n\n", M, N);

  // ---- 分配并初始化数据 ----
  size_t mat_bytes = (size_t)M * N * sizeof(float);
  size_t vec_bytes = N * sizeof(float);

  float *h_A = (float *)malloc(mat_bytes);
  float *h_x = (float *)malloc(vec_bytes);
  float *h_y_cpu = (float *)malloc(M * sizeof(float));
  float *h_y_gpu = (float *)malloc(M * sizeof(float));

  for (int i = 0; i < M * N; i++) h_A[i] = 0.001f * (i % 1000);
  for (int i = 0; i < N; i++) h_x[i] = 1.0f;

  // ---- Phase 0: Assess — CPU 基准 ----
  printf("---- Phase 0: Assess (CPU 基准) ----\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  matvec_cpu(h_A, h_x, h_y_cpu, M, N);
  auto t1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("CPU 时间: %.3f ms\n\n", cpu_ms);

  // 分配设备内存
  float *d_A, *d_x, *d_y;
  cudaMalloc(&d_A, mat_bytes);
  cudaMalloc(&d_x, vec_bytes);
  cudaMalloc(&d_y, M * sizeof(float));
  cudaMemcpy(d_A, h_A, mat_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x, vec_bytes, cudaMemcpyHostToDevice);

  int grid = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // ---- Phase 1: Parallelize — 朴素 CUDA ----
  printf("---- Phase 1: Parallelize (朴素 CUDA) ----\n");
  double ms_naive = time_kernel([&]() {
    matvec_naive_kernel<<<grid, BLOCK_SIZE>>>(d_A, d_x, d_y, M, N);
  });
  printf("GPU 朴素: %.3f ms (加速比: %.2fx)\n\n", ms_naive,
         cpu_ms / ms_naive);

  // 验证正确性
  cudaMemcpy(h_y_gpu, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);
  float max_err = 0.0f;
  for (int i = 0; i < M; i++) {
    max_err = fmaxf(max_err, fabsf(h_y_cpu[i] - h_y_gpu[i]));
  }
  printf("正确性验证: 最大误差 = %e %s\n\n", max_err,
         max_err < 1e-2f ? "✓ 通过" : "✗ 失败");

  // ---- Phase 2: Optimize — 共享内存 ----
  printf("---- Phase 2a: Optimize (共享内存) ----\n");
  double ms_shared = time_kernel([&]() {
    matvec_shared_kernel<<<grid, BLOCK_SIZE>>>(d_A, d_x, d_y, M, N);
  });
  printf("GPU 共享内存: %.3f ms (加速比: %.2fx, vs朴素: %.2fx)\n\n",
         ms_shared, cpu_ms / ms_shared, ms_naive / ms_shared);

  // ---- Phase 2+: Optimize — 向量化加载 ----
  printf("---- Phase 2b: Optimize (向量化 float4) ----\n");
  double ms_vec4 = time_kernel([&]() {
    matvec_vec4_kernel<<<grid, BLOCK_SIZE>>>(d_A, d_x, d_y, M, N);
  });
  printf("GPU 向量化: %.3f ms (加速比: %.2fx, vs朴素: %.2fx)\n\n",
         ms_vec4, cpu_ms / ms_vec4, ms_naive / ms_vec4);

  // ---- Phase 3: Deploy — 性能总结 ----
  printf("---- Phase 3: Deploy (性能总结) ----\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| 版本       | 时间(ms)   | 加速比    | 带宽(GB/s) |\n");
  printf("+------------+------------+------------+------------+\n");
  // 有效带宽计算: 读取 A(M*N*4) + x(N*4) + 写入 y(M*4)
  double bw_naive = (double)(mat_bytes + vec_bytes + M * 4) / (ms_naive * 1e-3) / 1e9;
  double bw_shared = (double)(mat_bytes + vec_bytes + M * 4) / (ms_shared * 1e-3) / 1e9;
  double bw_vec4 = (double)(mat_bytes + vec_bytes + M * 4) / (ms_vec4 * 1e-3) / 1e9;
  printf("| CPU 基准   | %10.3f | %10.2f | %10s |\n", cpu_ms, 1.0, "-");
  printf("| GPU 朴素   | %10.3f | %10.2f | %10.1f |\n", ms_naive,
         cpu_ms / ms_naive, bw_naive);
  printf("| GPU 共享   | %10.3f | %10.2f | %10.1f |\n", ms_shared,
         cpu_ms / ms_shared, bw_shared);
  printf("| GPU 向量化 | %10.3f | %10.2f | %10.1f |\n", ms_vec4,
         cpu_ms / ms_vec4, bw_vec4);
  printf("+------------+------------+------------+------------+\n\n");

  // ---- Amdahl 定律演示 ----
  printf("======== Amdahl 定律演示 ========\n");
  printf("公式: S = 1 / (s + (1-s)/p), s=串行比例, p=处理器数\n");
  printf("假设串行比例 s = 0.01 (1%% 串行):\n");
  for (int p = 1; p <= 16384; p *= 4) {
    printf("  p = %6d -> 加速比 S = %8.2f (效率 = %5.1f%%)\n",
           p, amdahl_speedup(0.01, p),
           amdahl_speedup(0.01, p) / p * 100.0);
  }
  printf("结论: 串行比例 1%% 时，最大加速比不超过 100 (无论多少处理器)\n\n");

  // ---- Gustafson 定律演示 ----
  printf("======== Gustafson 定律演示 ========\n");
  printf("公式: S = p - s*(p-1), 固定时间扩大问题规模\n");
  printf("假设串行比例 s = 0.01:\n");
  for (int p = 1; p <= 16384; p *= 4) {
    printf("  p = %6d -> 加速比 S = %8.2f (效率 = %5.1f%%)\n",
           p, gustafson_speedup(0.01, p),
           gustafson_speedup(0.01, p) / p * 100.0);
  }
  printf("结论: Gustafson 模型下，加速比随处理器数线性增长\n\n");

  printf("======== APOD 总结 ========\n");
  printf("1. Assess: 先在 CPU 上找到热点，建立基准\n");
  printf("2. Parallelize: 朴素移植，确保正确性\n");
  printf("3. Optimize: 逐步应用优化技术，每步都验证正确性\n");
  printf("4. Deploy: 性能达标后，添加错误处理进入生产环境\n");
  printf("5. APOD 是迭代过程: Deploy 后可重新进入 Assess 阶段\n");

  // 清理
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);
  free(h_A);
  free(h_x);
  free(h_y_cpu);
  free(h_y_gpu);

  return 0;
}

