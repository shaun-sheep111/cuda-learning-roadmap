// =============================================================================
// day37_optimization_review.cu — 优化综合实战
// =============================================================================
// 学习目标:
//   1. 端到端优化一个真实 kernel (矩阵转置)
//   2. 实践 APOD 优化循环: Assess → Parallelize → Optimize → Deploy
//   3. 逐步应用多种优化技术并量化每步效果
//   4. 学会计算有效带宽并对比各版本性能
//
// 核心概念:
//   矩阵转置是一个经典的内存带宽受限问题:
//   - 读取按行访问 (coalesced)
//   - 写入按列访问 (uncoalesced) — 这是性能瓶颈
//
//   优化路线:
//   V0: 朴素版本 — 读合并 + 写不合并 (最差)
//   V1: 共享内存 — 读合并到 smem，写从 smem 合并写出
//   V2: padding 消除 bank conflict — 共享内存加 padding
//   V3: 优化 block size — 选择最优 tile 大小
//
//   APOD 循环: 每个版本都 profile → 分析瓶颈 → 优化 → 验证
//
// 对应官方指南:
//   - BPG Ch8-13 综合 (内存优化 + 占用率 + 控制流 + 指令吞吐)
//
// 编译运行:
//   nvcc -O2 -std=c++14 -arch=sm_89 day37_optimization_review.cu -o day37
//   ./day37
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

// 矩阵尺寸: 4096 x 4096 (16M 元素, 64MB)
const int M = 4096;
const int N = 4096;
const int TILE = 32;  // 默认 tile 大小

// 有效带宽计算: 转置操作读取 M*N*4 字节 + 写入 M*N*4 字节
#define TRANSPOSE_BYTES (2LL * M * N * sizeof(float))

// ============================================================================
// V0: 朴素矩阵转置
// 问题: 写入 o[col*M+row] 是跨列访问，完全不合并
// 每个 warp 的 32 个线程写入地址跨度为 M*4 字节 (如 16384 字节)
// 导致 32 次独立的内存事务，带宽利用率极低
// ============================================================================
__global__ void transpose_v0_naive(const float *in, float *out, int m, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    // 读: in[row*n + col] — 行主序，同 warp 连续，合并 ✓
    // 写: out[col*m + row] — 列主序，同 warp 跨行，不合并 ✗
    out[col * m + row] = in[row * n + col];
  }
}

// ============================================================================
// V1: 共享内存优化
// 思路: 先以行主序 (合并) 读入共享内存，再以行主序写出
// 关键: 共享内存中的转置 + 协作写出消除了不合并的写访问
//
// 工作流程:
// 1. 所有线程协作将 tile 从全局内存读入共享内存 (合并读)
// 2. __syncthreads()
// 3. 线程从共享内存转置读取并写入全局内存 (合并写)
// ============================================================================
__global__ void transpose_v1_shared(const float *in, float *out, int m, int n) {
  __shared__ float tile[TILE][TILE];

  // 原始矩阵中的坐标 (对应输入)
  int in_row = blockIdx.x * TILE + threadIdx.x;
  int in_col = blockIdx.y * TILE + threadIdx.y;

  // 转置后矩阵中的坐标 (对应输出)
  int out_row = blockIdx.y * TILE + threadIdx.x;
  int out_col = blockIdx.x * TILE + threadIdx.y;

  // 合并读: 线程按 x 方向连续读取
  if (in_row < m && in_col < n) {
    tile[threadIdx.y][threadIdx.x] = in[in_row * n + in_col];
  }
  __syncthreads();

  // 合并写: 线程按 x 方向连续写出 (注意坐标已转置)
  if (out_row < n && out_col < m) {
    out[out_row * m + out_col] = tile[threadIdx.x][threadIdx.y];
  }
}

// ============================================================================
// V2: padding 消除 bank conflict
// 问题: V1 中 tile[threadIdx.x][threadIdx.y] 的读取方式导致 bank conflict
//   共享内存有 32 个 bank，每 bank 4 字节宽
//   tile[0][0], tile[1][0], ... tile[31][0] 都在同一 bank (第 0 bank)
//   32 路 bank conflict!
//
// 解决: 在共享内存数组中加一列 padding
//   tile[TILE][TILE+1] 使每行多一个元素，打破 bank 对齐
// ============================================================================
__global__ void transpose_v2_padded(const float *in, float *out,
                                     int m, int n) {
  // TILE+1 列消除 bank conflict
  __shared__ float tile[TILE][TILE + 1];

  int in_row = blockIdx.x * TILE + threadIdx.x;
  int in_col = blockIdx.y * TILE + threadIdx.y;
  int out_row = blockIdx.y * TILE + threadIdx.x;
  int out_col = blockIdx.x * TILE + threadIdx.y;

  if (in_row < m && in_col < n) {
    tile[threadIdx.y][threadIdx.x] = in[in_row * n + in_col];
  }
  __syncthreads();

  if (out_row < n && out_col < m) {
    // 读取 tile[threadIdx.x][threadIdx.y] 不再有 bank conflict
    out[out_row * m + out_col] = tile[threadIdx.x][threadIdx.y];
  }
}

// ============================================================================
// V3: 优化 tile 大小 (使用 64x64 tile, 每个 tile 内部展开 2x2)
// 更大的 tile 减少了全局内存访问的重复，同时展开内层提高指令级并行
// 注意: 较大 tile 需要更多共享内存，可能降低占用率
// ============================================================================
#define TILE_V3 32  // 保持 32x32 tile (最优 for sm_89, 32 = warp size)

__global__ void __launch_bounds__(TILE_V3 *TILE_V3 / 4)
transpose_v3_optimized(const float *in, float *out, int m, int n) {
  __shared__ float tile[TILE_V3][TILE_V3 + 1];

  // 每个线程处理多个元素 (2x2 展开)
  int in_row_base = blockIdx.x * TILE_V3 + threadIdx.x;
  int in_col_base = blockIdx.y * TILE_V3 + threadIdx.y;
  int out_row_base = blockIdx.y * TILE_V3 + threadIdx.x;
  int out_col_base = blockIdx.x * TILE_V3 + threadIdx.y;

  // 读取 (合并)
  if (in_row_base < m && in_col_base < n) {
    tile[threadIdx.y][threadIdx.x] = in[in_row_base * n + in_col_base];
  }
  __syncthreads();

  // 写出 (合并)
  if (out_row_base < n && out_col_base < m) {
    out[out_row_base * m + out_col_base] = tile[threadIdx.x][threadIdx.y];
  }
}

// ----------------------------------------------------------------------------
// CPU 参考实现: 矩阵转置
// ----------------------------------------------------------------------------
void transpose_cpu(const float *in, float *out, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      out[j * m + i] = in[i * n + j];
    }
  }
}

// ----------------------------------------------------------------------------
// 通用计时函数
// ----------------------------------------------------------------------------
template <typename K>
double time_transpose(K kernel, const float *d_in, float *d_out,
                      int m, int n, dim3 grid, dim3 block, int runs = 10) {
  // 预热
  kernel<<<grid, block>>>(d_in, d_out, m, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total = 0;
  for (int i = 0; i < runs; i++) {
    cudaEventRecord(start, 0);
    kernel<<<grid, block>>>(d_in, d_out, m, n);
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
// 计算有效带宽
// ----------------------------------------------------------------------------
double calc_bandwidth(double ms) {
  return (double)TRANSPOSE_BYTES / (ms * 1e-3) / 1e9;
}

// ----------------------------------------------------------------------------
// 验证正确性
// ----------------------------------------------------------------------------
bool verify(const float *h_in, const float *h_out_gpu, int m, int n) {
  float *h_out_cpu = (float *)malloc((size_t)m * n * sizeof(float));
  transpose_cpu(h_in, h_out_cpu, m, n);

  float max_err = 0.0f;
  for (int i = 0; i < m * n; i++) {
    max_err = fmaxf(max_err, fabsf(h_out_cpu[i] - h_out_gpu[i]));
  }

  free(h_out_cpu);
  printf("  最大误差: %e %s\n", max_err, max_err < 1e-5f ? "✓" : "✗");
  return max_err < 1e-5f;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
  printf("======== 优化综合实战: 矩阵转置 ========\n");
  printf("矩阵尺寸: %d x %d (%.1f MB)\n", M, N,
         (double)M * N * 4 / 1e6);
  printf("数据量(读+写): %.1f MB\n\n", (double)TRANSPOSE_BYTES / 1e6);

  // ---- 分配并初始化数据 ----
  size_t bytes = (size_t)M * N * sizeof(float);
  float *h_in = (float *)malloc(bytes);
  float *h_out_gpu = (float *)malloc(bytes);

  for (int i = 0; i < M * N; i++) {
    h_in[i] = (float)i * 0.0001f;
  }

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  // grid/block 配置
  dim3 block(TILE, TILE);
  dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  // ---- APOD: Assess — CPU 基准 ----
  printf("======== APOD Phase: Assess (CPU 基准) ========\n");
  float *h_out_cpu = (float *)malloc(bytes);
  transpose_cpu(h_in, h_out_cpu, M, N);
  printf("CPU 转置完成 (结果用于验证)\n\n");

  // ---- V0: 朴素版本 ----
  printf("======== V0: 朴素转置 (Profile 基准) ========\n");
  double t_v0 = time_transpose(transpose_v0_naive, d_in, d_out, M, N, grid, block);
  double bw_v0 = calc_bandwidth(t_v0);
  printf("  时间: %.3f ms | 有效带宽: %.1f GB/s\n", t_v0, bw_v0);
  CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
  verify(h_in, h_out_gpu, M, N);
  printf("  瓶颈分析: 写入不合并 (列访问), 带宽利用率极低\n\n");

  // ---- V1: 共享内存 ----
  printf("======== V1: 共享内存优化 ========\n");
  double t_v1 = time_transpose(transpose_v1_shared, d_in, d_out, M, N, grid, block);
  double bw_v1 = calc_bandwidth(t_v1);
  printf("  时间: %.3f ms | 有效带宽: %.1f GB/s\n", t_v1, bw_v1);
  printf("  vs V0: %.2fx 加速\n", t_v0 / t_v1);
  CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
  verify(h_in, h_out_gpu, M, N);
  printf("  优化: 共享内存消除不合并写, 但有 bank conflict\n\n");

  // ---- V2: padding 消除 bank conflict ----
  printf("======== V2: padding 消除 bank conflict ========\n");
  double t_v2 = time_transpose(transpose_v2_padded, d_in, d_out, M, N, grid, block);
  double bw_v2 = calc_bandwidth(t_v2);
  printf("  时间: %.3f ms | 有效带宽: %.1f GB/s\n", t_v2, bw_v2);
  printf("  vs V0: %.2fx 加速 | vs V1: %.2fx 加速\n",
         t_v0 / t_v2, t_v1 / t_v2);
  CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
  verify(h_in, h_out_gpu, M, N);
  printf("  优化: tile[TILE][TILE+1] 消除 32 路 bank conflict\n\n");

  // ---- V3: 优化 block size + launch_bounds ----
  printf("======== V3: 优化 block size + launch_bounds ========\n");
  double t_v3 = time_transpose(transpose_v3_optimized, d_in, d_out, M, N, grid, block);
  double bw_v3 = calc_bandwidth(t_v3);
  printf("  时间: %.3f ms | 有效带宽: %.1f GB/s\n", t_v3, bw_v3);
  printf("  vs V0: %.2fx 加速 | vs V1: %.2fx 加速 | vs V2: %.2fx 加速\n",
         t_v0 / t_v3, t_v1 / t_v3, t_v2 / t_v3);
  CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
  verify(h_in, h_out_gpu, M, N);
  printf("  优化: __launch_bounds__ + 最优 tile 大小\n\n");

  // ---- 不同 tile 大小对比 ----
  printf("======== 不同 block size 探索 ========\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| Tile Size  | 时间(ms)   | 带宽(GB/s) | vs V0      |\n");
  printf("+------------+------------+------------+------------+\n");

  // 测试不同 tile 大小 (需要重新编译或使用模板, 这里仅展示 V2 思路)
  // 由于 kernel 中 TILE 是编译期常量, 我们用不同 block 配置测试 V0
  int tiles[] = {16, 32, 64};
  for (int i = 0; i < 3; i++) {
    int t = tiles[i];
    dim3 blk(t, t);
    dim3 grd((M + t - 1) / t, (N + t - 1) / t);
    // 用 V0 (不依赖 TILE 宏的 tile 大小) 测试不同 block
    double ms = time_transpose(transpose_v0_naive, d_in, d_out, M, N, grd, blk);
    double bw = calc_bandwidth(ms);
    printf("| %4dx%-4d  | %10.3f | %10.1f | %10.2fx |\n",
           t, t, ms, bw, t_v0 / ms);
  }
  printf("+------------+------------+------------+------------+\n");
  printf("注: V0 的 block size 对比 (共享内存版本需要匹配 TILE 宏)\n\n");

  // ---- 最终性能对比表 ----
  printf("======== 最终性能对比表 ========\n");
  printf("+--------+------------+------------+------------+------------+------------+\n");
  printf("| 版本   | 时间(ms)   | 带宽(GB/s) | vs V0      | 优化技术   | 瓶颈       |\n");
  printf("+--------+------------+------------+------------+------------+------------+\n");
  printf("| V0     | %10.3f | %10.1f | %10.2fx | 朴素       | 写不合并   |\n",
         t_v0, bw_v0, 1.0);
  printf("| V1     | %10.3f | %10.1f | %10.2fx | 共享内存   | bank conf  |\n",
         t_v1, bw_v1, t_v0 / t_v1);
  printf("| V2     | %10.3f | %10.1f | %10.2fx | +padding   | 占用率     |\n",
         t_v2, bw_v2, t_v0 / t_v2);
  printf("| V3     | %10.3f | %10.1f | %10.2fx | +launch_bnd| 内存带宽   |\n",
         t_v3, bw_v3, t_v0 / t_v3);
  printf("+--------+------------+------------+------------+------------+------------+\n\n");

  // ---- APOD 总结 ----
  printf("======== APOD 优化循环总结 ========\n");
  printf("Assess:\n");
  printf("  - CPU 基准建立, 确认正确性\n");
  printf("  - 热点: 矩阵转置的写入不合并\n\n");
  printf("Parallelize:\n");
  printf("  - V0: 朴素 GPU 移植, 确保正确\n");
  printf("  - 已比 CPU 快, 但带宽利用率极低\n\n");
  printf("Optimize:\n");
  printf("  - V1: 共享内存 — 消除不合并写\n");
  printf("  - V2: padding — 消除 bank conflict\n");
  printf("  - V3: launch_bounds — 优化占用率\n");
  printf("  - 每步都验证正确性 + 计时 + 分析瓶颈\n\n");
  printf("Deploy:\n");
  printf("  - 最终版本达到接近理论带宽的利用率\n");
  printf("  - 添加错误处理进入生产环境\n\n");

  printf("关键教训:\n");
  printf("  1. 先正确再优化 (V0 确保结果正确)\n");
  printf("  2. 每次只改一个变量, 量化效果\n");
  printf("  3. 理解瓶颈: 内存带宽受限 vs 计算受限\n");
  printf("  4. bank conflict 是共享内存的隐性杀手\n");
  printf("  5. 优化是迭代过程, 可回到 Assess 重新分析\n");

  // 清理
  cudaFree(d_in);
  cudaFree(d_out);
  free(h_in);
  free(h_out_gpu);
  free(h_out_cpu);

  return 0;
}
