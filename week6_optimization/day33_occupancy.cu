// =============================================================================
// day33_occupancy.cu — Occupancy 占用率优化
// =============================================================================
// 学习目标:
//   1. 理解占用率 (Occupancy) 的概念和计算方法
//   2. 掌握 block size / 共享内存 / 寄存器对占用率的影响
//   3. 学会使用 cudaOccupancyMaxActiveBlocksPerMultiprocessor
//   4. 理解 __launch_bounds__ 的作用
//   5. 对比不同占用率配置下的性能
//
// 核心概念:
//   占用率 = 活跃 warp 数 / SM 最大 warp 数
//   - 理论占用率: 由寄存器、共享内存、block size 决定
//   - 实际占用率: 运行时实际活跃 warp 数 (需要 profiler)
//   高占用率不一定等于高性能，但低占用率通常意味着性能受限
//
//   限制 SM 上 block 数量的三个因素 (取最小值):
//   1. 每个 SM 最大 block 数 (如 24 on sm_89)
//   2. 每个 SM 最大 warp 数 (如 48 on sm_89)
//   3. 寄存器/共享内存限制
//
// 对应官方指南:
//   - PG Ch8.2.3 (Occupancy Calculator)
//   - BPG Ch11.1 (Optimizing for Execution Configuration)
//
// 编译运行:
//   nvcc -O2 -std=c++14 -arch=sm_89 day33_occupancy.cu -o day33
//   ./day33
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

const int N = 64 * 1024 * 1024;

// ============================================================================
// Kernel A: 基线
// ============================================================================
__global__ void baseline_kernel(float *out, const float *in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
    val = val * 2.0f + 1.0f;
    out[idx] = val;
  }
}

// ============================================================================
// 辅助: 真正压寄存器的"循环依赖"计算体 (48 个活跃寄存器)
// ============================================================================
#define REG_PRESSURE_BODY(NREG)                                                \
  if (idx < n) {                                                               \
    float v = in[idx];                                                         \
    float r[NREG];                                                             \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < NREG; i++) r[i] = (float)(i + 1);                      \
    _Pragma("unroll")                                                          \
    for (int it = 0; it < 4; it++) {                                           \
      _Pragma("unroll")                                                        \
      for (int i = 0; i < NREG; i++)                                           \
        r[i] = fmaf(r[i], v, r[(i + 1) % NREG]);                               \
    }                                                                          \
    float s = 0.0f;                                                            \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < NREG; i++) s += r[i];                                  \
    out[idx] = s;                                                              \
  }

// ============================================================================
// Kernel B: 真正的高寄存器 kernel (~55-60 regs/thread)
// ============================================================================
__global__ void high_register_kernel(float *out, const float *in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  REG_PRESSURE_BODY(48)
}

// ============================================================================
// Kernel C: __launch_bounds__(256, 6) 强制把寄存器压到 42 以下
//  256 × 6 = 1536 = SM 满线程 → 65536/1536 ≈ 42 regs/thread 上限
// ============================================================================
__global__ void __launch_bounds__(256, 6)
high_register_lb_kernel(float *out, const float *in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  REG_PRESSURE_BODY(48)
}

// ============================================================================
// Kernel D: 动态共享内存 (launch 时传 32KB)
// ============================================================================
__global__ void high_smem_kernel(float *out, const float *in, int n) {
  extern __shared__ float smem[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  smem[tid] = (idx < n) ? in[idx] : 0.0f;
  __syncthreads();
  if (idx < n) out[idx] = smem[tid] * 2.0f;
}

// ----------------------------------------------------------------------------
// 理论占用率分析
// ----------------------------------------------------------------------------
void print_occupancy_analysis(const char *name, int block_size,
                              int smem_per_block, int regs_per_thread) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  int    max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
  int    max_warps_per_sm  = prop.maxThreadsPerMultiProcessor / 32;
  size_t max_smem_per_sm   = prop.sharedMemPerMultiprocessor;
  int    max_regs_per_sm   = prop.regsPerMultiprocessor;

  int warps_per_block = block_size / 32;

  int blocks_by_block_limit = max_blocks_per_sm;
  int blocks_by_warp_limit  = (warps_per_block > 0)
      ? (max_warps_per_sm / warps_per_block) : max_blocks_per_sm;

  int regs_per_block = warps_per_block * 32 * regs_per_thread;
  const int reg_granularity = 256;
  regs_per_block = ((regs_per_block + reg_granularity - 1) / reg_granularity)
                   * reg_granularity;
  int blocks_by_reg_limit = (regs_per_block > 0)
      ? (max_regs_per_sm / regs_per_block) : max_blocks_per_sm;

  int blocks_by_smem_limit = max_blocks_per_sm;
  if (smem_per_block > 0) {
    int v = (int)(max_smem_per_sm / (size_t)smem_per_block);
    if (v < blocks_by_smem_limit) blocks_by_smem_limit = v;
  }

  int active_blocks = blocks_by_block_limit;
  if (blocks_by_warp_limit < active_blocks) active_blocks = blocks_by_warp_limit;
  if (blocks_by_reg_limit  < active_blocks) active_blocks = blocks_by_reg_limit;
  if (blocks_by_smem_limit < active_blocks) active_blocks = blocks_by_smem_limit;

  int   active_warps = active_blocks * warps_per_block;
  float occupancy    = (float)active_warps / max_warps_per_sm * 100.0f;

  printf("[%s] block=%d smem=%dB regs=%d\n", name, block_size,
         smem_per_block, regs_per_thread);
  printf("  限制分析: blocks SM=%d, warp=%d, reg=%d, smem=%d -> active=%d\n",
         blocks_by_block_limit, blocks_by_warp_limit,
         blocks_by_reg_limit, blocks_by_smem_limit, active_blocks);
  printf("  活跃 warps: %d / %d -> 占用率: %.1f%%\n\n",
         active_warps, max_warps_per_sm, occupancy);
}

// ----------------------------------------------------------------------------
// API 查询
// ----------------------------------------------------------------------------
void query_occupancy_api(const char *name, const void *kernel, int block_size,
                          int smem_per_block) {
  int max_active_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks, kernel, block_size, (size_t)smem_per_block));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int max_warps    = prop.maxThreadsPerMultiProcessor / 32;
  int active_warps = max_active_blocks * (block_size / 32);
  float occupancy  = (float)active_warps / max_warps * 100.0f;

  printf("[API 查询] %-10s max_active_blocks/SM = %d, 占用率 = %.1f%%\n",
         name, max_active_blocks, occupancy);
}

// ----------------------------------------------------------------------------
// 计时
// ----------------------------------------------------------------------------
double time_kernel_run(const void *kernel, float *d_out, const float *d_in,
                       int n, int block_size, int smem = 0) {
  int grid = (n + block_size - 1) / block_size;
  void *args[] = {(void *)&d_out, (void *)&d_in, (void *)&n};

  CUDA_CHECK(cudaLaunchKernel(kernel, dim3(grid), dim3(block_size),
                              args, (size_t)smem, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  double total = 0;
  for (int i = 0; i < 5; i++) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaLaunchKernel(kernel, dim3(grid), dim3(block_size),
                                args, (size_t)smem, 0));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    total += ms;
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total / 5;
}

// ============================================================================
int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("======== Occupancy 占用率优化 ========\n");
  printf("设备: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("SM: %d, 每 SM 最大线程: %d (warps: %d), 最大 block: %d\n",
         prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
         prop.maxThreadsPerMultiProcessor / 32, prop.maxBlocksPerMultiProcessor);
  printf("每 SM 寄存器: %d, 共享内存: %zu bytes\n\n",
         prop.regsPerMultiprocessor, prop.sharedMemPerMultiprocessor);

  printf("======== 理论占用率分析 ========\n\n");
  print_occupancy_analysis("Baseline",   256,  0,     20);
  print_occupancy_analysis("BigBlock",   1024, 0,     20);
  print_occupancy_analysis("SmallBlock", 64,   0,     20);
  print_occupancy_analysis("HighReg",    256,  0,     56);  // 约 56 regs
  print_occupancy_analysis("HighSmem",   256,  32768, 20);

  printf("======== CUDA API 占用率查询 ========\n");
  query_occupancy_api("Baseline",  (const void *)baseline_kernel,          256, 0);
  query_occupancy_api("HighReg",   (const void *)high_register_kernel,     256, 0);
  query_occupancy_api("HighReg+LB",(const void *)high_register_lb_kernel,  256, 0);
  query_occupancy_api("HighSmem",  (const void *)high_smem_kernel,         256, 32768);
  printf("\n");

  size_t bytes = (size_t)N * sizeof(float);
  float *h_in  = (float *)malloc(bytes);
  float *h_out = (float *)malloc(bytes);
  for (int i = 0; i < N; i++) h_in[i] = (float)i * 0.001f;

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in,  bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  // baseline 各 block size 性能
  printf("======== 性能对比 (baseline, 不同 block size) ========\n");
  printf("+------------+------------+------------+------------+\n");
  printf("| Block Size | 时间(ms)   | 带宽(GB/s) | 占用率(%%)  |\n");
  printf("+------------+------------+------------+------------+\n");

  int block_sizes[] = {64, 128, 256, 512, 1024};
  for (int i = 0; i < 5; i++) {
    int bs = block_sizes[i];
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, (const void *)baseline_kernel, bs, 0));
    float occ = (float)(max_blocks * (bs / 32)) /
                (prop.maxThreadsPerMultiProcessor / 32) * 100.0f;
    double ms = time_kernel_run((const void *)baseline_kernel, d_out, d_in, N, bs);
    double bw = 2.0 * bytes / (ms * 1e-3) / 1e9;
    printf("| %10d | %10.3f | %10.1f | %10.1f |\n", bs, ms, bw, occ);
  }
  printf("+------------+------------+------------+------------+\n\n");

  // LB 效果对比
  printf("======== __launch_bounds__ 效果对比 (真正有寄存器压力) ========\n");
  double ms_no_lb = time_kernel_run((const void *)high_register_kernel,
                                     d_out, d_in, N, 256);
  double ms_lb    = time_kernel_run((const void *)high_register_lb_kernel,
                                     d_out, d_in, N, 256);
  printf("无 __launch_bounds__: %.3f ms\n", ms_no_lb);
  printf("有 __launch_bounds__: %.3f ms (加速: %.2fx)\n\n",
         ms_lb, ms_no_lb / ms_lb);

  // 高 smem
  printf("======== 高共享内存 kernel 性能 ========\n");
  double ms_smem = time_kernel_run((const void *)high_smem_kernel,
                                   d_out, d_in, N, 256, 32768);
  int max_blocks_smem = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_smem, (const void *)high_smem_kernel, 256, 32768));
  float occ_smem = (float)(max_blocks_smem * 8) /
                   (prop.maxThreadsPerMultiProcessor / 32) * 100.0f;
  printf("高共享内存 kernel: %.3f ms (占用率: %.1f%%)\n\n", ms_smem, occ_smem);

  printf("======== 总结 ========\n");
  printf("1. 占用率受 block size / 寄存器 / 共享内存三者共同约束\n");
  printf("2. 高寄存器压力 kernel 用 __launch_bounds__ 才能看到效果\n");
  printf("3. 内存密集型 kernel 到 50%% 占用率往往已饱和带宽\n");
  printf("4. 高占用率 != 高性能，要用 profiler 综合判断\n");

  cudaFree(d_in); cudaFree(d_out);
  free(h_in); free(h_out);
  return 0;
}


