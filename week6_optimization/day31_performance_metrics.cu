// =============================================================================
// day31_performance_metrics.cu — 性能度量
// =============================================================================
// 学习目标:
//   1. 理解理论带宽 (Theoretical Bandwidth) vs 有效带宽 (Effective Bandwidth)
//   2. 掌握三种计时方法: CPU chrono 计时、CUDA Event 计时、CPU clock 计时
//   3. 学会计算有效带宽和计算吞吐量
//
// 核心概念:
//   - 理论带宽: 硬件规格决定的最大带宽 (RTX 4090 = 1008 GB/s)
//   - 有效带宽: 实际达到的带宽 = (读字节 + 写字节) / 时间
//   - 计算吞吐量: FLOPS = 操作数 / 时间
//   - CUDA Event: GPU 端精确计时，不受 CPU-GPU 异步执行影响
//   - 内存带宽利用率 = 有效带宽 / 理论带宽 × 100%
//
// 对应官方指南:
//   - BPG Ch9 (Performance Metrics)
//   - PG Ch8.1 (Timing)
//
// 编译运行:
//   nvcc -O2 -std=c++14 -arch=sm_89 day31_performance_metrics.cu -o day31
//   ./day31
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// 错误检查宏
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ----------------------------------------------------------------------------
// 常量定义
// RTX 4090: GDDR6X, 21 Gbps per pin, 384-bit bus
// 理论带宽 = 21e9 * 384 / 8 = 1008 GB/s
// ----------------------------------------------------------------------------
#define RTX4090_THEORETICAL_BW_GBPS 1008.0

// 向量大小: 256M 个 float = 1 GB (适合测试带宽)
const int N = 256 * 1024 * 1024;
const int BYTES = N * sizeof(float);

// ----------------------------------------------------------------------------
// GPU kernel: 简单的向量拷贝 (y = x)
// 这是最纯粹的内存带宽测试: 每个元素读 4 字节 + 写 4 字节
// ----------------------------------------------------------------------------
__global__ void vector_copy_kernel(const float *x, float *y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx];
  }
}

// ----------------------------------------------------------------------------
// CPU 参考实现: 向量拷贝
// ----------------------------------------------------------------------------
void vector_copy_cpu(const float *x, float *y, int n) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i];
  }
}

// ----------------------------------------------------------------------------
// 方法1: CPU chrono 计时
// 优点: 简单易用
// 缺点: 受 CPU-GPU 异步影响，需 cudaDeviceSynchronize
// ----------------------------------------------------------------------------
double timing_cpu_chrono(const float *d_x, float *d_y, int n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  // 预热: 第一次 kernel 启动有额外开销
  vector_copy_kernel<<<grid_size, block_size>>>(d_x, d_y, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto start = std::chrono::high_resolution_clock::now();

  vector_copy_kernel<<<grid_size, block_size>>>(d_x, d_y, n);
  CUDA_CHECK(cudaDeviceSynchronize());  // 必须同步，否则时间不准

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count();  // 返回毫秒
}

// ----------------------------------------------------------------------------
// 方法2: CUDA Event 计时 (推荐)
// 优点: GPU 端精确计时，不受主机干扰
// 缺点: 需要额外创建/销毁 event 对象
// ----------------------------------------------------------------------------
double timing_cuda_event(const float *d_x, float *d_y, int n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // 预热
  vector_copy_kernel<<<grid_size, block_size>>>(d_x, d_y, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start, 0));
  vector_copy_kernel<<<grid_size, block_size>>>(d_x, d_y, n);
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));  // 等待 stop 事件完成

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return (double)ms;
}

// ----------------------------------------------------------------------------
// 方法3: 多次运行取平均 (减少噪声)
// ----------------------------------------------------------------------------
double timing_cuda_event_avg(const float *d_x, float *d_y, int n, int runs) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // 预热
  vector_copy_kernel<<<grid_size, block_size>>>(d_x, d_y, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  double total_ms = 0.0;
  for (int i = 0; i < runs; i++) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    vector_copy_kernel<<<grid_size, block_size>>>(d_x, d_y, n);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms / runs;
}

// ----------------------------------------------------------------------------
// 计算有效带宽 (GB/s)
// 公式: 有效带宽 = (读取字节数 + 写入字节数) / (时间_ms × 1e6) × 1e9
// 简化: 有效带宽(GB/s) = (读+写字节) / 时间(ms) / 1e6
// 向量拷贝: 读 N×4 字节 + 写 N×4 字节 = 8N 字节
// ----------------------------------------------------------------------------
double calc_effective_bandwidth_gbps(long long bytes_rw, double ms) {
  // bytes_rw / (ms * 1e-3) = bytes/s
  // bytes/s / 1e9 = GB/s
  return (double)bytes_rw / (ms * 1e-3) / 1e9;
}

// ----------------------------------------------------------------------------
// 打印设备信息
// ----------------------------------------------------------------------------
void print_device_info() {
  int dev_id = 0;
  CUDA_CHECK(cudaGetDevice(&dev_id));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));

  printf("======== 设备信息 ========\n");
  printf("设备名称:         %s\n", prop.name);
  printf("计算能力:         %d.%d\n", prop.major, prop.minor);
  printf("SM 数量:          %d\n", prop.multiProcessorCount);
  printf("全局内存总量:     %.2f GB\n",
         (double)prop.totalGlobalMem / 1e9);

  // 计算理论带宽
  // 内存时钟速率 (kHz) × 总线宽度 (bit) / 8 / 1e6 = GB/s
  // 注意: memoryClockRate 是 kHz, memoryBusWidth 是 bit
  // GDDR6X 是双数据率 (DDR), 但 NVIDIA 驱动返回的 clockRate 已经是等效速率
  double theory_bw = 2.0 * prop.memoryClockRate * 1000.0 *
                     (prop.memoryBusWidth / 8.0) / 1e9;
  printf("内存时钟:         %.0f MHz\n", prop.memoryClockRate / 1000.0);
  printf("总线宽度:         %d bit\n", prop.memoryBusWidth);
  printf("计算理论带宽:     %.1f GB/s\n", theory_bw);
  printf("RTX 4090 标称带宽: %.1f GB/s\n", RTX4090_THEORETICAL_BW_GBPS);
  printf("==========================\n\n");
}

// ----------------------------------------------------------------------------
// 主函数
// ----------------------------------------------------------------------------
int main() {
  print_device_info();

  // 分配并初始化主机数据
  float *h_x = (float *)malloc(BYTES);
  float *h_y = (float *)malloc(BYTES);
  for (int i = 0; i < N; i++) {
    h_x[i] = (float)(i % 1000) * 0.001f;
  }

  // 分配设备内存并拷贝数据
  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, BYTES));
  CUDA_CHECK(cudaMalloc(&d_y, BYTES));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, BYTES, cudaMemcpyHostToDevice));

  // 数据量信息
  printf("======== 测试参数 ========\n");
  printf("向量元素数:       %d (%.0f M)\n", N, N / 1e6);
  printf("数据总量(单数组): %.2f GB\n", (double)BYTES / 1e9);
  printf("读+写数据量:      %.2f GB\n", 2.0 * BYTES / 1e9);
  printf("==========================\n\n");

  // ---- 方法1: CPU chrono 计时 ----
  double ms_chrono = timing_cpu_chrono(d_x, d_y, N);
  double bw_chrono = calc_effective_bandwidth_gbps(2LL * BYTES, ms_chrono);

  printf("======== 计时方法对比 (向量拷贝 y=x) ========\n");
  printf("[CPU chrono]  时间: %7.3f ms | 有效带宽: %7.1f GB/s\n",
         ms_chrono, bw_chrono);

  // ---- 方法2: CUDA Event 计时 (单次) ----
  double ms_event = timing_cuda_event(d_x, d_y, N);
  double bw_event = calc_effective_bandwidth_gbps(2LL * BYTES, ms_event);
  printf("[CUDA Event]  时间: %7.3f ms | 有效带宽: %7.1f GB/s\n",
         ms_event, bw_event);

  // ---- 方法3: CUDA Event 计时 (10次平均) ----
  double ms_avg = timing_cuda_event_avg(d_x, d_y, N, 10);
  double bw_avg = calc_effective_bandwidth_gbps(2LL * BYTES, ms_avg);
  printf("[Event x10]   时间: %7.3f ms | 有效带宽: %7.1f GB/s\n",
         ms_avg, bw_avg);

  // ---- 带宽利用率分析 ----
  printf("\n======== 带宽利用率分析 ========\n");
  printf("RTX 4090 理论带宽:   %.1f GB/s\n", RTX4090_THEORETICAL_BW_GBPS);
  printf("实测有效带宽:        %.1f GB/s\n", bw_avg);
  printf("带宽利用率:          %.1f%%\n",
         bw_avg / RTX4090_THEORETICAL_BW_GBPS * 100.0);
  printf("说明: 向量拷贝是纯内存操作，带宽利用率应达到 80-95%%\n");
  printf("      若低于 70%%，需检查是否合并访问或 block size 不当\n");

  // ---- 验证结果正确性 ----
  CUDA_CHECK(cudaMemcpy(h_y, d_y, BYTES, cudaMemcpyDeviceToHost));
  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (fabsf(h_y[i] - h_x[i]) > 1e-6f) {
      errors++;
    }
  }
  printf("\n======== 正确性验证 ========\n");
  printf("错误数: %d / %d %s\n", errors, N,
         errors == 0 ? "✓ 通过" : "✗ 失败");

  // 清理
  cudaFree(d_x);
  cudaFree(d_y);
  free(h_x);
  free(h_y);

  printf("\n总结:\n");
  printf("  1. CUDA Event 计时是 GPU 性能测量的标准方法\n");
  printf("  2. 多次运行取平均可减少计时噪声\n");
  printf("  3. 有效带宽 = (读+写字节) / 时间，是衡量内存密集型 kernel 的关键指标\n");
  printf("  4. 带宽利用率 = 有效带宽 / 理论带宽，越接近 100%% 越好\n");

  return 0;
}
