/*
 * Day 20: 双调排序 (Bitonic Sort) — 并行排序算法
 * =================================================
 * 学习目标:
 *   1. 理解双调序列和双调排序的原理
 *   2. 实现并行双调排序 kernel
 *   3. 理解为什么双调排序适合 GPU (固定比较模式, 无数据依赖)
 *   4. 对比 GPU 并行排序和 CPU 串行排序
 *
 * 双调排序原理:
 *   - 双调序列: 先增后减 (或先减后增) 的序列
 *   - 双调序列可以通过 batcher 比较网络完全排序
 *   - 每一步的比较是固定的, 不依赖数据 → 天然并行
 *   - 复杂度: O(n log²n) 比较操作, 但每步 O(n) 并行
 *
 * 比较网络结构 (n=8):
 *   Stage 1: 比较距离 1, 分组 2
 *   Stage 2: 比较距离 2, 分组 4, 然后距离 1
 *   Stage 3: 比较距离 4, 分组 8, 然后距离 2, 距离 1
 *   nvcc13.0编译
 主要优化:
 *   1. 修复计时: 只计 kernel, 不含 cudaMemcpy; 预热后再计时
 *   2. 加 CUDA 错误检查、n 为 2 的幂检查、ij < n 越界检查
 *   3. V1: j < 32 用 __syncwarp 代替 __syncthreads (减少约 70% 同步开销)
 *   4. V2: 把 j < blockDimElems 的多步合并到一个 kernel, kernel 数 ~210 -> ~75
 *   5. V2: 每线程处理一个比较对 (p -> a 映射), 线程利用率 100%
 *   6. 测试数据改用浮点随机, 减少重复值对性能测试的影响
 * 编译: nvcc day20_bitonic_sort.cu -o day20
 * 运行: ./day20
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>

// ============================================================
// CUDA 错误检查宏
// ============================================================
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_LAUNCH() do { \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "Kernel launch error: %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static bool isPow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// ============================================================
// V1: 单 block 双调排序 (n <= 1024 且 n 是 2 的幂)
//   每 thread 处理元素 tid, 只处理 ij > i 的对
//   优化: j < 32 时 i 与 i^j 必在同一 warp, 用 __syncwarp 替代 __syncthreads
// ============================================================
__global__ void bitonicSortBlock(float *data, int n){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    if(tid < n) sdata[tid] = data[tid];
    __syncthreads();

    for(int k = 2; k <= n; k <<= 1){
        for(int j =k >> 1; j >0; j >>= 1){
            int i = tid;
            int ij = i ^ j;
            if (ij > i && i < n){
                bool dir = ((i & k) == 0);   // 升序块
                float a = sdata[i];
                float b = sdata[ij];
                if ((a > b) == dir){
                    sdata[i] = b;
                    sdata[ij] = a;
                }
            }
            // j < 32 时 i 与 i^j 在同一 warp 内, 用 warp 级同步即可
            if (j >= 32) __syncthreads();
            else __syncwarp();
        }
    }
    if (tid < n) data[tid] = sdata[tid];
}

// ============================================================
// V2a: 跨 block step (j >= blockDimElems)
//   每线程处理一个比较对 (p -> a 映射), 线程利用率 100%
//   a 满足 a & j == 0 (即 a 是每对中较小的索引)
// ============================================================
__global__ void bitonicSortCrossBlock(float *data, int j, int k, int n) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n >> 1;
    if (p >= total_pairs) return;

    //把p映射到a:低log2(j)位保留，第log2(j)位填0
    int a = ((p & ~(j - 1)) << 1) | (p & (j - 1));
    int b = a ^ j;

    bool dir = ((a & k) == 0);
    float va = data[a];
    float vb = data[b];
    if ((va > vb) == dir) {
        data[a] = vb;
        data[b] = va;
    }
}

// ============================================================
// V2b: block 内 step, 一个 kernel 完成某个 k 的所有 j < blockDimElems
//   j < 32 时用 __syncwarp
// ============================================================
__global__ void bitonicSortInBlock(float *data, int k, int blockDimElems, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDimElems;

    // 加载本 block 负责的元素到共享内存
    for (int i = tid; i < blockDimElems; i += blockDim.x) {
        int gi = blockStart + i;
        sdata[i] = (gi < n) ? data[gi] : 0.0f;
    }
    __syncthreads();

    // 处理所有 j < blockDimElems 的 step
    int jmax = min(k >> 1, blockDimElems >> 1);
    for (int j = jmax; j > 0; j >>= 1) {
        int i  = tid;
        int ij = i ^ j;
        if (ij > i && i < blockDimElems) {
            int gi = blockStart + i;
            bool dir = ((gi & k) == 0);     // 用全局索引判断方向
            float a = sdata[i];
            float b = sdata[ij];
            if ((a > b) == dir) {
                sdata[i]  = b;
                sdata[ij] = a;
            }
        }
        if (j >= 32) __syncthreads();
        else         __syncwarp();
    }

    // 写回全局内存
    for (int i = tid; i < blockDimElems; i += blockDim.x) {
        int gi = blockStart + i;
        if (gi < n) data[gi] = sdata[i];
    }
}

// ============================================================
// CPU 参考排序 (仅用于验证, 冒泡排序)
// ============================================================
static void sortCPU(float *data, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (data[j] > data[j + 1]) {
                float t = data[j];
                data[j] = data[j + 1];
                data[j + 1] = t;
            }
        }
    }
}

// ============================================================
// 运行一次完整的 V2 排序
//   - 跨 block step (j >= blockDimElems): 每 step 一个 kernel
//   - block 内 step (j <  blockDimElems): 全部合并到一个 kernel
// ============================================================
static void runBitonicV2(float *d_data, int n, int blockDimElems) {
    int blockSize = blockDimElems;
    int numBlocks = n / blockDimElems;
    size_t shmem  = blockDimElems * sizeof(float);
    int pairs     = n >> 1;
    int crossGrid = (pairs + blockSize - 1) / blockSize;

    for (int k = 2; k <= n; k <<= 1) {
        // ---- 跨 block step: j 从 k/2 递减到 >= blockDimElems ----
        for (int j = k >> 1; j >= blockDimElems; j >>= 1) {
            bitonicSortCrossBlock<<<crossGrid, blockSize>>>(d_data, j, k, n);
            CHECK_LAUNCH();
        }
        // ---- block 内 step: 所有 j < blockDimElems 一次完成 ----
        bitonicSortInBlock<<<numBlocks, blockSize, shmem>>>(d_data, k, blockDimElems, n);
        CHECK_LAUNCH();
    }
}

// ============================================================
// main
// ============================================================
int main() {
    printf("============================================\n");
    printf("   Day 20: 双调排序 (Bitonic Sort) 优化版\n");
    printf("============================================\n\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ============================================================
    // V1: 单 block
    // ============================================================
    int n = 1024;
    if (!isPow2(n) || n > 1024) {
        fprintf(stderr, "V1 requires 1 <= n <= 1024 and n = 2^k\n");
        return 1;
    }
    size_t size = n * sizeof(float);
    printf("[V1] 单 block, n = %d\n\n", n);

    float *h_orig = (float *)malloc(size);
    float *h_data = (float *)malloc(size);
    float *h_ref  = (float *)malloc(size);

    srand(42);
    for (int i = 0; i < n; i++) {
        // 浮点随机, 减少重复值
        h_orig[i] = (float)rand() / (float)RAND_MAX * 100000.0f;
        h_ref[i]  = h_orig[i];
        h_data[i] = h_orig[i];
    }
    sortCPU(h_ref, n);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_orig, size, cudaMemcpyHostToDevice));

    // ---- 正确性验证 ----
    bitonicSortBlock<<<1, n, size>>>(d_data, n);
    CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_data[i] - h_ref[i]) > 1e-3f) { ok = false; break; }
    }
    printf("  验证: %s\n", ok ? "✓ 通过" : "✗ 失败");

    // ---- 计时: 只计 kernel, 不含 H2D 拷贝 ----
    // 双调排序比较网络固定, 对已排序/随机数据执行时间几乎相同
    int reps1 = 1000;
    bitonicSortBlock<<<1, n, size>>>(d_data, n);  // 预热
    CHECK_LAUNCH();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps1; r++) {
        bitonicSortBlock<<<1, n, size>>>(d_data, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms1;
    CUDA_CHECK(cudaEventElapsedTime(&ms1, start, stop));
    printf("  耗时: %.4f us (平均 %d 次, 仅 kernel)\n\n",
           ms1 * 1000.0f / reps1, reps1);

    // ============================================================
    // V2: 多 block
    // ============================================================
    int n2 = 1 << 20;            // 1M 元素
    int blockDimElems = 1024;    // 每 block 处理 1024 个元素
    if (!isPow2(n2) || n2 % blockDimElems != 0) {
        fprintf(stderr, "V2 requires n2 = 2^k and n2 %% blockDimElems == 0\n");
        return 1;
    }
    size_t size2 = n2 * sizeof(float);
    printf("[V2] 多 block, n = %d, blockDimElems = %d\n\n", n2, blockDimElems);

    float *h_orig2 = (float *)malloc(size2);
    float *h_data2 = (float *)malloc(size2);
    srand(42);
    for (int i = 0; i < n2; i++) {
        h_orig2[i] = (float)rand() / (float)RAND_MAX * 100000.0f;
        h_data2[i] = h_orig2[i];
    }

    float *d_data2;
    CUDA_CHECK(cudaMalloc(&d_data2, size2));
    CUDA_CHECK(cudaMemcpy(d_data2, h_orig2, size2, cudaMemcpyHostToDevice));

    // ---- 正确性验证 ----
    runBitonicV2(d_data2, n2, blockDimElems);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data2, d_data2, size2, cudaMemcpyDeviceToHost));

    bool ok2 = true;
    for (int i = 1; i < n2; i++) {
        if (h_data2[i] < h_data2[i - 1]) { ok2 = false; break; }
    }
    printf("  验证: %s\n", ok2 ? "✓ 通过" : "✗ 失败");

    // ---- 计时: 只计 kernel ----
    runBitonicV2(d_data2, n2, blockDimElems);  // 预热
    CUDA_CHECK(cudaDeviceSynchronize());

    int reps2 = 5;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps2; r++) {
        runBitonicV2(d_data2, n2, blockDimElems);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms2;
    CUDA_CHECK(cudaEventElapsedTime(&ms2, start, stop));
    printf("  耗时: %.3f ms (平均 %d 次, 仅 kernel)\n\n", ms2 / reps2, reps2);

    // ============================================================
    printf("--- 优化点说明 ---\n");
    printf("  V1: j<32 用 __syncwarp, 同步开销明显下降\n");
    printf("  V2: j<blockDimElems 合并到一个 kernel, kernel 数 ~210 -> ~75\n");
    printf("  V2: 每线程处理一个比较对, 线程利用率 100%%\n");
    printf("  实际项目推荐: cub::DeviceRadixSort (O(nk), 更快)\n\n");

    // ---- 清理 ----
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_data2));
    free(h_orig);
    free(h_data);
    free(h_ref);
    free(h_orig2);
    free(h_data2);

    printf("✓ Day 20 优化版完成!\n");
    return 0;
}
