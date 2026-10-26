/*
 * Day 22: Warp 级编程 — 修复版
 * =============================
 * 修复:
 *   1. 补头文件 <cuda_runtime.h> <stdlib.h> <math.h>
 *   2. warpScanShuffle: 所有线程都调用 __shfl_up_sync, 条件累加
 *   3. 所有 _sync 原语: 不提前 return, 用 (idx<n)?val:0 保证 mask 完整
 *   4. 加 CUDA_CHECK / CHECK_LAUNCH
 *   5. 调用 divergenceDemo 并计时对比
 *   6. 清理未使用变量、srand 固定种子
 *   7. nvcc13.0编译
 * 编译: nvcc day22_warp_programming_fixed.cu -o day22
 * 运行: ./day22
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define WARP_SIZE  32
#define BLOCK_SIZE 256

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

// ============================================================
// Demo 1: Warp divergence
// ============================================================
__global__ void divergenceDemo(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    if (warpId == 0) {
        // warp 0: 奇偶 lane 走不同分支 → divergence
        if (laneId % 2 == 0) {
            output[idx] = sinf((float)idx) * cosf((float)idx);
        } else {
            output[idx] = cosf((float)idx) * tanf((float)idx);
        }
    } else {
        // 其他 warp: 无 divergence
        output[idx] = sinf((float)idx) * cosf((float)idx);
    }
}
// ============================================================
// Demo 2: Warp 级归约 (shuffle)
//   注意: 只做 warp 内归约, 输出是每个 warp 的和
//   不提前 return, 用 (idx<n)?val:0 保证 shuffle mask 完整
// ============================================================
__global__ void warpReduceShuffle(const float *input, float *output, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (threadIdx.x % WARP_SIZE == 0) {
        output[blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + threadIdx.x / WARP_SIZE] = val;
    }
}

// ============================================================
// Demo 3: Warp 级 inclusive scan (修复版)
//   关键: 所有线程都必须调用 __shfl_up_sync, 用条件决定是否累加
// ============================================================
__global__ void warpScanShuffle(const float *input, float *output, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float v   = (idx < n) ? input[idx] : 0.0f;
    int lane  = threadIdx.x % WARP_SIZE;

    float t;
    t = __shfl_up_sync(0xffffffff, v, 1);  if (lane >= 1)  v += t;
    t = __shfl_up_sync(0xffffffff, v, 2);  if (lane >= 2)  v += t;
    t = __shfl_up_sync(0xffffffff, v, 4);  if (lane >= 4)  v += t;
    t = __shfl_up_sync(0xffffffff, v, 8);  if (lane >= 8)  v += t;
    t = __shfl_up_sync(0xffffffff, v, 16); if (lane >= 16) v += t;

    if (idx < n) output[idx] = v;
}

// ============================================================
// Demo 4: __ballot_sync
// ============================================================
__global__ void ballotDemo(const float *input, int *warpMasks, int n) {
    int idx   = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lane  = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    // 用 0.0f 填充, 保证所有线程都参与 ballot
    float v = (idx < n) ? input[idx] : 0.0f;
    bool condition = (v > 0.5f);
    unsigned int mask = __ballot_sync(0xffffffff, condition);

    if (lane == 0) {
        warpMasks[blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId] = (int)mask;
    }
}

// ============================================================
// Demo 5: Warp 内最大值 + 广播
// ============================================================
__global__ void broadcastDemo(const float *input, float *output, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = (idx < n) ? input[idx] : -1e30f;

    float maxVal = val;
    maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, 16));
    maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, 8));
    maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, 4));
    maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, 2));
    maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, 1));

    // lane 0 拥有最大值, 广播给全 warp
    float broadcast = __shfl_sync(0xffffffff, maxVal, 0);

    if (idx < n) output[idx] = broadcast;
}

int main() {
    printf("============================================\n");
    printf("   Day 22: Warp 级编程 (修复版)\n");
    printf("============================================\n\n");

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    srand(42);
    float *h_input = (float *)malloc(size);
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(rand() % 1000) / 1000.0f;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Demo 1: Warp divergence 对比 ---
    printf("--- Warp Divergence 演示 ---\n");
    {
        divergenceDemo<<<gridSize, BLOCK_SIZE>>>(d_output, n);
        CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 200; i++)
            divergenceDemo<<<gridSize, BLOCK_SIZE>>>(d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("  耗时: %.4f ms (200次平均)\n", ms / 200);
        printf("  (仅 warp 0 出现 divergence, 其他 warp 无)\n\n");
    }

    // --- Demo 2: Warp shuffle 归约 ---
    printf("--- Warp Shuffle 归约 ---\n");
    {
        warpReduceShuffle<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());

        // 验证第一个 warp 的结果
        float *h_out = (float *)malloc(sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_out, d_output, sizeof(float), cudaMemcpyDeviceToHost));
        float ref = 0.0f;
        for (int i = 0; i < WARP_SIZE; i++) ref += h_input[i];
        printf("  第一个 warp 归约值: %.4f (期望 %.4f)\n",
               h_out[0], ref);
        free(h_out);

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 200; i++)
            warpReduceShuffle<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("  耗时: %.4f ms (200次平均)\n", ms / 200);
        printf("  吞吐: %.2f GB/s\n\n", size / (ms / 200 * 1e6));
    }

    // --- Demo 3: Warp scan (修复版) ---
    printf("--- Warp Scan (修复版) ---\n");
    {
        warpScanShuffle<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());

        // 验证第一个 warp 的 inclusive scan
        float *h_out = (float *)malloc(WARP_SIZE * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_out, d_output, WARP_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("  第一个 warp inclusive scan:\n");
        printf("    输入[0..3]: %.3f %.3f %.3f %.3f\n",
               h_input[0], h_input[1], h_input[2], h_input[3]);
        printf("    输出[0..3]: %.3f %.3f %.3f %.3f\n",
               h_out[0], h_out[1], h_out[2], h_out[3]);
        printf("    输入[31]: %.3f, 输出[31](应为 warp 总和): %.3f\n",
               h_input[31], h_out[31]);

        float ref = 0.0f;
        for (int i = 0; i < WARP_SIZE; i++) ref += h_input[i];
        printf("    warp 总和: %.3f\n", ref);
        free(h_out);

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 200; i++)
            warpScanShuffle<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("  耗时: %.4f ms (200次平均)\n\n", ms / 200);
    }

    // --- Demo 4: __ballot_sync ---
    printf("--- __ballot_sync 演示 ---\n");
    {
        int numWarps = gridSize * (BLOCK_SIZE / WARP_SIZE);
        int *d_masks;
        CUDA_CHECK(cudaMalloc(&d_masks, numWarps * sizeof(int)));

        ballotDemo<<<gridSize, BLOCK_SIZE>>>(d_input, d_masks, n);
        CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());

        int *h_masks = (int *)malloc(8 * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h_masks, d_masks, 8 * sizeof(int),
                              cudaMemcpyDeviceToHost));

        printf("  前 8 个 warp 的掩码 (条件: value > 0.5):\n");
        for (int i = 0; i < 8; i++) {
            printf("    warp %d: 0x%08X (满足 lane 数: %d)\n",
                   i, (unsigned)h_masks[i],
                   __builtin_popcount((unsigned)h_masks[i]));
        }
        printf("\n");

        CUDA_CHECK(cudaFree(d_masks));
        free(h_masks);
    }

    // --- Demo 5: 广播 ---
    printf("--- Warp 广播 (每个 warp 最大值) ---\n");
    {
        broadcastDemo<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());

        float *h_out = (float *)malloc(WARP_SIZE * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_out, d_output, WARP_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
        printf("  第一个 warp 输出:\n    ");
        for (int i = 0; i < 8; i++) printf("%.4f ", h_out[i]);
        printf("...\n");

        float wmax = 0.0f;
        for (int i = 0; i < WARP_SIZE; i++) wmax = fmaxf(wmax, h_input[i]);
        printf("  实际最大值: %.4f\n\n", wmax);
        free(h_out);
    }

    printf("--- Warp 编程要点 ---\n");
    printf("  ✓ Warp 是 GPU 最小执行单元 (32 threads)\n");
    printf("  ✓ __shfl_sync: 寄存器间直接交换, 无共享内存\n");
    printf("  ✓ __ballot_sync: 收集 warp 内条件结果\n");
    printf("  ✓ _sync 原语: mask 内所有线程都必须调用\n");
    printf("  ✓ 避免 divergence: 尽量让 warp 内走相同分支\n");
    printf("  ✓ block size 应为 32 的倍数\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);

    printf("\n✓ Day 22 完成! Warp 级编程是高级 CUDA 优化的关键。\n");
    return 0;
}