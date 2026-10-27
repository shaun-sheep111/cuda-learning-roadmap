/*
 * Day 23: Warp Shuffle 进阶 — 高效数据交换 (fixed)
 * ============================================
 * 学习目标:
 *   1. 深入掌握 __shfl_sync 的各种变体
 *   2. 使用 shuffle 实现矩阵转置 (无需共享内存)
 *   3. 实现跨 warp 的广播和归约
 *   4. 理解 shuffle vs 共享内存的性能差异
 *   5. nvcc13.0编译
 *   编译: nvcc day23_warp_shuffle.cu -o day23
 *   运行: ./day23
 *   
 * 注意: 本文件中的 transposeShuffle 要求 n 为 32 的倍数。
 *       这里 n = 1<<22 满足条件。
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

// ============================================================
// 1. 4x8 矩阵转置 (使用 warp shuffle)
// ============================================================
__global__ void transposeShuffle(const float *input, float *output, int n) {
    int idx  = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;

    // 条件读取,避免warp内部分线程提前 return 导致 shfl mask 违规
    float val = (idx < n) ? input[idx] : 0.0f;

    // 4x8 -> 8x4 转置:
    //   原位置 p 对应的转置目标位置 f(p) = (p%8)*4 + p/8
    //   本线程需要 output[lane] = input[f^-1(lane)]
    //   f^-1(lane) = (lane%4)*8 + lane/4
    int srcLane = (lane % 4) * 8 + lane / 4;

    float result = __shfl_sync(0xffffffff, val, srcLane);

    if (idx < n) output[idx] = result;
}

// ============================================================
// 2. Warp 内全归约 — 使用 __shfl_xor_sync
// ============================================================
__device__ float warpReduceXor(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__global__ void reduceXorKernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    val = warpReduceXor(val);

    int warpStart = blockIdx.x * BLOCK_SIZE
                  + (threadIdx.x / WARP_SIZE) * WARP_SIZE;
    if ((threadIdx.x % WARP_SIZE) == 0 && warpStart < n) {
        output[blockIdx.x * (BLOCK_SIZE / WARP_SIZE)
             + threadIdx.x / WARP_SIZE] = val;
    }
}

// ============================================================
// 3. Block 级归约 (warp shuffle + 共享内存)
// ============================================================
__global__ void blockReduceShuffle(const float *input, float *output, int n){
    __shared__ float warpResults[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;

    // 关键修复: 不提前 return，避免 __syncthreads() 死锁
    float val = (idx < n) ? input[idx] : 0.0f;

    val = warpReduceXor(val);

    int warpId = tid / WARP_SIZE;
    int lane   = tid % WARP_SIZE;
    if (lane == 0){
        warpResults[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0){
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? warpResults[lane] : 0.0f;
        val = warpReduceXor(val);

        if (lane == 0 && blockIdx.x * BLOCK_SIZE < n) {
            output[blockIdx.x] = val;
         }
    }
}

// ============================================================
// 4. 使用 shuffle 实现 warp 内广播
// ============================================================
__global__ void broadcastShuffle(const float *input, float *output, int n) {
    int idx  = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;

    float val = (idx < n) ? input[idx] : 0.0f;

    float maxVal = val;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        maxVal = fmaxf(maxVal, __shfl_down_sync(0xffffffff, maxVal, offset));
    }

    float broadcasted = __shfl_sync(0xffffffff, maxVal, 0);
    if (idx < n) output[idx] = broadcasted;
}

// ============================================================
// 5. Warp 内前缀和 (使用 __shfl_up_sync)
// ============================================================
__device__ float warpInclusiveScan(float val) {
    int lane = threadIdx.x % WARP_SIZE;

    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float nb = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += nb;
    }
    return val;
}

__global__ void scanKernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;
    val = warpInclusiveScan(val);
    if (idx < n) output[idx] = val;
}

// ============================================================
int main() {
    printf("============================================\n");
    printf("   Day 23: Warp Shuffle 进阶 (fixed)\n");
    printf("============================================\n\n");

    int n = 1 << 22;
    size_t size = n * sizeof(float);

    float *h_input = (float *)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = (float)(i % 100) / 100.0f;

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // --- XOR 归约 ---
    printf("--- Warp XOR 归约 ---\n");
    {
        reduceXorKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 200; i++)
            reduceXorKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  耗时: %.4f ms (200次平均)\n", ms / 200);
        printf("  吞吐: %.2f GB/s\n\n", size / (ms / 200 * 1e6));
    }

    // --- Block 归约 ---
    printf("--- Block 归约 (shuffle + shared) ---\n");
    {
        blockReduceShuffle<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 200; i++)
            blockReduceShuffle<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  耗时: %.4f ms (200次平均)\n", ms / 200);
        printf("  吞吐: %.2f GB/s\n\n", size / (ms / 200 * 1e6));
    }

    // --- Warp scan ---
    printf("--- Warp Scan (shuffle) ---\n");
    {
        scanKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 200; i++)
            scanKernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  耗时: %.4f ms (200次平均)\n\n", ms / 200);
    }

    // --- 矩阵转置 ---
    printf("--- 矩阵转置 (shuffle) ---\n");
    {
        transposeShuffle<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 200; i++)
            transposeShuffle<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  耗时: %.4f ms (200次平均)\n\n", ms / 200);
    }

    printf("--- Shuffle vs 共享内存 ---\n");
    printf("  Shuffle 优势:\n");
    printf("    + 延迟更低 (1 cycle vs ~5 cycle)\n");
    printf("    + 不占用共享内存\n");
    printf("    + 代码更简洁\n");
    printf("  共享内存优势:\n");
    printf("    + 支持跨 warp 通信\n");
    printf("    + 支持动态索引访问\n");
    printf("    + 适合不规则访问模式\n");
    printf("  建议: warp 内优先 shuffle, 跨 warp 用共享内存\n");

    cudaFree(d_input); cudaFree(d_output);
    free(h_input);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("\n Day 23 完成! Shuffle 是高性能 CUDA 编程的利器。\n");
    return 0;
}