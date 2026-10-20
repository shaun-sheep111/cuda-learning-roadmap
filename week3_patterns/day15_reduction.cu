/*
 * Day 15: 并行归约 — 从朴素到最优
 * =================================
 * 学习目标:
 *   1. 深入理解并行归约的多种策略
 *   2. 对比 interleaved vs sequential addressing
 *   3. 理解 warp divergence 对性能的影响
 *   4. 学习 warp 级归约优化 (最后 32 个线程无需同步)
 *
 * 归约是并行计算的基础模式:
 *   sum, max, min, product 等都需要归约
 *   性能差异可达 5-10 倍!
 *
 * 编译: nvcc day15_reduction.cu -o day15
 * 运行: ./day15
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define BLOCK_SIZE 512

#define CUDA_CHECK(call) do {                                                 \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(err_));                \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

// ============================================================
// V1: 朴素归约 — interleaved addressing
// 问题: tid % (2*stride) 导致大量 warp divergence
// ============================================================
__global__ void reduceV1(const float *input, float *output, int n){
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// V2: 朴素归约 — sequential addressing
// 问题: 降低了 warp divergence
// ============================================================
__global__ void reduceV2(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// V3: 加载时做第一次加法，每个 block 处理 2*BLOCK_SIZE 个元素
// ============================================================
__global__ void reduceV3(const float *input, float *output, int n){
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

    float val = 0.0f;
    if (idx < n)                 val += input[idx];
    if (idx + BLOCK_SIZE < n)    val += input[idx + BLOCK_SIZE];
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// V4: Warp 级归约优化 (使用 __shfl_down_sync，适配 Volta+)
// ============================================================
__device__ __forceinline__ float warpReduceSum(float val) {
    // 全 warp 参与 (调用处保证 tid < 32 时 blockDim >= 32)
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__global__ void reduceV4(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

    float val = 0.0f;
    if (idx < n)                 val += input[idx];
    if (idx + BLOCK_SIZE < n)    val += input[idx + BLOCK_SIZE];
    sdata[tid] = val;
    __syncthreads();

    // 只用 shared memory 归约到 stride >= 32
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 最后一个 warp 用 shuffle 归约，无需 __syncthreads()
    if (tid < 32) {
        float v = sdata[tid];
        v = warpReduceSum(v);
        if (tid == 0) sdata[0] = v;
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================
// 最终归约: 跨步循环累加全部部分和
// 原版只读取 data[tid] (tid < 512)，会丢弃大量部分和。
// ============================================================
__global__ void finalReduce(float *data, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    float val = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        val += data[i];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) data[0] = sdata[0];
}

int main() {
    printf("============================================\n");
    printf("   Day 15: 并行归约 — 4种策略对比 (修正版)\n");
    printf("============================================\n\n");

    int n = 1 << 24;  // 16M
    size_t size = n * sizeof(float);
    printf("数据量: %d 元素 (%.2f MB)\n\n", n, (float)size / 1e6);

    srand(12345);
    float *h_input = (float *)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = (float)(rand() % 1000) / 1000.0f;

    double cpuSum = 0;
    for (int i = 0; i < n; i++) cpuSum += h_input[i];
    printf("CPU 参考结果: %f\n\n", (float)cpuSum);

    float *d_input = nullptr;
    float *d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // 每个 block 处理 BLOCK_SIZE 个元素 (V1/V2)
    int gridSize1 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // 每个 block 处理 2*BLOCK_SIZE 个元素 (V3/V4)
    int gridSize2 = (n + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
    printf("V1/V2 grid = %d, V3/V4 grid = %d\n\n", gridSize1, gridSize2);

    CUDA_CHECK(cudaMalloc(&d_partial, gridSize1 * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const char *names[] = {
        "V1 朴素 (interleaved)",
        "V2 改进 (sequential)",
        "V3 加载时加法",
        "V4 Warp归约优化"
    };

    for (int v = 1; v <= 4; v++){
        printf("--- %s ---\n", names[v - 1]);
        int gs = (v <= 2) ? gridSize1 : gridSize2;

        // 预热
        switch (v) {
            case 1: reduceV1<<<gs, BLOCK_SIZE>>>(d_input, d_partial, n); break;
            case 2: reduceV2<<<gs, BLOCK_SIZE>>>(d_input, d_partial, n); break;
            case 3: reduceV3<<<gs, BLOCK_SIZE>>>(d_input, d_partial, n); break;
            case 4: reduceV4<<<gs, BLOCK_SIZE>>>(d_input, d_partial, n); break;
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        const int ITER = 100;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITER; i++) {
            switch (v) {
                case 1: reduceV1<<<gs, BLOCK_SIZE>>>(d_input, d_partial, n); break;
                case 2: reduceV2<<<gs, BLOCK_SIZE>>>(d_input, d_partial, n); break;
                case 3: reduceV3<<<gs, BLOCK_SIZE>>>(d_input, d_partial, n); break;
                case 4: reduceV4<<<gs, BLOCK_SIZE>>>(d_input, d_partial, n); break;
            }
        }
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avgMs = ms / ITER;

        // 对 gs 个部分和做最终归约
        finalReduce<<<1, BLOCK_SIZE>>>(d_partial, gs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float gpuSum = 0.0f;
        CUDA_CHECK(cudaMemcpy(&gpuSum, d_partial, sizeof(float),
                            cudaMemcpyDeviceToHost));

        printf("  GPU 结果: %f\n", gpuSum);
        printf("  误差: %e\n", fabsf(gpuSum - (float)cpuSum));
        printf("  耗时: %.4f ms (100次平均)\n", avgMs);
        printf("  吞吐: %.2f GB/s\n\n", size / (avgMs * 1e6));
    }

    printf("--- 优化效果总结 ---\n");
    printf("  V1→V2: 消除 warp divergence\n");
    printf("  V2→V3: 加载时加法, 减少 block 数量\n");
    printf("  V3→V4: warp shuffle 隐式同步, 省去最后几层 __syncthreads()\n\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));
    free(h_input);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("✓ Day 15 完成! 归约优化是 GPU 编程的经典案例。\n");
    return 0;
}