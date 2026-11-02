/*
 * Day 24: 动态并行 (Dynamic Parallelism) — GPU 启动 GPU
 * ======================================================
 * 学习目标:
 *   1. 理解 CUDA 动态并行的概念和用途
 *   2. 从 device 代码中启动新的 kernel
 *   3. 理解动态并行的限制和开销
 *   4. 实现自适应并行算法
 *
 * 什么是动态并行:
 *   - 传统 CUDA: 只有 host (CPU) 可以启动 kernel
 *   - 动态并行: device (GPU) 也可以启动 kernel
 *   - 需要 compute capability >= 3.5
 *
 * 应用场景:
 *   - 自适应网格: 根据数据动态决定并行度
 *   - 递归算法: 快排、树遍历
 *   - 不规则计算: 网格细化、自适应采样
 *
 * 注意事项:
 *   - 动态启动有开销 (不要频繁启动小 kernel)
 *   - 需要编译选项: -rdc=true
 *   - 子 kernel 与父 kernel 异步
 *   - cudaEvent 只跟踪父 kernel；计时/取结果前要 cudaDeviceSynchronize()
 *   - nvcc13.0
 * 编译: nvcc -arch=sm_89 -rdc=true week4_advanced/day24_dynamic_parallelism.cu -o day24 -lcudadevrt
 * 运行: ./day24
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_LEVELS   32
#define REDUCE_BLOCK 256

// ============================================================
// 子 kernel: 处理一段数据
// ============================================================
__global__ void processChunk(float *data, int start, int end){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int globalIdx = start + idx;
    if (globalIdx >= end) return;

    // 简单计算: 平方 + 缩放
    data[globalIdx] = data[globalIdx] * data[globalIdx] * 0.5f;
}

// ============================================================
// 父 kernel: 根据数据特征动态决定子 kernel 的并行度
// ============================================================
__global__ void parentKernel(float *data, int n, int chunkSize) {
    int chunkId = blockIdx.x;

    int start = chunkId * chunkSize;
    if (start >= n) return;
    int end = start + chunkSize;
    if (end > n) end = n;
    int len = end - start;

    // 简化: 用第一个元素估计chunk的轻重
    float firstVal = data[start];

    int childBlockSize;
    int childGridSize;

    if (firstVal > 0.5f){
        // "重" chunk: 更多线程
        childBlockSize = 256;
        childGridSize  = (len + childBlockSize - 1) / childBlockSize;
    } else {
        // "轻" chunk: 较少线程
        childBlockSize = 64;
        childGridSize  = (len + childBlockSize - 1) / childBlockSize;
    }

    // 动态启动子kernel
    processChunk<<<childGridSize, childBlockSize>>>(data, start, end);
}

// ============================================================
// 递归归约: 动态并行版
//
// 正确性要点:
//   1. bufIn / bufOut 分离，绝不 in-place 覆盖别的 block 的输入。
//   2. 每个 block 写 bufOut[blockIdx.x] 后 __threadfence()，
//      再 atomicAdd(counter[level])。
//   3. atomicAdd 返回 gridDim.x - 1 的 block 是"最后一个"，
//      它启动下一级子 kernel。
//   4. 每级 blockDim 固定为 2 的幂 (REDUCE_BLOCK)，
//      否则标准 shared-memory 归约在非 2 的幂时会漏元素。
//   5. 最后一级 gridDim == 1 时直接把结果写入 result。
// ============================================================
__global__ void recursiveReduce(float *bufIn, float *bufOut, float *result,
                                int n, int *counter, int level){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? bufIn[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x /2; stride > 0; stride >>= 1){
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0){
        float s = sdata[0];

        if (gridDim.x == 1){
            // 最后一级: 直接输出最终结果
            *result = s;
        } else {
            bufOut[blockIdx.x] = s;
            __threadfence();  // 确保 bufOut 的写对其他 block 可见

            int old = atomicAdd(&counter[level], 1);
            if (old == gridDim.x - 1){
                // 我是最后一个完成的 block
                int remaining = gridDim.x;
                int childGrid = (remaining + REDUCE_BLOCK - 1) / REDUCE_BLOCK;

                // 为下一级重置计数器
                counter[level + 1] = 0;
                __threadfence();

                recursiveReduce<<<childGrid, REDUCE_BLOCK,
                                  REDUCE_BLOCK * sizeof(float)>>>(
                    bufOut, bufIn, result, remaining, counter, level + 1);
            }
        }
    }
}


// ============================================================
// 常规归约 (对比用): 只做第一阶段
// ============================================================
__global__ void standardReduce(const float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main(){
    printf("============================================\n");
    printf("   Day 24: 动态并行 (Dynamic Parallelism)\n");
    printf("============================================\n\n");

    // ---- 检查设备是否支持动态并行 ----
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0){
        printf("  ✗ 没有找到 CUDA 设备\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)){
        printf("  ✗ 设备计算能力 %d.%d, 动态并行需要 >= 3.5\n",
            prop.major, prop.minor);
        return 1;
    }
    printf("设备: %s (计算能力 %d.%d)\n\n", prop.name, prop.major, prop.minor);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // ========================================================
    // Demo 1: 自适应 chunk 处理
    // ========================================================
    printf("--- Demo 1: 自适应 chunk 处理 ---\n");
    {
        int n = 1 << 20;
        size_t size = n * sizeof(float);
        int chunkSize = 4096;
        int numChunks = (n + chunkSize - 1) / chunkSize;

        float *h_data = (float *)malloc(size);
        srand(12345);
        for (int i = 0; i < n; i++) h_data[i] = (float)(rand() % 100) / 100.0f;

        float *d_data;
        cudaMalloc(&d_data, size);
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        parentKernel<<<numChunks, 1>>>(d_data, n, chunkSize);
        // 关键: 等待父 kernel 以及它动态启动的所有子 kernel 完成
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

        printf("  数据量: %d, chunk 数: %d\n", n, numChunks);
        printf("  耗时: %.3f ms (含动态启动开销)\n", ms);
        printf("  示例输出: data[0]=%.4f, data[%d]=%.4f\n\n",
               h_data[0], n - 1, h_data[n - 1]);

        cudaFree(d_data);
        free(h_data);
    }

    // ========================================================
    // Demo 2: 递归归约 (动态并行)
    // ========================================================
    printf("--- Demo 2: 递归归约 (动态并行) ---\n");
    {
        int n2 = 1 << 16;
        size_t size2 = n2 * sizeof(float);
        int blockSize = REDUCE_BLOCK;
        int gridSize  = (n2 + blockSize - 1) / blockSize;

        float *h_data2 = (float *)malloc(size2);
        for (int i = 0; i < n2; i++) h_data2[i] = 1.0f;  // 全 1, 期望 n2

        double cpuSum = 0.0;
        for (int i = 0; i < n2; i++) cpuSum += h_data2[i];

        float *d_bufA = nullptr;
        float *d_bufB = nullptr;
        float *d_result = nullptr;
        int   *d_counter = nullptr;

        cudaMalloc(&d_bufA,    size2);
        cudaMalloc(&d_bufB,    (size_t)gridSize * sizeof(float));
        cudaMalloc(&d_result,  sizeof(float));
        cudaMalloc(&d_counter, MAX_LEVELS * sizeof(int));
        cudaMemset(d_counter, 0, MAX_LEVELS * sizeof(int));
        cudaMemcpy(d_bufA, h_data2, size2, cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        recursiveReduce<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
            d_bufA, d_bufB, d_result, n2, d_counter, 0);
        // 等待父 kernel 以及所有递归子 kernel 完成
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float gpuSum = 0.0f;
        cudaMemcpy(&gpuSum, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        printf("  数据量: %d (全1)\n", n2);
        printf("  GPU 结果: %f (期望 %d)\n", gpuSum, n2);
        printf("  耗时: %.3f ms\n", ms);
        printf("  验证: %s\n\n",
               fabsf(gpuSum - (float)cpuSum) < 1.0f ? "✓ 通过" : "✗ 失败");

        cudaFree(d_bufA);
        cudaFree(d_bufB);
        cudaFree(d_result);
        cudaFree(d_counter);
        free(h_data2);
    }

    printf("--- 动态并行使用建议 ---\n");
    printf("  ✓ 适合: 自适应算法、递归、不规则数据\n");
    printf("  ✓ 适合: 需要在 GPU 上做调度决策\n");
    printf("  ✓ 必须: 子 kernel 启动前用 __threadfence() + atomic 做 grid 同步\n");
    printf("  ✓ 必须: 计时/取结果前 cudaDeviceSynchronize()，event 不等子 kernel\n");
    printf("  ✗ 避免: 频繁启动小 kernel (启动开销大)\n");
    printf("  ✗ 避免: 可以用固定 grid 解决的问题\n");
    printf("  注意: 需要 -arch=sm_35+ -rdc=true 编译\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n✓ Day 24 完成! 动态并行让 GPU 具备了自主调度能力。\n");
    return 0;
}