/*
 * Day 19: 流压缩 (Compact) — Scan 的实战应用
 * ==============================================
 * 学习目标:
 *   1. 理解流压缩: 从数组中过滤满足条件的元素
 *   2. 使用 scan 计算每个元素的输出位置
 *   3. 实现 "predicate → scan → scatter" 三步流水线
 *   4. 理解为什么需要 exclusive scan
 *
 * 流压缩示例:
 *   输入:  [3, 0, 5, 0, 2, 0, 7, 0]
 *   条件:  非零
 *   掩码:  [1, 0, 1, 0, 1, 0, 1, 0]
 *   scan:  [0, 1, 1, 2, 2, 3, 3, 4]  (exclusive)
 *   输出:  [3, 5, 2, 7]  (长度=4)
 *   nvcc13.0
 * 编译: nvcc day19_compact.cu -o day19
 * 运行: ./day19
 */

#include<stdio.h>
#include<stdlib.h>

#define BLOCK_SIZE 256
// BLOCK_SIZE 必须是 2 的幂, 否则 Blelloch scan 的树形结构会失效
static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
              "BLOCK_SIZE must be a power of two for Blelloch scan");

// ============================================================
// CUDA 错误检查宏
// ============================================================
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

// ============================================================
// Step 1: 生成 predicate 数组 (1=保留, 0=丢弃)
// ============================================================
__global__ void generatePredicate(const int *input, int *predicate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        predicate[idx] = (input[idx] % 2 == 0) ? 1 : 0;
    }
}

// ============================================================
// Step 2: Exclusive scan on predicate (计算输出位置)
// 使用 int 而非 float 累积, 避免大 BLOCK_SIZE 时的精度问题
// ============================================================
__global__ void scanPredicate(const int *predicate, int *scanned,
                               int *blockSums, int n) {
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? predicate[idx] : 0;
    __syncthreads();

    // Up-sweep (reduction)
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            sdata[index] += sdata[index - stride];
        }
        __syncthreads();
    }

    // 保存 block sum, 根节点清零 (exclusive scan 的关键)
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[BLOCK_SIZE - 1];
        sdata[BLOCK_SIZE - 1] = 0;
    }
    __syncthreads();

    // Down-sweep
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            int tmp = sdata[index - stride];
            sdata[index - stride] = sdata[index];
            sdata[index] += tmp;
        }
        __syncthreads();
    }

    if (idx < n) {
        scanned[idx] = sdata[tid];
    }
}

// ============================================================
// Step 2.5a: 对 blockSums 再做一次 exclusive scan (单 block 内完成)
// 把 O(n*numBlocks) 的朴素累加变成 O(numBlocks)
// numBlocks 通常远小于 BLOCK_SIZE, 因此这里只用一个 block 就够
// ============================================================
__global__ void scanBlockSums(const int *blockSums, int *scannedBlockSums,
                              int *totalSum, int numBlocks) {
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    sdata[tid] = (tid < numBlocks) ? blockSums[tid] : 0;
    __syncthreads();

    // Up-sweep
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            sdata[index] += sdata[index - stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *totalSum = sdata[BLOCK_SIZE - 1];  // 全局 predicate 总和 = 输出长度
        sdata[BLOCK_SIZE - 1] = 0;
    }
    __syncthreads();

    // Down-sweep
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            int tmp = sdata[index - stride];
            sdata[index - stride] = sdata[index];
            sdata[index] += tmp;
        }
        __syncthreads();
    }

    if (tid < numBlocks) {
        scannedBlockSums[tid] = sdata[tid];
    }
}

// ============================================================
// Step 2.5b: 给每个 block 的 scan 结果加上前缀偏移 (O(1) per element)
// ============================================================
__global__ void addOffsets(int *scanned, const int *scannedBlockSums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        scanned[idx] += scannedBlockSums[blockIdx.x];
    }
}

// ============================================================
// Step 3: Scatter — 将满足条件的元素写入紧凑输出
// ============================================================
__global__ void scatterCompact(const int *input, const int *predicate,
                                const int *scanned, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && predicate[idx] == 1) {
        output[scanned[idx]] = input[idx];
    }
}

int main() {
    printf("============================================\n");
    printf("   Day 19: 流压缩 (Compact)\n");
    printf("============================================\n\n");

    int n = 1 << 20;  // 1M
    size_t size = n * sizeof(int);
    printf("数据量: %d 元素\n", n);

    int *h_input  = (int *)malloc(size);
    int *h_output = (int *)malloc(size);

    srand(42);
    int expectedCount = 0;
    for (int i = 0; i < n; i++) {
        h_input[i] = rand() % 1000;
        if (h_input[i] % 2 == 0) expectedCount++;
    }
    printf("条件: 偶数\n");
    printf("期望输出长度: %d\n\n", expectedCount);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int *d_input = nullptr, *d_predicate = nullptr, *d_scanned = nullptr;
    int *d_output = nullptr, *d_blockSums = nullptr, *d_scannedBlockSums = nullptr;
    int *d_outLen = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_predicate, size));
    CUDA_CHECK(cudaMalloc(&d_scanned, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMalloc(&d_blockSums, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scannedBlockSums, gridSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_outLen, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 执行三步流水线
    CUDA_CHECK(cudaEventRecord(start));

    // Step 1: predicate
    generatePredicate<<<gridSize, BLOCK_SIZE>>>(d_input, d_predicate, n);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: 每个 block 内部做 exclusive scan
    scanPredicate<<<gridSize, BLOCK_SIZE>>>(d_predicate, d_scanned, d_blockSums, n);
    CUDA_CHECK(cudaGetLastError());

    // Step 2.5: 对 blockSums 再做一次 scan 得到前缀偏移, 顺便得到总长度
    scanBlockSums<<<1, BLOCK_SIZE>>>(d_blockSums, d_scannedBlockSums, d_outLen, gridSize);
    CUDA_CHECK(cudaGetLastError());

    // 加上块级偏移
    addOffsets<<<gridSize, BLOCK_SIZE>>>(d_scanned, d_scannedBlockSums, n);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: scatter
    scatterCompact<<<gridSize, BLOCK_SIZE>>>(d_input, d_predicate, d_scanned, d_output, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 获取结果
    int outputLen;
    CUDA_CHECK(cudaMemcpy(&outputLen, d_outLen, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, (size_t)outputLen * sizeof(int),
                          cudaMemcpyDeviceToHost));

    printf("--- 执行结果 ---\n");
    printf("  总耗时: %.3f ms\n", ms);
    printf("  输出长度: %d (期望 %d)\n", outputLen, expectedCount);
    printf("  验证: %s\n", outputLen == expectedCount ? "✓ 长度正确" : "✗ 长度错误");

    // 验证所有输出都是偶数
    bool allEven = true;
    for (int i = 0; i < outputLen; i++) {
        if (h_output[i] % 2 != 0) { allEven = false; break; }
    }
    printf("  所有输出为偶数: %s\n", allEven ? "✓ 是" : "✗ 否");

    // 打印示例
    printf("\n  输入示例: ");
    for (int i = 0; i < 20; i++) printf("%d ", h_input[i]);
    printf("...\n  输出示例: ");
    for (int i = 0; i < 20 && i < outputLen; i++) printf("%d ", h_output[i]);
    printf("...\n");

    printf("\n--- 流压缩流程 ---\n");
    printf("  1. Predicate: 为每个元素生成 0/1 掩码\n");
    printf("  2. Scan:      对掩码做 exclusive prefix sum → 得到输出位置\n");
    printf("  3. Scatter:   只写入掩码=1 的元素到对应位置\n");
    printf("  scan 是连接步骤, 决定了并行 scatter 的正确性\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_predicate));
    CUDA_CHECK(cudaFree(d_scanned));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_scannedBlockSums));
    CUDA_CHECK(cudaFree(d_outLen));
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\n✓ Day 19 完成! 流压缩是 scan 最重要的应用之一。\n");
    return 0;
}