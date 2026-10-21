/*
 * Day 16: 前缀和 (Scan) — 并行计算的核心算法
 * ============================================
 * 学习目标:
 *   1. 理解前缀和 (prefix sum / scan) 的定义和重要性
 *   2. 实现 Blelloch scan (work-efficient, up-sweep + down-sweep)
 *   3. 理解 inclusive vs exclusive scan
 *   4. 学习 block scan + global scan 的两阶段策略
 *
 * 前缀和定义:
 *   exclusive: out[i] = in[0] + in[1] + ... + in[i-1], out[0] = 0
 *   inclusive: out[i] = in[0] + in[1] + ... + in[i], out[0] = in[0]
 *
 * 应用场景:
 *   - compact (过滤): 需要计算每个元素的输出位置
 *   - radix sort: 统计每个桶中元素的数量
 *   - 流压缩、去重
 *   - nvcc13.0
 * 编译: nvcc day16_prefix_sum.cu -o day16
 * 运行: ./day16
 */

#include <stdio.h>

#define BLOCK_SIZE 512

// ============================================================
// V1: 朴素 inclusive scan (Hillis-Steele)
// 每步 stride 翻倍, 总共 log2(N) 步
// work: O(N log N), step: O(log N)
// 适合: 追求最少步骤数, work 不高效
// ============================================================
__global__ void scanInclusiveNaive(const float *input, float *output,
                                     float *blockSums, int n){
    __shared__ float sdata[BLOCK_SIZE * 2]; //双缓冲
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Hillis-Steele inclusive scan
    int pout = 0, pin = 1;
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2){
        pout = 1 - pout;
        pin = 1- pout;

        if (tid >= stride)
        {
            sdata[pout * BLOCK_SIZE + tid] = sdata[pin * BLOCK_SIZE + tid] +
                                                sdata[pin * BLOCK_SIZE + tid - stride];
            } else {
                sdata[pout * BLOCK_SIZE + tid] = sdata[pin * BLOCK_SIZE + tid];
        }
        __syncthreads();
    }

    // 写出结果
    if (idx < n)
    {
        output[idx] = sdata[pout * BLOCK_SIZE + tid];
    }

    // block 的总和用于全局scan
    if (blockSums != NULL && tid == BLOCK_SIZE - 1){
        blockSums[blockIdx.x] = sdata[pout * BLOCK_SIZE + tid];
    }
}

// ============================================================
// V2: Blelloch scan (work-efficient, exclusive)
// up-sweep (归约) + down-sweep (分发)
// work: O(N), step: O(log N)
// 更高效但实现更复杂
// ============================================================
__global__ void scanExclusiveBlelloch(const float *input, float *output,
                                        float *blockSums, int n) {
    __shared__ float sdata[BLOCK_SIZE * 2]; // 双缓冲
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //加载
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // --- Up-sweep (reducion phase) ---
    int stride = 1;
    while (stride < BLOCK_SIZE) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            sdata[index] += sdata[index - stride];
        }
        stride *= 2;
        __syncthreads();
    }

    // 保存 block sum,清零最后一个元素
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[BLOCK_SIZE - 1];
        sdata[BLOCK_SIZE - 1] = 0.0f;
    }
    __syncthreads();

    // --- Down-sweep (distribution phase) ---
    stride = BLOCK_SIZE / 2;
    while (stride > 0) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            float tmp = sdata[index - stride];
            sdata[index - stride] = sdata[index];
            sdata[index] += tmp;
        }
        stride >>= 1;
        __syncthreads();
    }

    // 写出结果
    if (idx < n) {
        output[idx] = sdata[tid];
    }
}

// 给每个 block 的结果加上前面所有 block 的和
__global__ void addBlockSums(float *output, const float *blockSums, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && idx < n) {
        // 累加前面所有 block 的和
        float prefix = 0.0f;
        for (int i = 0 ; i < blockIdx.x; i++){
            prefix += blockSums[i];
        }
        output[idx] += prefix;
    }
}

// CPU 参考实现
void scanCPU(const float *input, float *out, int n){
    out[0] = 0.0f;
    for (int i = 1; i < n; i++){
        out[i] = out[i - 1] + input[i - 1];
    }
}

int main()
{
    printf("============================================\n");
    printf("   Day 16: 前缀和 (Scan)\n");
    printf("============================================\n\n");

    int n = 1 << 22; // 4M elements
    size_t size = n * sizeof(float);
    printf("数据量: %d 元素\n\n", n);

    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float *h_ref = (float *)malloc(size);

    for (int i = 0; i < n; i++) h_input[i] = 1.0f;  // 全1, 方便验证

    scanCPU(h_input, h_ref, n);
    printf("CPU 参考: scan[%d] = %f (期望 %d)\n\n", n - 1, h_ref[n - 1], n - 1);

    float *d_input, *d_output, *d_blockSums;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_blockSums, gridSize * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // --- Hillis-Steele ---
    printf("--- Hillis-Steele (inclusive, 朴素) ---\n");
    {
        scanInclusiveNaive<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 100; i++){
            scanInclusiveNaive<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        // 加上block sums
        addBlockSums<<<gridSize, BLOCK_SIZE>>>(d_output, d_blockSums, n);
        
        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        bool ok = true;
        for(int i = 0; i < n; i++){
            // inclusive scan of all 1s should give i+1
            if (fabs(h_output[i] - (i + 1)) > 1e-3f) {ok = false; break;}
        }

        printf(" 耗时: %.4f ms (50次平均, kernel only)\n", ms / 50);
        printf("  验证: %s\n\n", ok ? "✓ 通过" : "✗ 失败");
    } 
    
    // --- Blelloch ---
    printf("--- Blelloch (exclusive, work-efficient) ---\n");
    {
        scanExclusiveBlelloch<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 50; i++)
            scanExclusiveBlelloch<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_blockSums, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        // 加上 block sums
        addBlockSums<<<gridSize, BLOCK_SIZE>>>(d_output, d_blockSums, n);

        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (int i = 0; i < n; i++) {
            // exclusive scan of all 1s should give i
            if (fabs(h_output[i] - (float)i) > 1e-3f) { ok = false; break; }
        }

        printf("  耗时: %.4f ms (50次平均, kernel only)\n", ms / 50);
        printf("  验证: %s\n\n", ok ? "✓ 通过" : "✗ 失败");
    }

    printf("--- 两种 scan 对比 ---\n");
    printf("  Hillis-Steele: 步数少(O(logN)), 工作量多(O(NlogN))\n");
    printf("  Blelloch:      步数多(O(logN)), 工作量少(O(N)) ← 更高效\n");
    printf("  实际选择取决于硬件特性\n");

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_blockSums);
    free(h_input); free(h_output); free(h_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("\n✓ Day 16 完成! Scan 是很多并行算法的基础组件。\n");
    return 0;
}

