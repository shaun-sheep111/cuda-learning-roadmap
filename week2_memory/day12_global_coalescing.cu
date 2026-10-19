/*
 * Day 12: 全局内存合并访问 — GPU 性能的关键
 * =============================================
 * 学习目标:
 *   1. 深入理解合并访问 (coalesced access) 的原理
 *   2. 理解为什么非合并访问会慢 10-30 倍
 *   3. 学会识别和修复非合并访问模式
 *   4. 对比 AoS vs SoA 数据布局的性能差异
 *
 * 关键概念:
 *   - 一个 warp (32 threads) 同时执行
 *   - 合并访问: warp 中连续 threads 访问连续地址 → 1 次内存事务
 *   - 非合并访问: warp 中 threads 访问非连续地址 → 多次内存事务
 *   - 最差情况: stride 访问 → 32 次事务
 *   - cuda13.0编译
 * 编译: nvcc day12_global_coalescing.cu -o day12
 * 运行: ./day12
 */

#include <stdio.h>
#define BLOCK_SIZE 256
#define TILE_DIM   32    // transposeRead 的 tile 边长 (32x33 float = 4224 B)
// ============================================================
// Kernel 1: 合并访问 — 连续线程读连续地址
// thread 0 读 array[0], thread 1 读 array[1], ...
// ============================================================
__global__ void coalescedRead(const float *input, float *output, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        output[idx] = input[idx];
    }
}

// ============================================================
// Kernel 2: 非合并访问 — stride 访问
// thread 0 读 array[0], thread 1 读 array[STRIDE], ...
// 这模拟了常见的非合并访问模式
// ============================================================
__global__ void stridedRead(const float *input, float *output, int n, int stride){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int accessIdx = idx * stride; //非连续访问
    if (accessIdx < n){
        output[idx] = input[accessIdx] * 2.0f;
    }
}

// ============================================================
// Kernel 3: AoS (Array of Structures) 访问
// struct { float x, y, z; } — 访问 x 时, y,z 被浪费
// ============================================================
struct float3_aos {
    float x, y, z;
};

__global__ void processAoS(const float3_aos *input, float3_aos *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 只需要 x, 但整个 struct 被加载 (warp 中每3个float只用到1个)
        output[idx].x = input[idx].x * 2.0f;
        output[idx].y = input[idx].y;
        output[idx].z = input[idx].z;
    }
}

// ============================================================
// Kernel 4: SoA (Structure of Arrays) 访问
// 三个独立数组 x[], y[], z[] — 访问 x 时完全合并
// ============================================================
struct float3_soa {
    float *x;
    float *y;
    float *z;
};

__global__ void processSoA(float3_soa input, float3_soa output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 访问 x[] 完全合并
        output.x[idx] = input.x[idx] * 2.0f;
        output.y[idx] = input.y[idx];
        output.z[idx] = input.z[idx];
    }
}

// ============================================================
// Kernel 5: 转置读取模式 (通过共享内存修复非合并)
// 原始模式: warp 中各线程读跨行 (stride=N), 非合并
// 优化: 先合并读入共享内存, 再转置读取
// ============================================================
__global__ void transposeRead(const float *matrix, float *colSums,
                                int rows, int cols){
    __shared__ float sdata[TILE_DIM  * TILE_DIM]; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    // 合并读: 每 thread 读一个元素
    if (row < rows && col < cols){
        sdata[ty * TILE_DIM + tx] = matrix[row * cols + col];
    }
    else {
        sdata[ty * TILE_DIM + tx] = 0.0f;
    }
    __syncthreads();

    // 沿列方向求和 (每个 thread 对应一列)
    // 注意: 这里 sdata[tx * TILE_DIM + ty] 是转置读
    if (tx < TILE_DIM && ty == 0) {
        float sum = 0.0f;
        for (int i = 0; i < TILE_DIM; i++) {
            sum += sdata[i * TILE_DIM + tx];
        }
        // 累加到全局结果
        atomicAdd(&colSums[blockIdx.x * TILE_DIM + tx], sum);
    }

}

int main() {
    printf("============================================\n");
    printf("   Day 12: 全局内存合并访问\n");
    printf("============================================\n\n");

    int n = 1 << 22;  // 4M
    size_t size = n * sizeof(float);

    float *h_data = (float *)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = (float)i;
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_data, size, cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);cudaEventCreate(&stop);

     // --- 合并访问 ---
    printf("--- 合并访问 (stride=1) ---\n");
    {
        coalescedRead<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 500; i++)
            coalescedRead<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  耗时: %.4f ms\n", ms / 500);
        printf("  有效带宽: %.2f GB/s\n\n", (2.0f * size) / (ms / 500 * 1e6));
    }

    // --- 不同 stride 的非合并访问 ---
    int strides[] = {2, 4, 8, 16, 32};
    for (int s = 0; s < 5; s++) {
        int stride = strides[s];
        int effectiveN = n / stride;
        int gs = (effectiveN + BLOCK_SIZE - 1) / BLOCK_SIZE;

        stridedRead<<<gs, BLOCK_SIZE>>>(d_in, d_out, n, stride);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 500; i++)
            stridedRead<<<gs, BLOCK_SIZE>>>(d_in, d_out, n, stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("--- 非合并访问 (stride=%2d) ---\n", stride);
        printf("  耗时: %.4f ms\n", ms / 500);
        printf("  有效带宽: %.2f GB/s\n", (2.0f * effectiveN * sizeof(float)) / (ms / 500 * 1e6));
        printf("  速度降低: %.1fx\n\n", (ms / 500) / ((2.0f * size) / (ms / 500 * 1e6) * 0 + 0.001));
    }

    // --- AoS vs SoA ---
    printf("--- AoS vs SoA (float3 结构体) ---\n\n");

    // AoS
    {
        int numElements = n / 3;
        size_t aosSize = numElements * sizeof(float3_aos);
        float3_aos *d_aos_in, *d_aos_out;
        cudaMalloc(&d_aos_in, aosSize);
        cudaMalloc(&d_aos_out, aosSize);

        float3_aos *h_aos = (float3_aos *)malloc(aosSize);
        for (int i = 0; i < numElements; i++) {
            h_aos[i].x = i; h_aos[i].y = i + 1; h_aos[i].z = i + 2;
        }
        cudaMemcpy(d_aos_in, h_aos, aosSize, cudaMemcpyHostToDevice);

        int gs = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        processAoS<<<gs, BLOCK_SIZE>>>(d_aos_in, d_aos_out, numElements);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 500; i++)
            processAoS<<<gs, BLOCK_SIZE>>>(d_aos_in, d_aos_out, numElements);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  AoS 耗时: %.4f ms (带宽 %.2f GB/s)\n",
               ms / 500, (2.0f * aosSize) / (ms / 500 * 1e6));

        cudaFree(d_aos_in); cudaFree(d_aos_out);
        free(h_aos);
    }

    // SoA
    {
        int numElements = n / 3;
        size_t arrSize = numElements * sizeof(float);
        float *d_x, *d_y, *d_z, *d_ox, *d_oy, *d_oz;
        cudaMalloc(&d_x, arrSize); cudaMalloc(&d_y, arrSize); cudaMalloc(&d_z, arrSize);
        cudaMalloc(&d_ox, arrSize); cudaMalloc(&d_oy, arrSize); cudaMalloc(&d_oz, arrSize);

        for (int i = 0; i < numElements; i++) {
            h_data[i] = i; h_data[numElements + i] = i + 1; h_data[2 * numElements + i] = i + 2;
        }
        cudaMemcpy(d_x, h_data, arrSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_data + numElements, arrSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, h_data + 2 * numElements, arrSize, cudaMemcpyHostToDevice);

        float3_soa in = {d_x, d_y, d_z};
        float3_soa out = {d_ox, d_oy, d_oz};
        int gs = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        processSoA<<<gs, BLOCK_SIZE>>>(in, out, numElements);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 500; i++)
            processSoA<<<gs, BLOCK_SIZE>>>(in, out, numElements);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  SoA 耗时: %.4f ms (带宽 %.2f GB/s)\n\n",
               ms / 500, (6.0f * arrSize) / (ms / 500 * 1e6));

        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
        cudaFree(d_ox); cudaFree(d_oy); cudaFree(d_oz);
    }

    printf("--- 合并访问原则 ---\n");
    printf("  ✓ warp 中连续 threads 访问连续地址\n");
    printf("  ✓ 对齐访问 (起始地址 128B 对齐)\n");
    printf("  ✓ 优先 SoA 数据布局\n");
    printf("  ✗ 避免 stride 访问 (如读列)\n");
    printf("  ✗ 避免 AoS 中只需要部分字段的情况\n");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out);
    free(h_data);

    printf("\n✓ Day 12 完成! 合并访问是 GPU 内存优化的第一法则。\n");
    return 0;
}