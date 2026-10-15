/*
 * Day 10: 常量内存 — 优化只读数据访问
 * =======================================
 * 学习目标:
 *   1. 理解常量内存的特点和使用场景
 *   2. 掌握 __constant__ 声明和 cudaMemcpyToSymbol
 *   3. 理解常量缓存 (constant cache) 的工作原理
 *   4. 对比常量内存 vs 全局内存 vs 共享内存的性能
 *
 * 常量内存特点:
 *   - 总共 64KB, 所有 SM 共享
 *   - 只读 (device 代码不能修改)
 *   - 有专用缓存, 延迟类似共享内存 (~5 cycle)
 *   - 广播机制: 当一个 warp 中所有线程读同一地址时, 只需 1 次访问
 *   - 如果 warp 中不同线程读不同地址, 会串行化 (比全局内存还慢!)
 *   - cuda13.0版本编译
 * 最佳场景: 所有线程读取相同的少量只读数据
 * 典型用途: 卷积核、变换矩阵、查找表
 *
 * 编译: nvcc day10_constant_memory.cu -o day10
 * 运行: ./day10
 */

#include <stdio.h>

#define N 1<<22
#define MASK_SIZE 16
#define BLOCK_SIZE 256

// ============================================================
// 常量内存声明 (全局作用域)
// ============================================================
__constant__ float c_mask[MASK_SIZE];
__constant__ float c_transform[4];  // 2x2 变换矩阵

// ============================================================
// Kernel 1: 使用常量内存的 2D 变换
// 每个线程读取相同的变换矩阵 → 广播, 高效
// ============================================================

__global__ void transform2D(const float *input, float *output, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 + 1 >= n) return;

    float x = input[idx * 2];
    float y = input[idx * 2 + 1];

    // 所有线程读取相同的 c_transform → 常量缓存广播
    // c_transform[0], c_transform[1], c_transform[2], c_transform[3]
    output[idx * 2] = c_transform[0] * x + c_transform[1] * y;
    output[idx * 2 + 1] = c_transform[2] * x + c_transform[3] * y;
}

// ============================================================
// Kernel 2: 使用全局内存的 2D 变换 (对比用)
// ============================================================
__global__ void transform2DGlobal(const float *input, float *output, const float *transform, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 + 1 >= n) return;

    float x = input[idx * 2];
    float y = input[idx * 2 + 1];

    output[idx * 2]     = transform[0] * x + transform[1] * y;
    output[idx * 2 + 1] = transform[2] * x + transform[3] * y;
}

// ============================================================
// Kernel 3: 使用常量内存的 FIR 滤波器
// 卷积核放在常量内存, 所有线程共享
// ============================================================
__global__ void firFilter(const float *input, float *output, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < MASK_SIZE; i++) {
        int sIdx = idx + i - MASK_SIZE / 2;
        sIdx = max(0, min(n - 1, sIdx));
        // c_mask[i] 被 warp 中所有线程同时读取 → 广播
        sum += input[sIdx] * c_mask[i];
    }
    output[idx] = sum;
}

// ============================================================
// Kernel 4: 使用全局内存的 FIR 滤波器 (对比用)
// ============================================================
__global__ void firFilterGlobal(const float *input, float *output,
                                const float *mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < MASK_SIZE; i++) {
        int sIdx = idx + i - MASK_SIZE / 2;
        sIdx = max(0, min(n - 1, sIdx));
        sum += input[sIdx] * mask[i];
    }
    output[idx] = sum;
}

int main()
{
    printf("============================================\n");
    printf("   Day 10: 常量内存\n");
    printf("============================================\n\n");

    int n = 1 << 22;
    size_t size = n * sizeof(float);

    float *h_input  = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float h_mask[MASK_SIZE];

    //生成高斯滤波核
    for (int i = 0; i < MASK_SIZE; i++){
        float x = (float)(i - MASK_SIZE / 2);
        h_mask[i] = expf(-x * x / 8.0f);
    }
    // 归一化
    float maskSum = 0.0f;
    for(int i = 0; i < MASK_SIZE; i++) maskSum += h_mask[i];
    for (int i = 0; i < MASK_SIZE; i++) h_mask[i] /= maskSum;

    for (int i = 0; i < n; i++){
        h_input[i] = sinf(i * 0.01f) + (rand() % 100 - 50) * 0.01f;
    }

    // 变换矩阵(旋转30度)
    float h_transform[4] = {
        0.866f, -0.5f, // cos(30), -sin(30)
        0.5f, 0.866f  // sim(30), cos(30)
    };

    float *d_input, *d_output, *d_mask, *d_transform;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_mask, MASK_SIZE * sizeof(float));
    cudaMalloc(&d_transform, 4 * sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, MASK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transform, h_transform, 4 * sizeof(float), cudaMemcpyHostToDevice);

    // 拷贝到常量内存
    cudaMemcpyToSymbol(c_mask, h_mask, MASK_SIZE * sizeof(float));
    cudaMemcpyToSymbol(c_transform, h_transform, 4 * sizeof(float));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // --- FIR 滤波器对比 ---
    printf("--- FIR 滤波器 (%d tap) ---\n\n", MASK_SIZE);
    printf(" 长量内存版本:\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        firFilter<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 200; i++)
            firFilter<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("    耗时: %.3f ms\n", ms / 200);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("  全局内存版本:\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        firFilterGlobal<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_mask, n);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 200; i++)
            firFilterGlobal<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_mask, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("    耗时: %.3f ms\n", ms / 200);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // --- 2D 变换对比 ---
    printf("\n--- 2D 变换 ---\n\n");

    printf("  常量内存版本:\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        transform2D<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 200; i++)
            transform2D<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("    耗时: %.3f ms\n", ms / 200);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("  全局内存版本:\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        transform2DGlobal<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_transform, n);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 200; i++)
            transform2DGlobal<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, d_transform, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("    耗时: %.3f ms\n", ms / 200);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("\n--- 常量内存使用原则 ---\n");
    printf("  ✓ 适合: 所有线程读取相同数据 (广播)\n");
    printf("  ✓ 适合: 数据量小 (<64KB) 且只读\n");
    printf("  ✗ 不适合: 不同线程读不同地址 (会串行化)\n");
    printf("  ✗ 不适合: 需要写入的数据\n");

    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_mask); cudaFree(d_transform);
    free(h_input); free(h_output);

    printf("\n✓ Day 10 完成! 常量内存是只读广播数据的最佳选择。\n");
    return 0;

}
