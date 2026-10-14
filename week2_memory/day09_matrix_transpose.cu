/*
 * Day 09: 矩阵转置 — 共享内存经典案例
 * =====================================
 * 学习目标:
 *   1. 矩阵转置是共享内存的教科书级应用
 *   2. 理解全局内存合并访问 (coalesced access) 的重要性
 *   3. 理解为什么朴素转置慢 (写不合并)
 *   4. 用共享内存实现合并读写
 *
 * 核心问题:
 *   朴素转置: 读行(合并) → 写列(不合并, stride=N)
 *   共享内存转置: 读行(合并) → 写共享内存 → 读共享内存(转置) → 写行(合并)
 *   cuda13.0版本编译
 *   编译: nvcc day09_matrix_transpose.cu -o day09
 *   运行: ./day09
 */

#include <stdio.h>
#define TILE_SIZE 32

// ============================================================
// Kernel 1: 朴素矩阵转置 (只用全局内存)
// 读: 合并 (连续 thread 读连续列)
// 写: 不合并 (连续 thread 写到 stride=N 的位置)
// ============================================================
__global__ void transposeNaive(const float *input, float *output, int n) {
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;  // 列
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;  // 行

    if (x < n && y < n) {
        // output[x][y] = input[y][x]  即 output[x*n+y] = input[y*n+x]
        output[x * n + y] = input[y * n + x];
    }
}
// ============================================================
// Kernel 2: 共享内存矩阵转置
// 关键: 通过共享内存实现 "读行写行" 的合并模式
// ============================================================
__global__ void transposeShared(const float *input, float *output, int n) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 消除 bank conflict!

    // 坐标计算
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 从全局内存读取 (合并访问) 到共享内存
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    }
    __syncthreads();

    // 转置后的坐标: 交换 blockIdx
    x = blockIdx.y * TILE_SIZE + threadIdx.x;  // 注意: blockIdx.y
    y = blockIdx.x * TILE_SIZE + threadIdx.y;  // 注意: blockIdx.x

    // 从共享内存读取 (转置读) 写入全局内存 (合并访问)
    if (x < n && y < n) {
        output[y * n + x] = tile[threadIdx.x][threadIdx.y];  // 交换 x,y
    }
}

// ============================================================
// Kernel 3: 对角线优化转置
// 处理非方阵或进一步优化时, 可以按对角线遍历 tile
// 这里演示基本原理
// ============================================================
__global__ void transposeDiagonal(const float *input, float *output, int n) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    // 处理对角线 block (blockIdx.x == blockIdx.y)
    // 非对角线 block 需要 swap
    int blockIdx_x, blockIdx_y;
    if (gridDim.x == gridDim.y) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    } else {
        blockIdx_y = blockIdx.x;
        blockIdx_x = blockIdx.y;
    }

    int x = blockIdx_x * TILE_SIZE + threadIdx.x;
    int y = blockIdx_y * TILE_SIZE + threadIdx.y;

    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    }
    __syncthreads();

    x = blockIdx_y * TILE_SIZE + threadIdx.x;
    y = blockIdx_x * TILE_SIZE + threadIdx.y;

    if (x < n && y < n) {
        output[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    printf("============================================\n");
    printf("   Day 09: 矩阵转置\n");
    printf("============================================\n\n");

    // 使用较大的方阵
    int n = 4096;
    size_t size = (size_t)n * n * sizeof(float);
    printf("矩阵: %d x %d (%.2f MB)\n\n", n, n, (float)size / 1e6);

    float *h_in  = (float *)malloc(size);
    float *h_out = (float *)malloc(size);

    for (int i = 0; i < n * n; i++) {
        h_in[i] = (float)i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // --- 朴素版本 ---
    printf("--- 朴素转置 (全局内存) ---\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        transposeNaive<<<gridSize, blockSize>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            transposeNaive<<<gridSize, blockSize>>>(d_in, d_out, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        printf("  耗时: %.3f ms (100次平均)\n", ms / 100);
        printf("  有效带宽: %.2f GB/s (理论峰值需要看设备)\n",
               (2.0f * size) / (ms / 100 * 1e6));
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // --- 共享内存版本 ---
    printf("\n--- 共享内存转置 ---\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        transposeShared<<<gridSize, blockSize>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            transposeShared<<<gridSize, blockSize>>>(d_in, d_out, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        printf("  耗时: %.3f ms (100次平均)\n", ms / 100);
        printf("  有效带宽: %.2f GB/s\n", (2.0f * size) / (ms / 100 * 1e6));
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // --- 对角线优化版本 ---
    printf("\n--- 对角线优化转置 ---\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        transposeDiagonal<<<gridSize, blockSize>>>(d_in, d_out, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 100; i++) {
            transposeDiagonal<<<gridSize, blockSize>>>(d_in, d_out, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        printf("  耗时: %.3f ms (100次平均)\n", ms / 100);
        printf("  有效带宽: %.2f GB/s\n", (2.0f * size) / (ms / 100 * 1e6));
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // 验证
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < 100 && ok; i++) {
        for (int j = 0; j < 100 && ok; j++) {
            if (fabs(h_out[j * n + i] - h_in[i * n + j]) > 1e-3f) ok = false;
        }
    }
    printf("\n验证: %s\n", ok ? "✓ 通过" : "✗ 失败");

    printf("\n--- bank conflict 说明 ---\n");
    printf("  共享内存声明 float tile[32][32] 会有 bank conflict\n");
    printf("  因为转置读取时, 连续线程读同一列 → 同一 bank\n");
    printf("  解决: 加 padding → float tile[32][32+1] 消除冲突\n");
    printf("  本代码已使用 padding 优化\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    printf("\n✓ Day 09 完成! 矩阵转置是共享内存最经典的应用。\n");
    return 0;
}

