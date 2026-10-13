/*
 * Day 08: 共享内存基础 — 理解最快的 GPU 内存
 * =============================================
 * 学习目标:
 *   1. 理解 GPU 内存层次: 全局 → L2 → 共享内存 → 寄存器
 *   2. 掌握 __shared__ 声明和共享内存使用模式
 *   3. 理解共享内存的 bank 和 bank conflict
 *   4. 实现共享内存版的向量加法并对比性能
 *
 * 内存层次 (速度从快到慢):
 *   寄存器 (~1 cycle) > 共享内存 (~5 cycle) > L1/L2 缓存 > 全局内存 (~400 cycle)
 *
 * 共享内存特点:
 *   - 每个 SM 有一定量的共享内存 (通常 48KB-100KB)
 *   - 在 block 内共享, 不同 block 之间隔离
 *   - 生命周期与 block 一致
 *   - 被组织为 32 个 bank, 同一 bank 的并发访问会产生 bank conflict
 *
 * 编译: nvcc day08_shared_memory.cu -o day08
 * 运行: ./day08
 */

#include <stdio.h>

#define BLOCK_SIZE 256

// ============================================================
// Kernel 1: 不使用共享内存 — 每个 thread 从全局内存重复读取
// 场景: 计算 Y[i] = X[i] + X[i-1] + X[i+1] (3点平均滤波)
// 每个 X[i] 被周围 3 个线程读取, 全局内存访问冗余
// ============================================================
__global__ void smoothGlobal(const float *X, float *Y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 || idx == n - 1) return;

    Y[idx] = (X[idx - 1] + X[idx] + X[idx + 1]) / 3.0f;
}

// ============================================================
// Kernel 2: 使用共享内存 — block 内协作加载, 减少全局内存访问
// 每个 block 加载一段数据到共享内存, 然后从共享内存读取
// ============================================================
__global__ void smoothShared(const float *X, float *Y, int n) {
    __shared__ float sdata[BLOCK_SIZE + 2];  // +2 for halo

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载中心数据
    if (gid < n) {
        sdata[tid + 1] = X[gid];  // 偏移 1 留给左侧 halo
    }

    // 加载左侧 halo (第一个线程加载前一个 block 的最后一个元素)
    if (tid == 0 && gid > 0) {
        sdata[0] = X[gid - 1];
    } else if (gid == 0) {
        sdata[0] = X[0];  // 边界 clamp
    }

    // 加载右侧 halo (最后一个线程加载下一个 block 的第一个元素)
    if (tid == BLOCK_SIZE - 1 && gid < n - 1) {
        sdata[BLOCK_SIZE + 1] = X[gid + 1];
    } else if (gid == n - 1) {
        sdata[BLOCK_SIZE + 1] = X[n - 1];  // 边界 clamp
    }

    __syncthreads();  // 等待所有数据加载完成

    // 从共享内存计算 (快速!)
    if (gid > 0 && gid < n - 1) {
        Y[gid] = (sdata[tid] + sdata[tid + 1] + sdata[tid + 2]) / 3.0f;
    }
}

// ============================================================
// 演示: 共享内存的 bank conflict
// ============================================================
__global__ void bankConflictDemo(float *out) {
    // 32 个 bank, 每个 bank 宽度 4 bytes (32-bit)
    // sdata[0], sdata[32], sdata[64]... 在 bank 0
    // sdata[1], sdata[33], sdata[65]... 在 bank 1
    __shared__ float sdata[1024];

    int tid = threadIdx.x;

    // 无冲突: 每个 thread 访问不同 bank
    // sdata[tid] -> thread 0 访问 bank 0, thread 1 访问 bank 1, ...
    sdata[tid] = tid * 1.0f;

    // 有冲突: 所有 thread 访问同一 bank
    // sdata[0] -> 所有 thread 访问 bank 0, 32-way conflict!
    // float val = sdata[0];  // 不要这样做!

    __syncthreads();
    out[tid] = sdata[tid];
}

int main() {
    printf("============================================\n");
    printf("   Day 08: 共享内存基础\n");
    printf("============================================\n\n");

    int n = 1 << 22;  // 4M
    size_t size = n * sizeof(float);

    float *h_X = (float *)malloc(size);
    float *h_Y = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        h_X[i] = (float)i;
    }

    float *d_X, *d_Y;
    cudaMalloc(&d_X, size);
    cudaMalloc(&d_Y, size);
    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 测试全局内存版本
    printf("--- 全局内存版本 ---\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        smoothGlobal<<<gridSize, BLOCK_SIZE>>>(d_X, d_Y, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 1000; i++) {
            smoothGlobal<<<gridSize, BLOCK_SIZE>>>(d_X, d_Y, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  耗时: %.4f ms (1000次平均)\n", ms / 1000);
        printf("  有效带宽: %.2f GB/s\n", (3.0f * size) / (ms / 1000 * 1e6));
        // 3.0f 因为每个元素被读3次 (但可能被缓存)
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // 测试共享内存版本
    printf("\n--- 共享内存版本 ---\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        smoothShared<<<gridSize, BLOCK_SIZE>>>(d_X, d_Y, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 1000; i++) {
            smoothShared<<<gridSize, BLOCK_SIZE>>>(d_X, d_Y, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  耗时: %.4f ms (1000次平均)\n", ms / 1000);
        printf("  有效带宽: %.2f GB/s\n", (1.0f * size) / (ms / 1000 * 1e6));
        // 1.0f 因为每个元素从全局内存只读1次
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // 验证结果
    cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 1; i < n - 1; i++) {
        float expected = (h_X[i-1] + h_X[i] + h_X[i+1]) / 3.0f;
        if (fabs(h_Y[i] - expected) > 1e-4f) { ok = false; break; }
    }
    printf("\n  验证: %s\n", ok ? "✓ 通过" : "✗ 失败");

    // 打印共享内存信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n--- 设备共享内存信息 ---\n");
    printf("  每 block 共享内存: %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("  每 SM 共享内存:    %.1f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
    printf("  共享内存 bank 数:  %d\n", 32);  // 固定为 32

    // 清理
    cudaFree(d_X); cudaFree(d_Y);
    free(h_X); free(h_Y);

    printf("\n✓ Day 08 完成! 共享内存是 GPU 编程的关键利器。\n");
    return 0;
}
