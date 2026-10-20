/*
 * Day 13: 固定内存 — 加速 Host-Device 数据传输
 * ================================================
 * 学习目标:
 *   1. 理解可分页内存 vs 固定内存 (pinned memory) 的区别
 *   2. 掌握 cudaMallocHost / cudaFreeHost 的使用
 *   3. 理解为什么固定内存能加速传输
 *   4. 学习异步传输 (cudaMemcpyAsync) 和固定内存的配合
 *
 * 关键概念:
 *   - 普通malloc分配的是可分页内存, OS可以将其换出到磁盘
 *   - GPU不能直接访问可分页内存, 需要先拷贝到暂存区 (staging buffer)
 *   - 固定内存不可换出, GPU可以直接DMA访问, 省去中间拷贝
 *   - 固定内存传输速度可达满带宽, 可分页内存通常只有一半
 *
 * 注意:
 *   - 固定内存不能分配太多 (会减少系统可用内存)
 *   - 过度使用固定内存会降低整体系统性能
 *   - 适合频繁传输的场景
 *   - cuda13.0编译
 * 编译: nvcc day13_pinned_memory.cu -o day13
 * 运行: ./day13
 */

#include <stdio.h>

#define BLOCK_SIZE 256
#define ITERS      100          // 统一循环次数(原代码 H2D 用 100, D2H 用 10, 公式却乘 10)

__global__ void vectorScale(const float *in, float *out, float alpha, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * alpha;
}

int main()
{
    printf("============================================\n");
    printf("   Day 13: 固定内存 (Pinned Memory) 修正版\n");
    printf("============================================\n\n");

    int sizes[] = {1 << 20, 1 << 22, 1 << 24, 1 << 26};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("%-10s  %-15s  %-15s  %-10s\n", "大小(MB)", "可分页(GB/s)", "固定(GB/s)", "加速比");
    printf("-----------------------------------------------------------\n");

    for (int s = 0; s < numSizes; s++){
        int n = sizes[s];
        size_t size = n * sizeof(float);
        float sizeMB = (float)size / 1e6;

        // ---------- pageable ----------
        float *h_pageable = (float *)malloc(size);
        float *d_data;
        cudaMalloc(&d_data, size);

        cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);  // 预热
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++)
            cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float h2dPageableMs; cudaEventElapsedTime(&h2dPageableMs, start, stop);

        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++)                                // 与 H2D 对齐
            cudaMemcpy(h_pageable, d_data, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float d2hPageableMs; cudaEventElapsedTime(&d2hPageableMs, start, stop);

        // ---------- pinned ----------
        float *h_pinned;
        cudaMallocHost(&h_pinned, size);

        cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);    // 预热
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++)
            cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float h2dPinnedMs; cudaEventElapsedTime(&h2dPinnedMs, start, stop);

        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++)
            cudaMemcpy(h_pinned, d_data, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float d2hPinnedMs; cudaEventElapsedTime(&d2hPinnedMs, start, stop);

        // 带宽 = 总字节 / 总时间; H2D ITERS 次 + D2H ITERS 次
        float pageableBW = (2.0f * size * ITERS) / ((h2dPageableMs + d2hPageableMs) * 1e6);
        float pinnedBW   = (2.0f * size * ITERS) / ((h2dPinnedMs   + d2hPinnedMs)   * 1e6);

        printf("%-10.1f  %-15.2f  %-15.2f  %-10.1fx\n",
               sizeMB, pageableBW, pinnedBW, pinnedBW / pageableBW);

        free(h_pageable);
        cudaFreeHost(h_pinned);
        cudaFree(d_data);
    }

    // ========================================================
    // 异步传输 + 计算重叠 (修正版)
    // ========================================================
    printf("\n--- 异步传输 + 计算重叠 (修正版) ---\n\n");

    int n = 1 << 24;                    // 16M
    size_t size = n * sizeof(float);

    float *h_in, *h_out;
    cudaMallocHost(&h_in, size);
    cudaMallocHost(&h_out, size);
    for (int i = 0; i < n; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    int gridFull = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ---------- A: 同步 ----------
    cudaEventRecord(start);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    vectorScale<<<gridFull, BLOCK_SIZE>>>(d_in, d_out, 2.0f, n);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float syncMs; cudaEventElapsedTime(&syncMs, start, stop);

    bool ok = true;
    for (int i = 0; i < n; i += 1024)
        if (h_out[i] != h_in[i] * 2.0f) { ok = false; break; }
    printf("  同步结果验证: %s\n", ok ? "OK" : "FAIL");

    // ---------- B: 单 stream 异步 (退化对照组) ----------
    cudaStream_t s1;
    cudaStreamCreate(&s1);

    cudaEventRecord(start);
    cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, s1);
    vectorScale<<<gridFull, BLOCK_SIZE, 0, s1>>>(d_in, d_out, 2.0f, n);
    cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, s1);
    cudaStreamSynchronize(s1);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float singleMs; cudaEventElapsedTime(&singleMs, start, stop);

    // ---------- C: 双 stream pipeline (真正重叠) ----------
    const int NUM_CHUNKS = 8;
    int chunk = n / NUM_CHUNKS;                       // 2^24 / 8 = 2^21, 是 BLOCK_SIZE 整数倍
    size_t chunkBytes = (size_t)chunk * sizeof(float);
    int gridChunk = (chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaStream_t s[2];
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_CHUNKS; i++) {
        cudaStream_t st = s[i & 1];
        size_t off = (size_t)i * chunk;

        // 同一 stream 内三步自动有序: 传这块 -> 算这块 -> 回传这块
        cudaMemcpyAsync(d_in  + off, h_in  + off, chunkBytes,
                        cudaMemcpyHostToDevice, st);
        vectorScale<<<gridChunk, BLOCK_SIZE, 0, st>>>(
            d_in + off, d_out + off, 2.0f, chunk);
        cudaMemcpyAsync(h_out + off, d_out + off, chunkBytes,
                        cudaMemcpyDeviceToHost, st);
    }
    cudaStreamSynchronize(s[0]);
    cudaStreamSynchronize(s[1]);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float pipelineMs; cudaEventElapsedTime(&pipelineMs, start, stop);

    ok = true;
    for (int i = 0; i < n; i += 1024)
        if (h_out[i] != h_in[i] * 2.0f) { ok = false; break; }
    printf("  Pipeline 结果验证: %s\n", ok ? "OK" : "FAIL");

    printf("\n  同步 memcpy 串行:         %.3f ms\n", syncMs);
    printf("  单 stream 异步 (退化):     %.3f ms\n", singleMs);
    printf("  双 stream pipeline (重叠): %.3f ms\n", pipelineMs);
    printf("  pipeline / 同步:  %.2fx\n", syncMs / pipelineMs);
    printf("  pipeline / 单stream: %.2fx\n", singleMs / pipelineMs);

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s[0]);
    cudaStreamDestroy(s[1]);
    cudaFreeHost(h_in); cudaFreeHost(h_out);
    cudaFree(d_in); cudaFree(d_out);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    printf("\n✓ Day 13 完成!\n");
    return 0;
}