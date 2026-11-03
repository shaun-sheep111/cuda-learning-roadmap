/*
 * Day 25: 流与事件 — 异步执行与计算重叠
 * ============================================
 * 学习目标:
 *   1. 理解 CUDA 流 (stream) 的概念和用途
 *   2. 掌握多流实现计算与传输重叠
 *   3. 使用 CUDA 事件进行精确计时和同步控制
 *   4. 理解默认流 vs 非默认流的行为差异
 *
 * 关键概念:
 *   - 同一流中的操作按顺序执行
 *   - 不同流中的操作可以并发执行
 *   - H2D 传输、Kernel 计算、D2H 传输可以重叠
 *   - 需要 pinned memory 才能真正异步传输
 *
 * 流的类型:
 *   - 默认流 (stream 0): 与所有非默认流同步
 *   - 非默认流: 之间可以真正并发
 *   - per-thread 默认流: 每线程独立 (编译选项 --default-stream per-thread)
 *   - nvcc13.0编译
 * 编译: nvcc day25_streams_events.cu -o day25
 * 运行: ./day25
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE   256
#define NUM_STREAMS  4

__global__ void vectorCompute(const float *in, float *out, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = in[idx];
    for (int i = 0; i < iters; i++) {
        val = sinf(val) + cosf(val * 0.5f);
        val = val * val;
    }
    out[idx] = val;
}



int main() {
    printf("============================================\n");
    printf("   Day 25: 流与事件 (Streams & Events)\n");
    printf("============================================\n\n");

    int n = 1 << 22;  // 4M
    size_t size = (size_t)n * sizeof(float);
    int iters = 200;

    printf("数据量: %d 元素 (%.2f MB)\n", n, (float)size / 1e6f);
    printf("计算迭代: %d 次/元素\n\n", iters);

    // ---------- 分配 pinned memory (异步传输前提) ----------
    float *h_in = nullptr, *h_out = nullptr;
    float *h_ref = nullptr;
    cudaMallocHost((void**)&h_in,  size);
    cudaMallocHost((void**)&h_out, size);
    cudaMallocHost((void**)&h_ref, size);
    for (int i = 0; i < n; i++) h_in[i] = (float)i * 0.0001f;

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  size);
    cudaMalloc(&d_out, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // =====================================================
    // Warmup: 预热 module load / JIT / 首次 launch
    // 否则方式 1 会吃掉 20-40 ms 冷启动开销，数字不可比
    // =====================================================
    {
        float *w_in = nullptr, *w_out = nullptr;
        float *dw_in = nullptr, *dw_out = nullptr;
        cudaMallocHost((void**)&w_in,  size);
        cudaMallocHost((void**)&w_out, size);
        cudaMalloc((void**)&dw_in,  size);
        cudaMalloc((void**)&dw_out, size);

        cudaMemcpy(dw_in, h_in, size, cudaMemcpyHostToDevice);
        vectorCompute<<<gridSize, BLOCK_SIZE>>>(dw_in, dw_out, n, iters);
        cudaMemcpy(h_ref, dw_out, size, cudaMemcpyDeviceToHost);  // 顺便当参考值
        cudaDeviceSynchronize();

        cudaFree(dw_in); cudaFree(dw_out);
        cudaFreeHost(w_in); cudaFreeHost(w_out);
        printf("warmup 完成 (参考结果已生成)\n\n");
    }

    // =====================================================
    // 方式 1: 同步（默认流，串行）
    // =====================================================
    printf("--- 方式 1: 同步 (串行) ---\n");
    {
        cudaEventRecord(start);
        cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
        vectorCompute<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n, iters);
        cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  总耗时: %.3f ms\n", ms);
        printf("  (H2D + Kernel + D2H 串行执行)\n\n");
    }

    // =====================================================
    // 方式 2: 单流异步
    // =====================================================
    printf("--- 方式 2: 单流异步 ---\n");
    {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        cudaEventRecord(start, stream);
        cudaMemcpyAsync(d_in, h_in, size,
                        cudaMemcpyHostToDevice, stream);
        vectorCompute<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_in, d_out, n, iters);
        cudaMemcpyAsync(h_out, d_out, size,
                        cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(stop, stream);

        // 演示“CPU 不被阻塞”：同步前先做点 CPU 活
        volatile long long acc = 0;
        for (long long i = 0; i < 20000000LL; ++i) acc += i;
        (void)acc;

        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  总耗时: %.3f ms\n", ms);
        printf("  (同一流中仍串行; CPU 可并行干别的, 由事件同步)\n\n");

        cudaStreamDestroy(stream);
    }

    // =====================================================
    // 方式 3: 多流流水线
    // =====================================================
    printf("--- 方式 3: 多流流水线 (%d 流) ---\n", NUM_STREAMS);
    {
        cudaStream_t streams[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }

        // 计时用独立的流，避免默认流隐式同步污染计时
        cudaStream_t timerStream;
        cudaStreamCreateWithFlags(&timerStream, cudaStreamNonBlocking);

        // 每个 chunk 的 offset / 长度（处理余数）
        int offsets[NUM_STREAMS], lens[NUM_STREAMS];
        int base = n / NUM_STREAMS;
        for (int i = 0; i < NUM_STREAMS; i++) {
            offsets[i] = i * base;
            lens[i]    = (i == NUM_STREAMS - 1) ? (n - offsets[i]) : base;
        }

        // 每个流的 device chunk
        float *d_in_chunks[NUM_STREAMS], *d_out_chunks[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaMalloc((void**)&d_in_chunks[i],
                       (size_t)lens[i] * sizeof(float));
            cudaMalloc((void**)&d_out_chunks[i],
                       (size_t)lens[i] * sizeof(float));
        }

        // 每流完成事件（仅同步用，关闭计时）
        cudaEvent_t done[NUM_STREAMS];
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming);
        }

        // --- 计时开始: timerStream record start，所有工作流等 start ---
        cudaEventRecord(start, timerStream);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamWaitEvent(streams[i], start, 0);
        }

        // --- 提交流水线 ---
        for (int i = 0; i < NUM_STREAMS; i++) {
            size_t chunkBytes = (size_t)lens[i] * sizeof(float);
            int    chunkGrid  = (lens[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;

            cudaMemcpyAsync(d_in_chunks[i],
                            h_in + offsets[i],   // 直接使用原 pinned 内存
                            chunkBytes,
                            cudaMemcpyHostToDevice, streams[i]);
            vectorCompute<<<chunkGrid, BLOCK_SIZE, 0, streams[i]>>>(
                d_in_chunks[i], d_out_chunks[i], lens[i], iters);
            cudaMemcpyAsync(h_out + offsets[i],  // 直接写回原 pinned 内存
                            d_out_chunks[i],
                            chunkBytes,
                            cudaMemcpyDeviceToHost, streams[i]);

            cudaEventRecord(done[i], streams[i]);
        }

        // --- timerStream 等所有流完成，再 record stop ---
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamWaitEvent(timerStream, done[i], 0);
        }
        cudaEventRecord(stop, timerStream);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  总耗时: %.3f ms\n", ms);
        printf("  (H2D/Kernel/D2H 在不同流间重叠)\n\n");

        // --- 清理 ---
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaEventDestroy(done[i]);
            cudaStreamDestroy(streams[i]);
            cudaFree(d_in_chunks[i]);
            cudaFree(d_out_chunks[i]);
        }
        cudaStreamDestroy(timerStream);
    }

    // =====================================================
    // 方式 4: 事件做跨流依赖
    // =====================================================
    printf("--- 方式 4: 事件同步 ---\n");
    {
        cudaStream_t s1, s2;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

        cudaEvent_t ev_transfer_done, ev_kernel_done;
        cudaEventCreateWithFlags(&ev_transfer_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev_kernel_done,   cudaEventDisableTiming);

        // start 记录在 s1，与 s1 的第一个 H2D 同流有序
        cudaEventRecord(start, s1);

        // s1: H2D
        cudaMemcpyAsync(d_in, h_in, size,
                        cudaMemcpyHostToDevice, s1);
        cudaEventRecord(ev_transfer_done, s1);

        // s2: 等 H2D 完成后计算
        cudaStreamWaitEvent(s2, ev_transfer_done, 0);
        vectorCompute<<<gridSize, BLOCK_SIZE, 0, s2>>>(d_in, d_out, n, iters);
        cudaEventRecord(ev_kernel_done, s2);

        // s1: 等 kernel 完成后 D2H，然后 record stop
        cudaStreamWaitEvent(s1, ev_kernel_done, 0);
        cudaMemcpyAsync(h_out, d_out, size,
                        cudaMemcpyDeviceToHost, s1);
        cudaEventRecord(stop, s1);
        cudaEventSynchronize(stop);
        // s1 的 stop 依赖 s2 的 kernel（经 ev_kernel_done），
        // 因此 stop 已覆盖全部工作，无需再同步 s2。

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  总耗时: %.3f ms (事件控制跨流依赖)\n\n", ms);

        cudaEventDestroy(ev_transfer_done);
        cudaEventDestroy(ev_kernel_done);
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
    }

    // =====================================================
    // 小结
    // =====================================================
    printf("--- 流使用建议 ---\n");
    printf("  [OK] 需要 pinned memory 才能真正异步传输\n");
    printf("  [OK] 多流 + 分块 = 流水线重叠\n");
    printf("  [OK] 用事件做跨流依赖控制\n");
    printf("  [OK] 流数量不宜过多 (通常 4-8 个足够)\n");
    printf("  [OK] 计时事件与工作同流记录, 或经 waitEvent 串起\n");
    printf("  [NO] 避免在默认流中混用阻塞/非阻塞流\n");
    printf("  [NO] 不要先 host sync 再 record stop, 计时会偏大\n");

    // ---------- 清理 ----------
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n✓ Day 25 完成! 多流异步是隐藏延迟的关键技术。\n");
    return 0;
}

