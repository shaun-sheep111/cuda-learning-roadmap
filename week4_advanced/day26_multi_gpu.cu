/*
 * Day 26: 多 GPU 编程 — 跨设备并行 (benchmark 改进版)
 * =====================================================
 * 相比原始版本新增:
 *   - warmup: 吃掉首次 context/模块加载/升频开销
 *   - 分阶段计时: H2D / kernel / D2H 分别计时 (单卡用 event, 多卡用 chrono+sync)
 *   - 重复 repeat 次取最好
 *   - 多卡两种调度模式对比:
 *       staged    : H2D→同步→kernel→同步→D2H (三阶段清晰, 每阶段 4 卡并行)
 *       pipelined : 三阶段连续提交, 只在最后同步 (允许跨 GPU 重叠)
 *   - 打印各 GPU 的 PCIe busID
 *
 * 编译: nvcc day26_multi_gpu.cu -o day26 -O2
 * 可选: nvcc day26_multi_gpu.cu -o day26 -O2 -use_fast_math
 * 运行: ./day26
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define BLOCK_SIZE 256

// 数据量: 1<<26 = 67,108,864 元素 = 256 MB (float)
// 如果 pinned memory 分配失败, 可改小到 1<<25
#define N_ELEMS (1 << 26)
#define REPEAT  3

struct Timings {
    double h2d;    // 主机→设备
    double kernel; // kernel 执行
    double d2h;    // 设备→主机
    double total;  // 总计
};

__global__ void vectorCompute(const float *in, float *out, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * alpha + sinf(in[idx]);
    }
}

// ---------------------------------------------------------------
// 单 GPU: 分阶段计时 (event 在单设备上使用是正确的)
// ---------------------------------------------------------------
Timings singleGPU(const float *h_in, float *h_out, int n, float alpha, int repeat) {
    Timings best = {1e30, 1e30, 1e30, 1e30};

    cudaSetDevice(0);
    size_t size = n * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);

    for (int r = 0; r < repeat; r++) {
        cudaEventRecord(e0);
        cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, 0);
        cudaEventRecord(e1);

        vectorCompute<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, alpha, n);
        cudaEventRecord(e2);

        cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, 0);
        cudaEventRecord(e3);
        cudaEventSynchronize(e3);

        float h2d, ker, d2h;
        cudaEventElapsedTime(&h2d, e0, e1);
        cudaEventElapsedTime(&ker, e1, e2);
        cudaEventElapsedTime(&d2h, e2, e3);
        double tot = (double)h2d + ker + d2h;

        if (tot < best.total) {
            best.h2d    = h2d;
            best.kernel = ker;
            best.d2h    = d2h;
            best.total  = tot;
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaEventDestroy(e2); cudaEventDestroy(e3);
    return best;
}

// ---------------------------------------------------------------
// 多 GPU: 分阶段 / 流水线 两种调度
//   staged=true  : 每阶段结束同步所有 stream, 三段计时清晰
//   staged=false : 三阶段连续提交, 仅末尾同步, 允许跨 GPU 重叠
// 返回的 h2d/kernel/d2h 仅在 staged=true 时有意义
// ---------------------------------------------------------------
Timings multiGPU(const float *h_in, float *h_out, int n, float alpha,
                 int numGPUs, bool staged, int repeat) {
    Timings best = {1e30, 1e30, 1e30, 1e30};
    if (numGPUs <= 0) return best;

    // 余数分给前 rem 块, 不丢尾部
    int base = n / numGPUs, rem = n % numGPUs;
    int *offsets = (int *)malloc(numGPUs * sizeof(int));
    int *counts  = (int *)malloc(numGPUs * sizeof(int));
    int off = 0;
    for (int g = 0; g < numGPUs; g++) {
        offsets[g] = off;
        counts[g]  = base + (g < rem ? 1 : 0);
        off += counts[g];
    }

    float **h_in_c  = (float **)malloc(numGPUs * sizeof(float *));
    float **h_out_c = (float **)malloc(numGPUs * sizeof(float *));
    float **d_in_c  = (float **)malloc(numGPUs * sizeof(float *));
    float **d_out_c = (float **)malloc(numGPUs * sizeof(float *));
    cudaStream_t *streams = (cudaStream_t *)malloc(numGPUs * sizeof(cudaStream_t));

    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        size_t cb = (size_t)counts[g] * sizeof(float);
        cudaMallocHost(&h_in_c[g], cb);
        cudaMallocHost(&h_out_c[g], cb);
        cudaMalloc(&d_in_c[g], cb);
        cudaMalloc(&d_out_c[g], cb);
        cudaStreamCreate(&streams[g]);
        memcpy(h_in_c[g], h_in + offsets[g], cb);
    }

    auto now = []{ return std::chrono::steady_clock::now(); };
    auto dt  = [](std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto syncAll = [&]() {
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }
    };

    for (int r = 0; r < repeat; r++) {
        auto T0 = now();

        // ---- 阶段 1: H2D ----
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            cudaMemcpyAsync(d_in_c[g], h_in_c[g],
                            (size_t)counts[g] * sizeof(float),
                            cudaMemcpyHostToDevice, streams[g]);
        }
        if (staged) syncAll();
        auto T1 = now();

        // ---- 阶段 2: kernel ----
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            int gs = (counts[g] + BLOCK_SIZE - 1) / BLOCK_SIZE;
            vectorCompute<<<gs, BLOCK_SIZE, 0, streams[g]>>>(
                d_in_c[g], d_out_c[g], alpha, counts[g]);
        }
        if (staged) syncAll();
        auto T2 = now();

        // ---- 阶段 3: D2H ----
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            cudaMemcpyAsync(h_out_c[g], d_out_c[g],
                            (size_t)counts[g] * sizeof(float),
                            cudaMemcpyDeviceToHost, streams[g]);
        }
        syncAll();
        auto T3 = now();

        double tot = dt(T0, T3);
        if (tot < best.total) {
            best.h2d    = dt(T0, T1);
            best.kernel = dt(T1, T2);
            best.d2h    = dt(T2, T3);
            best.total  = tot;
        }
    }

    // 合并结果
    for (int g = 0; g < numGPUs; g++) {
        memcpy(h_out + offsets[g], h_out_c[g],
               (size_t)counts[g] * sizeof(float));
    }

    // 清理
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaFreeHost(h_in_c[g]);
        cudaFreeHost(h_out_c[g]);
        cudaFree(d_in_c[g]);
        cudaFree(d_out_c[g]);
        cudaStreamDestroy(streams[g]);
    }
    cudaSetDevice(0);

    free(h_in_c); free(h_out_c); free(d_in_c); free(d_out_c);
    free(streams); free(offsets); free(counts);
    return best;
}

// ---------------------------------------------------------------
int main() {
    printf("============================================\n");
    printf("   Day 26: 多 GPU 编程 (benchmark 改进版)\n");
    printf("============================================\n\n");

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    printf("检测到 %d 个 GPU 设备\n\n", numGPUs);

    if (numGPUs == 0) {
        printf("  ✗ 未检测到可用的 CUDA 设备，退出。\n");
        return 1;
    }
    if (numGPUs < 2) {
        printf("  ⚠ 此系统只有 %d 个 GPU, 多 GPU 优势不明显\n\n", numGPUs);
    }

    for (int g = 0; g < numGPUs; g++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, g);
        printf("  GPU %d: %s (%.0f MB, %d SMs)  PCIe bus=0x%02x dev=0x%02x\n",
               g, prop.name,
               prop.totalGlobalMem / 1e6,
               prop.multiProcessorCount,
               prop.pciBusID, prop.pciDeviceID);
    }
    printf("\n");

    if (numGPUs >= 2) {
        int p01, p10;
        cudaDeviceCanAccessPeer(&p01, 0, 1);
        cudaDeviceCanAccessPeer(&p10, 1, 0);
        printf("P2P 访问: GPU0→GPU1=%s, GPU1→GPU0=%s\n\n",
               p01 ? "支持" : "不支持", p10 ? "支持" : "不支持");
    }

    int n = N_ELEMS;
    size_t size = (size_t)n * sizeof(float);
    float alpha = 3.14f;
    printf("数据量: %d 元素 (%.2f MB), repeat=%d\n\n",
           n, (float)size / 1e6, REPEAT);

    float *h_in, *h_out_single, *h_out_multi;
    cudaMallocHost(&h_in, size);
    cudaMallocHost(&h_out_single, size);
    cudaMallocHost(&h_out_multi, size);
    for (int i = 0; i < n; i++) h_in[i] = (float)i * 0.0001f;

    int useGPUs = (numGPUs < 4) ? numGPUs : 4;

    // ---------------- warmup ----------------
    printf("--- Warmup (排除首次 context / 模块 / 升频开销) ---\n");
    singleGPU(h_in, h_out_single, n, alpha, 1);
    multiGPU(h_in, h_out_multi, n, alpha, useGPUs, true, 1);
    printf("  warmup 完成\n\n");

    // ---------------- 单 GPU ----------------
    printf("--- 单 GPU ---\n");
    Timings t1 = singleGPU(h_in, h_out_single, n, alpha, REPEAT);
    printf("  H2D=%.3f ms  kernel=%.3f ms  D2H=%.3f ms  total=%.3f ms\n\n",
           t1.h2d, t1.kernel, t1.d2h, t1.total);

    // ---------------- 多 GPU: 分阶段 ----------------
    printf("--- 多 GPU (分阶段: H2D→sync→kernel→sync→D2H) ---\n");
    Timings tm_staged = multiGPU(h_in, h_out_multi, n, alpha, useGPUs, true, REPEAT);
    printf("  H2D=%.3f ms  kernel=%.3f ms  D2H=%.3f ms  total=%.3f ms\n\n",
           tm_staged.h2d, tm_staged.kernel, tm_staged.d2h, tm_staged.total);

    // ---------------- 多 GPU: 流水线 ----------------
    printf("--- 多 GPU (流水线: 三段连续提交, 仅末尾同步) ---\n");
    Timings tm_pipe = multiGPU(h_in, h_out_multi, n, alpha, useGPUs, false, REPEAT);
    printf("  total=%.3f ms  (注: 该模式下 h2d/kernel/d2h 只是提交时间, 无意义)\n\n",
           tm_pipe.total);

    // ---------------- 验证 ----------------
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (fabsf(h_out_single[i] - h_out_multi[i]) > 1e-4f) {
            ok = false;
            printf("  差异: single[%d]=%f, multi=%f\n",
                   i, h_out_single[i], h_out_multi[i]);
            break;
        }
    }
    printf("验证: %s\n\n", ok ? "✓ 通过" : "✗ 失败");

    // ---------------- 汇总 ----------------
    printf("================ 结果汇总 ================\n");
    printf("数据量: %.2f MB, %d GPU\n", (float)size / 1e6, useGPUs);
    printf("%-24s %10s %10s %10s %10s\n", "", "H2D(ms)", "kernel(ms)", "D2H(ms)", "total(ms)");
    printf("%-24s %10.3f %10.3f %10.3f %10.3f\n", "单 GPU",
           t1.h2d, t1.kernel, t1.d2h, t1.total);
    printf("%-24s %10.3f %10.3f %10.3f %10.3f\n", "多 GPU(分阶段)",
           tm_staged.h2d, tm_staged.kernel, tm_staged.d2h, tm_staged.total);
    printf("%-24s %10s %10s %10s %10.3f\n", "多 GPU(流水线)",
           "-", "-", "-", tm_pipe.total);
    printf("\n加速比:\n");
    printf("  kernel 阶段  (单卡 / 多卡分阶段) : %.2fx\n",
           tm_staged.kernel > 0 ? t1.kernel / tm_staged.kernel : 0.0);
    printf("  H2D 阶段     (单卡 / 多卡分阶段) : %.2fx\n",
           tm_staged.h2d > 0 ? t1.h2d / tm_staged.h2d : 0.0);
    printf("  D2H 阶段     (单卡 / 多卡分阶段) : %.2fx\n",
           tm_staged.d2h > 0 ? t1.d2h / tm_staged.d2h : 0.0);
    printf("  总时间       (单卡 / 多卡分阶段) : %.2fx\n",
           tm_staged.total > 0 ? t1.total / tm_staged.total : 0.0);
    printf("  总时间       (单卡 / 多卡流水线) : %.2fx\n",
           tm_pipe.total > 0 ? t1.total / tm_pipe.total : 0.0);

    printf("\n--- 多 GPU 最佳实践 ---\n");
    printf("  ✓ 每个 GPU 用独立的流和 pinned memory\n");
    printf("  ✓ 尽量减少 GPU 间通信\n");
    printf("  ✓ 考虑启用 P2P 访问 (cudaDeviceEnablePeerAccess)\n");
    printf("  ✓ 考虑使用统一内存简化编程 (cudaMallocManaged)\n");
    printf("  ✓ 负载均衡: 确保各 GPU 工作量相近\n");
    printf("  ✓ benchmark 必须先 warmup, 否则首次初始化开销会掩盖一切\n");
    printf("  ✓ 分阶段计时才能看出瓶颈在 PCIe 还是 kernel\n");

    cudaFreeHost(h_in);
    cudaFreeHost(h_out_single);
    cudaFreeHost(h_out_multi);

    printf("\n✓ Day 26 完成!\n");
    return 0;
}