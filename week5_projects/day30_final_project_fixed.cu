/*
 * Day 30: 终极项目 — 并行 N-Body 粒子模拟（修正版）
 * ============================================
 *
 * 主要修正:
 *   1. 修正注释导致的编译错误, 补充必要头文件
 *   2. 每次 GPU 测试都从相同的初始位置和零速度开始
 *   3. 正确性验证只运行 1 步, 与 CPU 单步结果对齐
 *   4. CPU 同样执行速度更新和位置更新
 *   5. Tiled Kernel 正确处理 n 不是 BLOCK_SIZE 整数倍的情况
 *   6. CPU 使用 std::chrono 计时, GPU 使用 CUDA Event 计时
 *   7. 增加 CUDA API 和 Kernel 启动错误检查
 *   8. 删除未使用的 forces 参数
 *
 * 编译: nvcc -O3 day30_final_project_fixed.cu -o day30
 * 运行: ./day30
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define EPS 0.001f

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(error));           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ============================================================
// Kernel 1: 朴素 N-Body 力计算
// 每个线程计算一个粒子受到的所有力, 并更新该粒子的速度
// ============================================================
__global__ void nbodyForceNaive(const float *pos,
                                float *vel,
                                int n,
                                float dt,
                                float softeningSq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    // 将当前粒子的位置保存在寄存器中
    float xi = pos[i * 3];
    float yi = pos[i * 3 + 1];
    float zi = pos[i * 3 + 2];

    for (int j = 0; j < n; ++j) {
        if (j == i) continue;

        float dx = pos[j * 3] - xi;
        float dy = pos[j * 3 + 1] - yi;
        float dz = pos[j * 3 + 2] - zi;

        float distSq = dx * dx + dy * dy + dz * dz + softeningSq;
        float invDist = rsqrtf(distSq);
        float invDist3 = invDist * invDist * invDist;

        fx += dx * invDist3;
        fy += dy * invDist3;
        fz += dz * invDist3;
    }

    // 假设所有粒子质量和引力常数的乘积均为 1
    vel[i * 3]     += fx * dt;
    vel[i * 3 + 1] += fy * dt;
    vel[i * 3 + 2] += fz * dt;
}

// ============================================================
// Kernel 2: Tiled N-Body 力计算
// 每个 block 协作加载一个 tile 到共享内存, 减少全局内存读取
// ============================================================
__global__ void nbodyForceTiled(const float *pos,
                                float *vel,
                                int n,
                                float dt,
                                float softeningSq) {
    __shared__ float spos[BLOCK_SIZE * 3];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    float xi = 0.0f;
    float yi = 0.0f;
    float zi = 0.0f;

    if (i < n) {
        xi = pos[i * 3];
        yi = pos[i * 3 + 1];
        zi = pos[i * 3 + 2];
    }

    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int tileStart = t * BLOCK_SIZE;
        int tileSize = n - tileStart;
        if (tileSize > BLOCK_SIZE) tileSize = BLOCK_SIZE;

        // 最后一个 tile 可能不足 BLOCK_SIZE 个元素
        int j = tileStart + tid;
        if (j < n) {
            spos[tid * 3]     = pos[j * 3];
            spos[tid * 3 + 1] = pos[j * 3 + 1];
            spos[tid * 3 + 2] = pos[j * 3 + 2];
        }

        // 所有线程都必须到达同步点, 因此不能放在 if (i < n) 内
        __syncthreads();

        if (i < n) {
            // 只访问当前 tile 中实际存在的粒子
            for (int k = 0; k < tileSize; ++k) {
                j = tileStart + k;
                if (j == i) continue;

                float dx = spos[k * 3] - xi;
                float dy = spos[k * 3 + 1] - yi;
                float dz = spos[k * 3 + 2] - zi;

                float distSq = dx * dx + dy * dy + dz * dz + softeningSq;
                float invDist = rsqrtf(distSq);
                float invDist3 = invDist * invDist * invDist;

                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }

        // 下一轮换共享内存之前, 等待本 block 的所有线程完成计算
        __syncthreads();
    }

    if (i < n) {
        vel[i * 3]     += fx * dt;
        vel[i * 3 + 1] += fy * dt;
        vel[i * 3 + 2] += fz * dt;
    }
}

// ============================================================
// Kernel 3: 使用更新后的速度更新位置
// 与先更新速度再更新位置的 CPU 半隐式 Euler 积分保持一致
// ============================================================
__global__ void updatePositions(float *pos,
                                const float *vel,
                                int n,
                                float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pos[i * 3]     += vel[i * 3] * dt;
    pos[i * 3 + 1] += vel[i * 3 + 1] * dt;
    pos[i * 3 + 2] += vel[i * 3 + 2] * dt;
}

// CPU 单步力计算与速度更新
void nbodyCPU(const float *pos,
              float *vel,
              int n,
              float dt,
              float softeningSq) {
    for (int i = 0; i < n; ++i) {
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        float xi = pos[i * 3];
        float yi = pos[i * 3 + 1];
        float zi = pos[i * 3 + 2];

        for (int j = 0; j < n; ++j) {
            if (j == i) continue;

            float dx = pos[j * 3] - xi;
            float dy = pos[j * 3 + 1] - yi;
            float dz = pos[j * 3 + 2] - zi;

            float distSq = dx * dx + dy * dy + dz * dz + softeningSq;
            float invDist = 1.0f / sqrtf(distSq);
            float invDist3 = invDist * invDist * invDist;

            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }

        vel[i * 3]     += fx * dt;
        vel[i * 3 + 1] += fy * dt;
        vel[i * 3 + 2] += fz * dt;
    }
}

void updatePositionsCPU(float *pos, const float *vel, int n, float dt) {
    for (int i = 0; i < n * 3; ++i) {
        pos[i] += vel[i] * dt;
    }
}

// 每次运行前都恢复同一份初始状态
void resetDeviceState(const float *h_pos,
                      const float *h_vel_init,
                      float *d_pos,
                      float *d_vel,
                      size_t dataSize) {
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel_init, dataSize, cudaMemcpyHostToDevice));
}

void printVerification(const char *name,
                       const float *gpuPos,
                       const float *gpuVel,
                       const float *cpuPos,
                       const float *cpuVel,
                       int n) {
    float maxVelAbsErr = 0.0f;
    float maxVelRelErr = 0.0f;
    float maxPosAbsErr = 0.0f;

    for (int i = 0; i < n * 3; ++i) {
        float velErr = fabsf(gpuVel[i] - cpuVel[i]);
        float posErr = fabsf(gpuPos[i] - cpuPos[i]);
        float denominator = fmaxf(1.0f, fabsf(cpuVel[i]));

        maxVelAbsErr = fmaxf(maxVelAbsErr, velErr);
        maxVelRelErr = fmaxf(maxVelRelErr, velErr / denominator);
        maxPosAbsErr = fmaxf(maxPosAbsErr, posErr);
    }

    printf("  %s vs CPU (同为 1 步):\n", name);
    printf("    速度最大绝对误差: %e\n", maxVelAbsErr);
    printf("    速度最大相对误差: %e\n", maxVelRelErr);
    printf("    位置最大绝对误差: %e\n", maxPosAbsErr);
}

void printPerformance(const char *name, int n, int steps, float ms) {
    // 每个粒子跳过自身, 因此实际粒子对数量为 N * (N - 1)
    // 20 FLOP/粒子对只是便于横向比较的近似值, 不代表精确硬件指令数
    double interactions = (double)n * (double)(n - 1);
    double approximateFlopsPerStep = 20.0 * interactions;
    double gflops = approximateFlopsPerStep * steps / ((double)ms * 1.0e6);

    printf("--- %s 性能 ---\n", name);
    printf("  %d 步耗时: %.3f ms (%.3f ms/步)\n", steps, ms, ms / steps);
    printf("  每步近似计算量: %.2f GFLOP\n", approximateFlopsPerStep / 1.0e9);
    printf("  近似有效性能: %.2f GFLOPS\n\n", gflops);
}

int main() {
    printf("============================================\n");
    printf("   Day 30: 终极项目 — N-Body 粒子模拟\n");
    printf("============================================\n\n");

    const int n = 16384;
    const int steps = 10;
    const float dt = 0.01f;
    const float softeningSq = EPS * EPS;
    const int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t dataSize = (size_t)n * 3 * sizeof(float);

    printf("粒子数: %d\n", n);
    printf("性能测试步数: %d\n", steps);
    printf("时间步: %f\n\n", dt);

    // h_vel_init 始终保持为零, 不会被任何 GPU 输出覆盖
    float *h_pos = (float *)malloc(dataSize);
    float *h_vel_init = (float *)calloc((size_t)n * 3, sizeof(float));
    float *h_pos_cpu = (float *)malloc(dataSize);
    float *h_vel_cpu = (float *)malloc(dataSize);
    float *h_pos_gpu = (float *)malloc(dataSize);
    float *h_vel_gpu = (float *)malloc(dataSize);

    if (!h_pos || !h_vel_init || !h_pos_cpu || !h_vel_cpu ||
        !h_pos_gpu || !h_vel_gpu) {
        fprintf(stderr, "Host memory allocation failed\n");
        free(h_pos);
        free(h_vel_init);
        free(h_pos_cpu);
        free(h_vel_cpu);
        free(h_pos_gpu);
        free(h_vel_gpu);
        return EXIT_FAILURE;
    }

    srand(42);
    for (int i = 0; i < n; ++i) {
        h_pos[i * 3]     = (float)(rand() % 10000) / 100.0f - 50.0f;
        h_pos[i * 3 + 1] = (float)(rand() % 10000) / 100.0f - 50.0f;
        h_pos[i * 3 + 2] = (float)(rand() % 10000) / 100.0f - 50.0f;
    }

    // --------------------------------------------------------
    // CPU 正确性参考: 只运行 1 个完整步骤
    // --------------------------------------------------------
    memcpy(h_pos_cpu, h_pos, dataSize);
    memcpy(h_vel_cpu, h_vel_init, dataSize);

    printf("--- CPU 参考计算 (1 步) ---\n");
    auto cpuStart = std::chrono::high_resolution_clock::now();
    nbodyCPU(h_pos_cpu, h_vel_cpu, n, dt, softeningSq);
    updatePositionsCPU(h_pos_cpu, h_vel_cpu, n, dt);
    auto cpuStop = std::chrono::high_resolution_clock::now();

    double cpuMs = std::chrono::duration<double, std::milli>(
                       cpuStop - cpuStart).count();
    printf("  CPU 1 步耗时: %.3f ms\n", cpuMs);
    printf("  示例: vel[0] = (%.6f, %.6f, %.6f)\n\n",
           h_vel_cpu[0], h_vel_cpu[1], h_vel_cpu[2]);

    float *d_pos = NULL;
    float *d_vel = NULL;
    CUDA_CHECK(cudaMalloc(&d_pos, dataSize));
    CUDA_CHECK(cudaMalloc(&d_vel, dataSize));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------------------------------------------
    // 正确性验证: 两种 GPU 实现都独立从初始状态运行 1 步
    // --------------------------------------------------------
    printf("--- 正确性验证 ---\n");

    resetDeviceState(h_pos, h_vel_init, d_pos, d_vel, dataSize);
    nbodyForceNaive<<<gridSize, BLOCK_SIZE>>>(
        d_pos, d_vel, n, dt, softeningSq);
    CUDA_CHECK(cudaGetLastError());
    updatePositions<<<gridSize, BLOCK_SIZE>>>(d_pos, d_vel, n, dt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_pos_gpu, d_pos, dataSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_gpu, d_vel, dataSize, cudaMemcpyDeviceToHost));
    printVerification("Naive", h_pos_gpu, h_vel_gpu,
                      h_pos_cpu, h_vel_cpu, n);

    resetDeviceState(h_pos, h_vel_init, d_pos, d_vel, dataSize);
    nbodyForceTiled<<<gridSize, BLOCK_SIZE>>>(
        d_pos, d_vel, n, dt, softeningSq);
    CUDA_CHECK(cudaGetLastError());
    updatePositions<<<gridSize, BLOCK_SIZE>>>(d_pos, d_vel, n, dt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_pos_gpu, d_pos, dataSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel_gpu, d_vel, dataSize, cudaMemcpyDeviceToHost));
    printVerification("Tiled", h_pos_gpu, h_vel_gpu,
                      h_pos_cpu, h_vel_cpu, n);
    printf("\n");

    // --------------------------------------------------------
    // 性能测试: 与正确性验证分开, 每个版本都重新初始化
    // 前面的单步验证同时起到了 Kernel 预热作用
    // --------------------------------------------------------
    float ms = 0.0f;

    resetDeviceState(h_pos, h_vel_init, d_pos, d_vel, dataSize);
    CUDA_CHECK(cudaEventRecord(start));
    for (int step = 0; step < steps; ++step) {
        nbodyForceNaive<<<gridSize, BLOCK_SIZE>>>(
            d_pos, d_vel, n, dt, softeningSq);
        CUDA_CHECK(cudaGetLastError());
        updatePositions<<<gridSize, BLOCK_SIZE>>>(d_pos, d_vel, n, dt);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printPerformance("朴素 N-Body", n, steps, ms);

    resetDeviceState(h_pos, h_vel_init, d_pos, d_vel, dataSize);
    CUDA_CHECK(cudaEventRecord(start));
    for (int step = 0; step < steps; ++step) {
        nbodyForceTiled<<<gridSize, BLOCK_SIZE>>>(
            d_pos, d_vel, n, dt, softeningSq);
        CUDA_CHECK(cudaGetLastError());
        updatePositions<<<gridSize, BLOCK_SIZE>>>(d_pos, d_vel, n, dt);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printPerformance("Tiled N-Body", n, steps, ms);

    CUDA_CHECK(cudaMemcpy(h_vel_gpu, d_vel, dataSize, cudaMemcpyDeviceToHost));
    printf("Tiled 性能测试最终状态示例: vel[0] = (%.6f, %.6f, %.6f)\n\n",
           h_vel_gpu[0], h_vel_gpu[1], h_vel_gpu[2]);

    // --------------------------------------------------------
    // 30 天 CUDA 学习路线总结
    // --------------------------------------------------------
    printf("============================================\n");
    printf("   30 天 CUDA 学习路线 — 完成总结\n");
    printf("============================================\n\n");

    printf("Week 1 — CUDA 基础:\n");
    printf("  Day 01: Hello CUDA (向量加法)\n");
    printf("  Day 02: GPU 设备查询\n");
    printf("  Day 03: 向量运算与 Grid/Block 配置\n");
    printf("  Day 04: 错误处理\n");
    printf("  Day 05: GPU 计时\n");
    printf("  Day 06: 并行归约 (点积)\n");
    printf("  Day 07: 综合练习 (1D 卷积)\n\n");

    printf("Week 2 — 内存模型:\n");
    printf("  Day 08: 共享内存基础\n");
    printf("  Day 09: 矩阵转置\n");
    printf("  Day 10: 常量内存\n");
    printf("  Day 11: 纹理内存\n");
    printf("  Day 12: 全局内存合并访问\n");
    printf("  Day 13: 固定内存与异步传输\n");
    printf("  Day 14: 分块矩阵乘法\n\n");

    printf("Week 3 — 并行模式:\n");
    printf("  Day 15: 并行归约 (4 种策略)\n");
    printf("  Day 16: 前缀和 (Scan)\n");
    printf("  Day 17: 分块矩阵乘法进阶\n");
    printf("  Day 18: 直方图 (原子操作)\n");
    printf("  Day 19: 流压缩\n");
    printf("  Day 20: 双调排序\n");
    printf("  Day 21: K-Means 聚类\n\n");

    printf("Week 4 — 高级主题:\n");
    printf("  Day 22: Warp 级编程\n");
    printf("  Day 23: Warp Shuffle 进阶\n");
    printf("  Day 24: 动态并行\n");
    printf("  Day 25: 流与事件\n");
    printf("  Day 26: 多 GPU 编程\n");
    printf("  Day 27: CUDA 库 (cuBLAS)\n");
    printf("  Day 28: Thrust 库\n\n");

    printf("Week 5 — 实战项目:\n");
    printf("  Day 29: 2D 图像卷积\n");
    printf("  Day 30: N-Body 粒子模拟\n\n");

    printf("============================================\n");
    printf("  恭喜完成 30 天 CUDA 学习路线!\n");
    printf("============================================\n\n");

    printf("下一步建议:\n");
    printf("  -> 学习 CUTLASS (矩阵乘法模板库)\n");
    printf("  -> 深入 cuDNN (深度学习加速)\n");
    printf("  -> 研究 NCCL (多 GPU 通信)\n");
    printf("  -> 探索 Triton (Python GPU 编程)\n");
    printf("  -> 阅读 NVIDIA 开源项目源码\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));

    free(h_pos);
    free(h_vel_init);
    free(h_pos_cpu);
    free(h_vel_cpu);
    free(h_pos_gpu);
    free(h_vel_gpu);

    return 0;
}
