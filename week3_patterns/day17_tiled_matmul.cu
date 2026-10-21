/*
 * Day 17: 分块矩阵乘法进阶 — 1D tile 和寄存器优化 (修复版)
 * ==================================================
 * 学习目标:
 *   1. 在 Day14 基础上深入优化矩阵乘法
 *   2. 使用寄存器 tile (每个线程计算多个输出元素)
 *   3. 理解计算强度 (arithmetic intensity) 的概念
 *   4. 对比不同 tile 大小和策略的性能
 *   5.cuda13.0编译
 * 编译: nvcc day17_tiled_matmul.cu -o day17 -lm
 * 运行: ./day17
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 32

// ============================================================
// V1: 基础分块
// ============================================================
__global__ void matMulTiled(const float * __restrict__ A,
                            const float * __restrict__ B,
                            float * __restrict__ C,
                            int M, int N, int K){
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++){
        s_A[threadIdx.y][threadIdx.x] = 
            (row < M && t * TILE_SIZE + threadIdx.x < K)
            ? A[row * K + t * TILE_SIZE + threadIdx.x] : 0.0f;
        s_B[threadIdx.y][threadIdx.x] = 
            (t * TILE_SIZE + threadIdx.y < K && col < N)
            ? B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++)
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        __syncthreads();
        
    }

    if (row < M && col < N) C[row * M + col] = sum;
}

// ============================================================
// V2: 寄存器 tile — 每个线程计算 RM x RN 个输出
// 关键修复: 加载阶段每个线程填 RM*RN=4 个元素, 覆盖完整 32x32 tile
// ============================================================
#define RM 2  // 每线程处理 2 行
#define RN 2  // 每线程处理 2 列
#define BM (TILE_SIZE / RM)  // 线程行数 = 16
#define BN (TILE_SIZE / RN)  // 线程列数 = 16

static_assert((TILE_SIZE % RM == 0) && (TILE_SIZE % RN == 0),
              "TILE_SIZE must be divisible by RM and RN");

__global__ void matMulRegTile(const float * __restrict__ A,
                              const float * __restrict__ B,
                              float * __restrict__ C,
                              int M, int N, int K)
{
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;  // 0..BN-1 (0..15)
    int ty = threadIdx.y;  // 0..BM-1 (0..15)

    int rowBase = blockIdx.y * TILE_SIZE + ty * RM;
    int colBase = blockIdx.x * TILE_SIZE + tx * RN;

    // 4个累加器
    float sum00 = 0.f, sum01 = 0.f, sum10 = 0.f, sum11 = 0.f;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++){
        // --- 加载A tile: 每个线程填2x2个元素, 256*4 = 1024 = 32x32 ---
        #pragma unroll
        for (int r = 0; r < RM; r++){
            #pragma unroll
            for (int c = 0; c < RN; c++) {
                int lr = ty * RM + r;
                int lc = tx * RN + c;
                int gRow = blockIdx.y * TILE_SIZE + lr;
                int gCol = t * TILE_SIZE + lc;
                s_A[lr][lc] = (gRow < M && gCol < K)
                              ? A[gRow * K + gCol] : 0.0f;
        }
    }
    // --- 加载 B tile:对称, 同样2x2 ---
    #pragma unroll
    for (int r = 0; r < RM; r++) {
        #pragma unroll
        for (int c = 0; c < RN; c++) {
            int lr = ty * RM + r;
            int lc = tx * RN + c;
            int gRow = t * TILE_SIZE + lr;
            int gCol = blockIdx.x * TILE_SIZE + lc;
            s_B[lr][lc] = (gRow < K && gCol < N)
                            ? B[gRow * N + gCol] : 0.0f;
        }
    } 
    __syncthreads();
    
    // --- 计算2x2输出 ---
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++){
        float a0 = s_A[ty * RM + 0][i];
        float a1 = s_A[ty * RM + 1][i];
        float b0 = s_B[i][tx * RN + 0];
        float b1 = s_B[i][tx * RN + 1];
        sum00 += a0 * b0;
        sum01 += a0 * b1;
        sum10 += a1 * b0;
        sum11 += a1 * b1;
    }
    __syncthreads();
    }

    // ---写回---
    int r0 = rowBase, r1 = rowBase + 1;
    int c0 = colBase, c1 = colBase + 1;
    if (r0 < M && c0 < N) C[r0 * N + c0] = sum00;
    if (r0 < M && c1 < N) C[r0 * N + c1] = sum01;
    if (r1 < M && c0 < N) C[r1 * N + c0] = sum10;
    if (r1 < M && c1 < N) C[r1 * N + c1] = sum11;
}

// ============================================================
// CPU 参考实现 (用于小尺寸验证)
// ============================================================
void matMulCPU(const float *A, const float *B, float *C,
               int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// ============================================================
// 正确性验证: 小尺寸, 与 CPU 参考逐元素比较
// ============================================================
template <typename Launcher>
bool verifySmall(const char *name, Launcher launch,
                 int M, int N, int K, float tol) {
    size_t szA = (size_t)M * K * sizeof(float);
    size_t szB = (size_t)K * N * sizeof(float);
    size_t szC = (size_t)M * N * sizeof(float);

    float *h_A = (float *)malloc(szA);
    float *h_B = (float *)malloc(szB);
    float *h_C = (float *)malloc(szC);
    float *h_ref = (float *)malloc(szC);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    matMulCPU(h_A, h_B, h_ref, M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, szA);
    cudaMalloc(&d_B, szB);
    cudaMalloc(&d_C, szC);
    cudaMemcpy(d_A, h_A, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, szB, cudaMemcpyHostToDevice);

    launch(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, szC, cudaMemcpyDeviceToHost);

    double maxErr = 0.0;
    for (int i = 0; i < M * N; i++) {
        double e = fabs((double)h_C[i] - (double)h_ref[i]);
        if (e > maxErr) maxErr = e;
    }
    bool ok = (maxErr < tol);
    printf("  %-12s maxErr = %.3e  ->  %s\n", name, maxErr,
           ok ? "✓ 通过" : "✗ 失败");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return ok;
}

int main() {
    printf("============================================\n");
    printf("   Day 17: 分块矩阵乘法进阶 (修复版)\n");
    printf("============================================\n\n");

    // ========================================================
    // 阶段 1: 小尺寸正确性验证 (128 x 128 x 128)
    // ========================================================
    printf("=== 阶段 1: 正确性验证 (128^3) ===\n");
    {
        int M = 128, N = 128, K = 128;
        dim3 blockV1(TILE_SIZE, TILE_SIZE);
        dim3 gridV1((N + TILE_SIZE - 1) / TILE_SIZE,
                    (M + TILE_SIZE - 1) / TILE_SIZE);

        dim3 blockV2(BN, BM);  // (16,16) threads, covers 32x32 elements
        dim3 gridV2 = gridV1;

        bool ok1 = verifySmall("V1 Tiled",
            [&](const float *a, const float *b, float *c, int m, int n, int k) {
                matMulTiled<<<gridV1, blockV1>>>(a, b, c, m, n, k);
            }, M, N, K, 1e-3f);

        bool ok2 = verifySmall("V2 RegTile",
            [&](const float *a, const float *b, float *c, int m, int n, int k) {
                matMulRegTile<<<gridV2, blockV2>>>(a, b, c, m, n, k);
            }, M, N, K, 1e-3f);

        if (!ok1 || !ok2) {
            fprintf(stderr, "\n正确性验证失败, 中止性能测试。\n");
            return 1;
        }
        printf("\n");
    }

    // ========================================================
    // 阶段 2: 大尺寸性能测试 (2048^3)
    // ========================================================
    printf("=== 阶段 2: 性能测试 (2048^3) ===\n");
    int M = 2048, N = 2048, K = 2048;
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);
    printf("矩阵: %dx%d * %dx%d = %dx%d\n\n", M, K, K, N, M, N);

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);

    srand(123);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITER = 10;
    double flops = 2.0 * M * N * K;

    // ---- V1 ----
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);

        matMulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < ITER; i++)
            matMulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        double avgMs = ms / ITER;
        double gflops = flops / (avgMs * 1e6);
        printf("--- V1: 基础分块 (tile=%d) ---\n", TILE_SIZE);
        printf("  耗时: %.3f ms\n", avgMs);
        printf("  性能: %.2f GFLOPS\n\n", gflops);
    }

    // ---- V2 ----
    {
        dim3 block(BN, BM);  // (16,16) threads, covers 32x32 elements
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);

        matMulRegTile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < ITER; i++)
            matMulRegTile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        double avgMs = ms / ITER;
        double gflops = flops / (avgMs * 1e6);
        printf("--- V2: 寄存器 tile (%dx%d per thread) ---\n", RM, RN);
        printf("  耗时: %.3f ms\n", avgMs);
        printf("  性能: %.2f GFLOPS\n\n", gflops);
    }

    // ---- 优化分析 (注释修正) ----
    printf("--- 优化分析 ---\n");
    printf("  朴素: 每元素 K 次全局读, AI = 2K / (2K*4B) = 0.25 FLOP/B\n");
    printf("  Tiled(%d): 每 tile 计算强度 = TILE/4 = %d FLOP/B\n",
           TILE_SIZE, TILE_SIZE / 4);
    printf("  RegTile(%dx%d): 不改变 global 访存总量 (AI 不变),\n",
           RM, RN);
    printf("      提升的是 shared memory 复用率和 ILP,\n");
    printf("      每线程做 %d 个输出、%d 次 FMA / shared 读\n",
           RM * RN, RM * RN);
    printf("  理论上限取决于 GPU 峰值 FLOPS 与内存带宽\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A); free(h_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n✓ Day 17 完成!\n");
    return 0;
}