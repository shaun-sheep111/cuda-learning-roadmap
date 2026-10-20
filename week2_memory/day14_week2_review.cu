/*
 * Day 14: Week 2 综合 — 分块矩阵乘法
 * =====================================
 * 本日综合运用 Week 2 所学内存知识:
 *   - 共享内存 (tiled 矩阵乘法)
 *   - 全局内存合并访问
 *   - 内存层次优化
 *   - Kernel 1: 朴素版 (对照)
 *   - Kernel 2: 基础 tiled 32×32 (对照)
 *   - Kernel 3: 8×8 register tiling  (主优化)
 *
 * 矩阵乘法 C = A * B (A: MxK, B: KxN, C: MxN)
 * 朴素版本: 每个元素需要 K 次全局内存读取
 * 分块版本: 使用共享内存 tile, 大幅减少全局内存访问
 *
 * 这是 CUDA 编程最重要的模式之一!
 *
 * 编译: nvcc day14_week2_review.cu -o day14
 * 运行: ./day14
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 32

// ============ Kernel 1: 朴素 ============
__global__ void matMulNaive(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) sum += A[row*K+i] * B[i*N+col];
        C[row*N+col] = sum;
    }
}

// ============ Kernel 2: 基础 tiled 32×32 ============
__global__ void matMulTiled(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        s_A[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row*K+aCol] : 0.0f;
        s_B[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow*N+col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++)
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = sum;
}

// ============ Kernel 3: 8×8 register tiling (核心优化) ============
#define BM 128    // 每 block 计算 C 的行数
#define BN 128    // 每 block 计算 C 的列数
#define BK 8      // K 方向每次 tile 大小

__global__ void matMulTiled8x8(const float *A, const float *B, float *C,
                               int M, int N, int K) {
    __shared__ float As[BM][BK];   // [128][8] = 4 KB
    __shared__ float Bs[BK][BN];   // [8][128]  = 4 KB

    int tx  = threadIdx.x;          // 0..15
    int ty  = threadIdx.y;          // 0..15
    int tid = ty * 16 + tx;         // 0..255

    int bm = blockIdx.y * BM;
    int bn = blockIdx.x * BN;

    float acc[8][8] = {0.0f};       // 每线程 64 个累加器

    int numTiles = (K + BK - 1) / BK;
    for (int t = 0; t < numTiles; t++) {
        int bk = t * BK;

        // --- 加载 A tile: 128×8 = 1024 float, 256 线程 × 4 ---
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int idx = tid * 4 + j;
            int r   = idx / BK;
            int c   = idx % BK;
            As[r][c] = (bm + r < M && bk + c < K)
                       ? A[(bm + r) * K + bk + c] : 0.0f;
        }
        // --- 加载 B tile: 8×128 = 1024 float, 256 线程 × 4 ---
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int idx = tid * 4 + j;
            int r   = idx / BN;
            int c   = idx % BN;
            Bs[r][c] = (bk + r < K && bn + c < N)
                       ? B[(bk + r) * N + bn + c] : 0.0f;
        }
        __syncthreads();

        // --- 核心: 每次迭代 8+8 次共享内存读 + 64 次 FMA ---
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float a[8], b[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) a[i] = As[ty * 8 + i][kk];
            #pragma unroll
            for (int j = 0; j < 8; j++) b[j] = Bs[kk][tx * 8 + j];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++)
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    // --- 写回 8×8 ---
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int r = bm + ty * 8 + i;
            int c = bn + tx * 8 + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

// ============ CPU 参考 ============
void matMulCPU(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[i*K+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}

// ============ 单组尺寸测试 ============
void testOneCase(int M, int N, int K, const char *label) {
    printf("=== %s: A[%d×%d] * B[%d×%d] = C[%d×%d] ===\n",
           label, M, K, K, N, M, N);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *h_A   = (float*)malloc(sizeA);
    float *h_B   = (float*)malloc(sizeB);
    float *h_C   = (float*)malloc(sizeC);
    float *h_ref = (float*)malloc(sizeC);

    for (int i = 0; i < M*K; i++) h_A[i] = (float)(rand()%100)/100.0f;
    for (int i = 0; i < K*N; i++) h_B[i] = (float)(rand()%100)/100.0f;

    matMulCPU(h_A, h_B, h_ref, M, N, K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA); cudaMalloc(&d_B, sizeB); cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // 朴素
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
        matMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) matMulNaive<<<grid, block>>>(d_A,d_B,d_C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
        float maxErr = 0;
        for (int i = 0; i < M*N; i++) maxErr = fmaxf(maxErr, fabsf(h_C[i]-h_ref[i]));
        printf("  朴素:          %.4f ms  %8.1f GFLOPS  err=%.2e\n",
               ms/10, 2.0*M*N*K/(ms/10*1e6), maxErr);
    }
    // 基础 tiled
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
        matMulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) matMulTiled<<<grid, block>>>(d_A,d_B,d_C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
        float maxErr = 0;
        for (int i = 0; i < M*N; i++) maxErr = fmaxf(maxErr, fabsf(h_C[i]-h_ref[i]));
        printf("  基础 tiled:    %.4f ms  %8.1f GFLOPS  err=%.2e\n",
               ms/10, 2.0*M*N*K/(ms/10*1e6), maxErr);
    }
    // 8×8 register tiling
    {
        dim3 block(16, 16);
        dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
        matMulTiled8x8<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        for (int i = 0; i < 10; i++) matMulTiled8x8<<<grid, block>>>(d_A,d_B,d_C,M,N,K);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
        float maxErr = 0;
        for (int i = 0; i < M*N; i++) maxErr = fmaxf(maxErr, fabsf(h_C[i]-h_ref[i]));
        printf("  8×8 register:  %.4f ms  %8.1f GFLOPS  err=%.2e\n",
               ms/10, 2.0*M*N*K/(ms/10*1e6), maxErr);
    }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    printf("\n");
}

int main() {
    printf("============================================\n");
    printf("   Day 14: Week2 综合 — 分块矩阵乘法 (优化版)\n");
    printf("============================================\n\n");

    int cases[][3] = {
        {1024, 1024, 1024},   // 整数倍基准
        {1000, 1000, 1000},   // 非整数倍: 验证边界
        {1000, 2000,  500},   // 三边都不等
        {  33,   65,  17},    // 极小: 跨 tile 边界
        {1024, 1024, 1023},   // K 少 1: tile 尾部半满
    };
    int nCases = sizeof(cases) / sizeof(cases[0]);
    for (int i = 0; i < nCases; i++)
        testOneCase(cases[i][0], cases[i][1], cases[i][2], i==0 ? "基准" : "边界");

    printf("✓ Day 14 完成!\n");
    return 0;
}