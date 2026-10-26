/*
 * Day 21: Week 3 综合 — 并行 K-Means 聚类 (优化版)
 * ===============================================
 * 优化点:
 *   1. 所有 CUDA 调用加错误检查
 *   2. assignPoints: centroids 加载到 shared memory
 *   3. computeCentroids: block 内 shared memory 聚合, 大幅减少 global 原子操作
 *   4. computeInertia: 用 double 归约, 消除精度损失
 *   5. 收敛判断: 第一次迭代特殊处理, 不再显示虚假的 100% 改善
 *   6. 初始中心随机选取, 避免"初始即收敛"
 *   7. partial buffer 复用, 加 kernel 计时
 *   8. nvc13.0编译
 * 编译: nvcc day21_week3_review_opt.cu -o day21_opt
 * 运行: ./day21_opt
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_K 64
#define DIM 2
#define MAX_ITERS 100

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_LAUNCH() do { \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "Kernel launch error: %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// Kernel 1: 分配每个点到最近的聚类中心
//   优化: 把 centroids 加载到 shared memory (K 大时收益明显)
// ============================================================
__global__ void assignPoints(const float *points, const float *centroids,
                              int *assignments, float *minDist, int N, int K) {
    __shared__ float sCentroids[MAX_K * DIM];
    int tid = threadIdx.x;

    for (int i = tid; i < K * DIM; i += blockDim.x) {
        sCentroids[i] = centroids[i];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx >= N) return ;

    float minD = 1e30f;
    int bestK = 0;

    // 预读点坐标到寄存器, 避免重复访问全局内存
    float pt[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) pt[d] = points[idx * DIM + d];

    for (int k = 0; k < K; k++){
        float dist = 0.0f;
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            float diff = pt[d] - sCentroids[k * DIM + d];
            dist += diff * diff;
        }
        if (dist < minD) { minD = dist; bestK = k;}
    }
    assignments[idx] = bestK;
    minDist[idx] = minD;
}

// ============================================================
// Kernel 2: 计算每簇坐标和 + 点数
//   优化: block 内先聚合到 shared memory, block 末统一原子累加到 global
//   global 原子操作次数: N*(DIM+1) -> gridSize*(K*DIM + K)
// ============================================================
__global__ void computeCentroids(const float *points, const int *assignments,
                                  float *centroidSums, int *clusterCounts,
                                  int N, int K) {
    __shared__ float sSums[MAX_K * DIM];
    __shared__ int   sCounts[MAX_K];
    int tid = threadIdx.x;

    for (int i = tid; i < K * DIM; i += blockDim.x) sSums[i]   = 0.0f;
    for (int i = tid; i < K;     i += blockDim.x) sCounts[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N){
        int k = assignments[idx];
        float pt[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; d++) pt[d] = points[idx * DIM + d];
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            atomicAdd(&sSums[k * DIM + d], pt[d]);  // shared memory 原子操作
        }
        atomicAdd(&sCounts[k], 1);
    }

    __syncthreads();

    // block 级汇总后,只做少量global原子操作
    for (int i = tid; i < K * DIM; i += blockDim.x) {
            if (sSums[i] != 0.0f) atomicAdd(&centroidSums[i], sSums[i]);
        }
        for (int i = tid; i < K; i += blockDim.x) {
            if (sCounts[i] != 0) atomicAdd(&clusterCounts[i], sCounts[i]);
        }
}

// ============================================================
// Kernel 3: 更新聚类中心
// ============================================================
__global__ void updateCentroids(float *centroids, const float *centroidSums,
                                 const int *clusterCounts, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    if (clusterCounts[k] > 0) {
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            centroids[k * DIM + d] = centroidSums[k * DIM + d] / clusterCounts[k];
        }
    }
}

// ============================================================
// Kernel 4: 总惯性 (double 归约, 精度好)
// ============================================================
__global__ void computeInertia(const float *minDist, double *partialSums, int N) {
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < N) ? (double)minDist[idx] : 0.0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) partialSums[blockIdx.x] = sdata[0];
}

// ============================================================
// 数据生成
// ============================================================
void generateData(float *points, int N, int K) {
    float centers[MAX_K * DIM];
    for (int k = 0; k < K; k++) {
        centers[k * DIM]     = (float)(rand() % 1000) / 10.0f;
        centers[k * DIM + 1] = (float)(rand() % 1000) / 10.0f;
    }
    for (int i = 0; i < N; i++) {
        int k = i % K;
        float cx = centers[k * DIM];
        float cy = centers[k * DIM + 1];
        float noise = 0.0f;
        for (int j = 0; j < 4; j++) noise += (rand() / (float)RAND_MAX - 0.5f);
        points[i * DIM] = cx + noise * 10.0f;
        noise = 0.0f;
        for (int j = 0; j < 4; j++) noise += (rand() / (float)RAND_MAX - 0.5f);
        points[i * DIM + 1] = cy + noise * 10.0f;
    }
}

// ============================================================
// main
// ============================================================
int main() {
    printf("============================================\n");
    printf("   Day 21: Week3 综合 — K-Means (优化版)\n");
    printf("============================================\n\n");

    int N = 1 << 18;   // 256K
    int K = 8;
    printf("数据点: %d, 聚类数: %d, 维度: %d\n\n", N, K, DIM);

    size_t pointsSize    = (size_t)N * DIM * sizeof(float);
    size_t centroidsSize = (size_t)K * DIM * sizeof(float);

    float *h_points = (float *)malloc(pointsSize);
    srand(42);
    generateData(h_points, N, K);

    // Device 分配
    float  *d_points, *d_centroids, *d_centroidSums, *d_minDist;
    int    *d_assignments, *d_clusterCounts;
    double *d_partialSums;

    CUDA_CHECK(cudaMalloc(&d_points,        pointsSize));
    CUDA_CHECK(cudaMalloc(&d_centroids,     centroidsSize));
    CUDA_CHECK(cudaMalloc(&d_centroidSums,  centroidsSize));
    CUDA_CHECK(cudaMalloc(&d_assignments,   N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusterCounts, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_minDist,       N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_points, h_points, pointsSize, cudaMemcpyHostToDevice));

    // 初始中心: 随机选 K 个点 (避免 i%K 生成 + 前 K 点初始化的"作弊")
    float *h_init = (float *)malloc(centroidsSize);
    for (int k = 0; k < K; k++) {
        int p = rand() % N;
        h_init[k * DIM]     = h_points[p * DIM];
        h_init[k * DIM + 1] = h_points[p * DIM + 1];
    }
    CUDA_CHECK(cudaMemcpy(d_centroids, h_init, centroidsSize, cudaMemcpyHostToDevice));

    int    gridSize    = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t partialSize = (size_t)gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_partialSums, partialSize));

    // 复用 host 缓冲区
    double *h_partial = (double *)malloc(partialSize);

    // 计时
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));

    printf("--- K-Means 迭代 ---\n");
    double prevInertia = -1.0;
    int    finalIter   = 0;
    for (int iter = 0; iter < MAX_ITERS; iter++) {
        assignPoints<<<gridSize, BLOCK_SIZE>>>(d_points, d_centroids,
                                                 d_assignments, d_minDist, N, K);
        CHECK_LAUNCH();

        CUDA_CHECK(cudaMemset(d_centroidSums,  0, centroidsSize));
        CUDA_CHECK(cudaMemset(d_clusterCounts, 0, K * sizeof(int)));

        computeCentroids<<<gridSize, BLOCK_SIZE>>>(d_points, d_assignments,
                                                     d_centroidSums, d_clusterCounts, N, K);
        CHECK_LAUNCH();
        updateCentroids<<<(K + 31) / 32, 32>>>(d_centroids, d_centroidSums,
                                                 d_clusterCounts, K);
        CHECK_LAUNCH();

        computeInertia<<<gridSize, BLOCK_SIZE>>>(d_minDist, d_partialSums, N);
        CHECK_LAUNCH();

        CUDA_CHECK(cudaMemcpy(h_partial, d_partialSums, partialSize, cudaMemcpyDeviceToHost));
        double inertia = 0.0;
        for (int i = 0; i < gridSize; i++) inertia += h_partial[i];

        finalIter = iter + 1;
        if (prevInertia < 0.0) {
            // 第一次迭代: 不计算 improvement
            printf("  迭代 %3d: inertia = %15.2f\n", finalIter, inertia);
        } else {
            double improvement = (prevInertia - inertia) / prevInertia * 100.0;
            printf("  迭代 %3d: inertia = %15.2f  (改善 %.4f%%)\n",
                   finalIter, inertia, improvement);
            if (improvement >= 0.0 && improvement < 0.01) {
                printf("\n  收敛! (改善 < 0.01%%)\n");
                prevInertia = inertia;
                break;
            }
        }
        prevInertia = inertia;
    }

    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float totalMs;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, evStart, evStop));
    printf("\n总耗时: %.3f ms (%d 次迭代, 平均 %.3f ms/iter)\n",
           totalMs, finalIter, totalMs / finalIter);

    // 最终结果
    float *h_centroids = (float *)malloc(centroidsSize);
    int   *h_counts    = (int   *)malloc(K * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_centroids, d_centroids,     centroidsSize,    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts,    d_clusterCounts, K * sizeof(int),  cudaMemcpyDeviceToHost));

    printf("\n--- 最终聚类中心 ---\n");
    for (int k = 0; k < K; k++) {
        printf("  簇 %d: (%.2f, %.2f)  -> %d 点\n",
               k, h_centroids[k * DIM], h_centroids[k * DIM + 1], h_counts[k]);
    }

    printf("\n--- Week 3 总结 ---\n");
    printf("  Day 15: 并行归约\n");
    printf("  Day 16: 前缀和 (Scan)\n");
    printf("  Day 17: 分块矩阵乘法进阶\n");
    printf("  Day 18: 直方图 (原子操作)\n");
    printf("  Day 19: 流压缩\n");
    printf("  Day 20: 双调排序\n");
    printf("  Day 21: K-Means (综合应用)\n");

    // 清理
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_centroidSums));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_clusterCounts));
    CUDA_CHECK(cudaFree(d_minDist));
    CUDA_CHECK(cudaFree(d_partialSums));
    free(h_points); free(h_init); free(h_centroids); free(h_partial); free(h_counts);

    printf("\n✓ Day 21 完成! Week 3 圆满结束!\n");
    return 0;
}

