/*
 * Day 29: 实战项目 — 2D 图像卷积 (高斯模糊)
 * =============================================
 * 本日综合运用所学知识实现一个实用的图像处理算法:
 *   - 共享内存 (tiled 卷积)
 *   - 常量内存 (卷积核)
 *   - 合并内存访问
 *   - 边界处理
 *
 * 2D 卷积是图像处理的基石:
 *   - 高斯模糊 (去噪)
 *   - 边缘检测 (Sobel)
 *   - 锐化
 *   - 深度学习中的卷积层
 *
 * 算法:
 * output[y][x] = sum_{j=-R}^{R} sum_{i=-R}^{R} input[y+j][x+i] * kernel[j+R][i+R]
 *
 * 编译: nvcc -arch=sm_89 day29_image_blur.cu -o day29
 * 运行: ./day29
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define RADIUS 5          // 卷积核半径 (11x11 kernel)
#define KERNEL_SIZE (2 * RADIUS + 1)
#define SM (TILE_SIZE + 2 * RADIUS)   // 共享内存 tile 边长 (26)

// 卷积核放在常量内存 (所有线程共享, 广播)
__constant__ float c_kernel[KERNEL_SIZE * KERNEL_SIZE];

// ============================================================
// V1: 朴素 2D 卷积 (全局内存)
// 每个像素需要 KERNEL_SIZE^2 次全局内存读取
// ============================================================
__global__ void conv2DNaive(const float *input, float *output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    for (int j = -RADIUS; j <= RADIUS; j++) {
        for (int i = -RADIUS; i <= RADIUS; i++) {
            int sx = x + i;
            int sy = y + j;
            // clamp 边界
            sx = max(0, min(width - 1, sx));
            sy = max(0, min(height - 1, sy));
            sum += input[sy * width + sx] * c_kernel[(j + RADIUS) * KERNEL_SIZE + (i + RADIUS)];
        }
    }
    output[y * width + x] = sum;
}

// ============================================================
// V2: 共享内存 2D 卷积 (tiled + halo)
//
// 正确性要点:
//   - 整个 block 协作加载 (TILE_SIZE + 2*RADIUS)^2 的完整区域，
//     包括中心 tile 和四周 halo。
//   - 每个共享内存位置都按 clamp 后的全局坐标读取，
//     因此无论 width/height 是否为 TILE_SIZE 的整数倍，
//     shared memory 都不会有未初始化位置被卷积读到。
// ============================================================
__global__ void conv2DShared(const float *input, float *output, int width, int height) {
    __shared__ float sdata[SM][SM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 当前 block 负责的输出 tile 的全局左上角坐标
    int blockX = blockIdx.x * TILE_SIZE;
    int blockY = blockIdx.y * TILE_SIZE;

    // 对应 shared tile 的全局左上角坐标（向外扩 RADIUS）
    int gx0 = blockX - RADIUS;
    int gy0 = blockY - RADIUS;

    // 统一加载整个 SM x SM 区域（中心 + halo），越界 clamp
    int tid = ty * blockDim.x + tx;
    int nthreads = blockDim.x * blockDim.y;
    for (int idx = tid; idx < SM * SM; idx += nthreads) {
        int sy = idx / SM;
        int sx = idx % SM;
        int gx = gx0 + sx;
        int gy = gy0 + sy;
        gx = max(0, min(width  - 1, gx));
        gy = max(0, min(height - 1, gy));
        sdata[sy][sx] = input[gy * width + gx];
    }

    __syncthreads();

    //卷积计算(从共享内存读取)
    int x = blockX + tx;
    int y = blockY + ty;
    if (x < width && y < height) {
        int s_x = tx + RADIUS;
        int s_y = ty + RADIUS;
        float sum = 0.0f;
        for (int j = -RADIUS; j <= RADIUS; j++) {
            for (int i = -RADIUS; i <= RADIUS; i++) {
                sum += sdata[s_y + j][s_x + i] *
                       c_kernel[(j + RADIUS) * KERNEL_SIZE + (i + RADIUS)];
            }
        }
        output[y * width + x] = sum;
    }
}

// ============================================================
// CPU 参考实现
// ============================================================
void conv2DCPU(const float *input, float *output, const float *kernel,
               int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int j = -RADIUS; j <= RADIUS; j++) {
                for (int i = -RADIUS; i <= RADIUS; i++) {
                    int sx = x + i;
                    int sy = y + j;
                    sx = (sx < 0) ? 0 : (sx >= width  ? width  - 1 : sx);
                    sy = (sy < 0) ? 0 : (sy >= height ? height - 1 : sy);
                    sum += input[sy * width + sx] *
                           kernel[(j + RADIUS) * KERNEL_SIZE + (i + RADIUS)];
                }
            }
            output[y * width + x] = sum;
        }
    }
}

int main() {
    printf("============================================\n");
    printf("   Day 29: 实战 — 2D 图像卷积\n");
    printf("============================================\n\n");

    int width  = 2048;
    int height = 2048;
    size_t size = (size_t)width * height * sizeof(float);
    printf("图像: %d x %d (%.2f MB)\n", width, height, (float)size / 1e6);
    printf("卷积核: %d x %d (高斯模糊)\n\n", KERNEL_SIZE, KERNEL_SIZE);

    // 生成高斯卷积核
    float h_kernel[KERNEL_SIZE * KERNEL_SIZE];
    float sigma = 2.0f;
    float kernelSum = 0.0f;
    for (int j = -RADIUS; j <= RADIUS; j++) {
        for (int i = -RADIUS; i <= RADIUS; i++) {
            float val = expf(-(float)(i * i + j * j) / (2.0f * sigma * sigma));
            h_kernel[(j + RADIUS) * KERNEL_SIZE + (i + RADIUS)] = val;
            kernelSum += val;
        }
    }
    // 归一化
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        h_kernel[i] /= kernelSum;
    }
    // 拷贝到常量内存
    cudaMemcpyToSymbol(c_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // 生成测试图像 (模拟有噪声的图像)
    float *h_input  = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float *h_ref    = (float *)malloc(size);
    srand(12345);
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)(rand() % 256) / 255.0f;
    }

    // CPU 参考
    printf("--- CPU 卷积 ---\n");
    conv2DCPU(h_input, h_ref, h_kernel, width, height);
    printf("  完成\n\n");

    // GPU
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width  + TILE_SIZE - 1) / TILE_SIZE,
                  (height + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 朴素版本
    printf("--- 朴素 2D 卷积 ---\n");
    {
        // warmup
        conv2DNaive<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 20; i++)
            conv2DNaive<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        float maxErr = 0.0f;
        for (int i = 0; i < width * height; i++)
            maxErr = fmaxf(maxErr, fabsf(h_output[i] - h_ref[i]));

        printf("  耗时: %.3f ms (20次平均)\n", ms / 20);
        printf("  吞吐: %.2f MPixels/s\n",
               (float)width * height / (ms / 20 * 1e3) / 1e6);
        printf("  最大误差: %e\n\n", maxErr);
    }

    // 共享内存版本
    printf("--- 共享内存 2D 卷积 ---\n");
    {
        // warmup
        conv2DShared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 20; i++)
            conv2DShared<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
        float maxErr = 0.0f;
        for (int i = 0; i < width * height; i++)
            maxErr = fmaxf(maxErr, fabsf(h_output[i] - h_ref[i]));

        printf("  耗时: %.3f ms (20次平均)\n", ms / 20);
        printf("  吞吐: %.2f MPixels/s\n",
               (float)width * height / (ms / 20 * 1e3) / 1e6);
        printf("  最大误差: %e\n\n", maxErr);
    }

    printf("--- 优化分析 ---\n");
    printf("  朴素: 每像素 %d 次全局读取, 大量冗余\n", KERNEL_SIZE * KERNEL_SIZE);
    printf("  共享内存: 每 block 协作加载 halo, 减少冗余\n");
    printf("  常量内存: 卷积核广播, 所有线程共享缓存\n");
    printf("  进一步优化: 分离卷积 (2D→两个1D), 计算 O(K)→O(2*sqrt(K))\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    free(h_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n✓ Day 29 完成! 图像卷积是 CUDA 实战的经典案例。\n");
    return 0;
}