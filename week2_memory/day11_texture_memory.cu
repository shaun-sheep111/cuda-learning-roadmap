/*
 * Day 11: 纹理内存 — 空间局部性优化
 * ====================================
 * 学习目标:
 *   1. 理解纹理内存的特点和适用场景
 *   2. 掌握 CUDA 纹理对象的创建和使用
 *   3. 理解纹理缓存的 2D 空间局部性
 *   4. 使用纹理内存的硬件特性: 自动边界处理、插值
 *
 * 纹理内存特点:
 *   - 有专用的 2D 空间缓存, 适合有空间局部性的访问模式
 *   - 支持硬件插值 (线性插值免费!)
 *   - 支持自动边界处理 (clamp, wrap, mirror)
 *   - 适合图像处理、有限元、 stencil 计算
 *   - cuda13.0版本编译
 * 现代 API: 使用 cudaTextureObject_t (CUDA 3.x+)
 * 旧 API: __device__ __bind_to_texture__ 已弃用
 *
 * 编译: nvcc day11_texture_memory.cu -o day11
 * 运行: ./day11
 */

#include <stdio.h>

// ============================================================
// Kernel 1: 使用纹理内存的 2D stencil 计算
// 利用纹理缓存的 2D 空间局部性
// ============================================================
__global__ void stencil2DTexture(cudaTextureObject_t tex,
                               float *output, int width, int height) {
    // 计算当前线程对应的像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > width || y >= height)  return;
    
    //  5 点 stencil: center + up + down + left + right
    // tex2D 自动处理边界 (默认 clamp to edge)
    float center = tex2D<float>(tex, x, y);
    float left   = tex2D<float>(tex, x - 1, y);
    float right  = tex2D<float>(tex, x + 1, y);
    float up    = tex2D<float>(tex, x, y - 1);
    float down = tex2D<float>(tex, x, y + 1);

    // 简单的平均滤波
    output[y * width + x] = (center + left + right + up + down) / 5.0f;
}
// ============================================================
// Kernel 2: 全局内存版本 (对比用)
// ============================================================
__global__ void stencil2DGlobal(float *input, float *output,
                              int width, int height) {
    // 计算当前线程对应的像素坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)  return;

    float center = input[y * width + x];
    float up     = (y > 0)          ? input[(y - 1) * width + x] : center;
    float down   = (y < height - 1) ? input[(y + 1) * width + x] : center;
    float left   = (x > 0)          ? input[y * width + x - 1]   : center;
    float right  = (x < width - 1)  ? input[y * width + x + 1]   : center;

    // 简单的平均滤波
    output[y * width + x] = (center + left + right + up + down) / 5.0f;
}   

// ============================================================
// Kernel 3: 使用纹理的硬件插值
// 对非整数坐标, tex2D 自动做双线性插值
// ============================================================
__global__ void bilinearSample(cudaTextureObject_t tex, float *output,
                                int width, int height) {
    // 计算当前线程对应的像素坐标
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= width * height)  return;

    // 生成螺旋采样路径
    int x = idx % width;
    int y = idx / width;

    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float r = 50.0f + (float)(idx % 100) * 0.5f;
    float angle = (float)idx * 0.01f; //旋转角度

    float sx = cx + r * cosf(angle);
    float sy = cy + r * sinf(angle);

    // tex2D 自动做双线性插值
    output[idx] = tex2D<float>(tex, sx, sy);
}       

// 辅助: 创建 2D 纹理对象
cudaTextureObject_t createTexture2D(float *h_data, int width, int height){
    //1. 创建 CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, h_data, width * height * sizeof(float),
                      cudaMemcpyHostToDevice);
    
    // 2. 创建资源描述符
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

     // 3. 创建纹理描述符
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;  // 边界 clamp
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;  // 线性插值!
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;  // 使用非归一化坐标

    // 4. 创建纹理对象
    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}

int main()
{
    printf("============================================\n");
    printf("   Day 11: 纹理内存\n");
    printf("============================================\n\n");

    int width = 4096;
    int height = 4096;
    size_t size = (size_t)width * height * sizeof(float);
    printf("图像: %d x %d (%.2f MB)\n\n", width, height, (float)size / 1e6);

    float *h_input  = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_input[y * width + x] = sinf(x * 0.01f) * cosf(y * 0.01f) * 100.0f;
        }
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // 创建纹理对象
    cudaTextureObject_t texObj = createTexture2D(h_input, width, height);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    // --- 纹理内存版本 ---
    printf("--- 2D Stencil (纹理内存) ---\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        stencil2DTexture<<<gridSize, blockSize>>>(texObj, d_output, width, height);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 50; i++) {
            stencil2DTexture<<<gridSize, blockSize>>>(texObj, d_output, width, height);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  耗时: %.3f ms (50次平均)\n", ms / 50);
        printf("  有效带宽: %.2f GB/s\n", (2.0f * size) / (ms / 50 * 1e6));
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // --- 全局内存版本 ---
    printf("\n--- 2D Stencil (全局内存) ---\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        stencil2DGlobal<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 50; i++) {
            stencil2DGlobal<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("  耗时: %.3f ms (50次平均)\n", ms / 50);
        printf("  有效带宽: %.2f GB/s\n", (2.0f * size) / (ms / 50 * 1e6));
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // --- 硬件双线性插值演示 ---
    printf("\n--- 硬件双线性插值 ---\n");
    {
        float *d_samples;
        int numSamples = 1024;
        cudaMalloc(&d_samples, numSamples * sizeof(float));

        bilinearSample<<<(numSamples + 255) / 256, 256>>>(texObj, d_samples, numSamples, 1);

        float *h_samples = (float *)malloc(numSamples * sizeof(float));
        cudaMemcpy(h_samples, d_samples, numSamples * sizeof(float), cudaMemcpyDeviceToHost);

        printf("  采样值示例: %.4f, %.4f, %.4f, %.4f\n",
               h_samples[0], h_samples[1], h_samples[2], h_samples[3]);
        printf("  (这些值由硬件自动双线性插值, 无需额外计算)\n");

        cudaFree(d_samples);
        free(h_samples);
    }

    // 清理
    cudaDestroyTextureObject(texObj);
    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output);

    printf("\n--- 纹理内存适用场景 ---\n");
    printf("  ✓ 2D 空间局部性访问 (stencil, 图像处理)\n");
    printf("  ✓ 需要硬件插值 (图像缩放, 采样)\n");
    printf("  ✓ 需要自动边界处理\n");
    printf("  ✓ 需要归一化坐标\n");

    printf("\n✓ Day 11 完成! 纹理内存擅长空间局部性访问和插值。\n");
    return 0;
}


