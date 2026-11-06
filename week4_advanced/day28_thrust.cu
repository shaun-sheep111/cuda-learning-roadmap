/*
 * Day 28: Thrust 库 — 高级并行编程
 * =====================================
 * 学习目标:
 *   1. 了解 Thrust 库 (类似 C++ STL 的 CUDA 接口)
 *   2. 使用 thrust::device_vector 管理 GPU 内存
 *   3. 使用 Thrust 算法: sort, reduce, scan, transform
 *   4. 理解 Thrust 的优势: 代码简洁 + 高性能
 *
 * Thrust 特点:
 *   - C++ 模板库, 类似 STL
 *   - 自动管理内存 (RAII)
 *   - 高级算法接口 (无需手写 kernel)
 *   - 底层自动选择最优实现
 *
 * 编译: nvcc day28_thrust.cu -o day28
 * 运行: ./day28
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

// ------ 自定义 functor ----
struct squareFunctor{
    __host__ __device__ float operator()(float x) const {
        return x * x;
    }
};

struct sigmoidFunctor {
    __host__ __device__ float operator()(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};

struct isPositive {
    __host__ __device__ bool operator()(float x) const {
        return x > 0.0f;
    }
};

// ---------- 极简计时辅助 ----------
template <typename F>
static float timed(F&& f) {
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    f();
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(stop - start).count();
}

int main(){
    printf("============================================\n");
    printf("   Day 28: Thrust 库\n");
    printf("============================================\n\n");

    const int N = 1 << 20;  // 1M

    // --- 1. 创建和填充 device_vector ---
    printf("--- 1. 向量创建与填充 ---\n");
    thrust::device_vector<float> h_vec(N);

    thrust::minstd_rand rng(42);
    thrust::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (int i = 0; i < N; i++) {
        h_vec[i] = dist(rng);
    }

    thrust::device_vector<float> d_vec = h_vec;

    printf("  向量大小: %d 元素\n", N);
    printf("  前5个值: %.2f %.2f %.2f %.2f %.2f\n\n",
           h_vec[0], h_vec[1], h_vec[2], h_vec[3], h_vec[4]);

    // --- 2. Transform ---
    printf("--- 2. Transform (逐元素变换) ---\n");
    {
        thrust::device_vector<float> d_result(N);

        float ms = timed([&] {
            thrust::transform(d_vec.begin(), d_vec.end(),
                              d_result.begin(), squareFunctor());
        });
        printf("  平方变换: 耗时 %.3f ms\n", ms);
        printf("  示例: %.2f^2 = %.2f\n", h_vec[0], (float)d_result[0]);

        thrust::transform(d_vec.begin(), d_vec.end(),
                          d_result.begin(), sigmoidFunctor());
        printf("  Sigmoid 变换: 示例 %.2f -> %.2f\n", h_vec[0], (float)d_result[0]);
    }

    // --- 3. Reduce ---
    printf("--- 3. Reduce (归约) ---\n");
    {
        float sum = 0.0f;
        float ms = timed([&] {
            sum = thrust::reduce(d_vec.begin(), d_vec.end(),
                                 0.0f, thrust::plus<float>());
        });
        printf("  求和: %.2f, 耗时 %.3f ms\n", sum, ms);

        float maxVal = thrust::reduce(d_vec.begin(), d_vec.end(),
                                      -1e30f, thrust::maximum<float>());
        printf("  最大值: %.2f\n", maxVal);

        float minVal = thrust::reduce(d_vec.begin(), d_vec.end(),
                                      1e30f, thrust::minimum<float>());
        printf("  最小值: %.2f\n", minVal);
    }
    
    // --- 4. Sort ---
    printf("--- 4. Sort (排序) ---\n");
    {
        thrust::device_vector<float> d_sort = d_vec;

        float ms = timed([&] {
            thrust::sort(d_sort.begin(), d_sort.end());
        });
        printf("  排序 %d 元素: %.3f ms\n", N, ms);
        printf("  最小值: %.2f, 最大值: %.2f\n\n",
               (float)d_sort[0], (float)d_sort[N - 1]);
    }

    // --- 5. Scan ---
    printf("--- 5. Scan (前缀和) ---\n");
    {
        thrust::device_vector<float> d_scan(N);

        float ms = timed([&] {
            thrust::exclusive_scan(d_vec.begin(), d_vec.end(), d_scan.begin());
        });
        printf("  Exclusive scan: %.3f ms\n", ms);
        printf("  scan[0]=%.2f, scan[%d]=%.2f\n\n",
               (float)d_scan[0], N - 1, (float)d_scan[N - 1]);
    }

    // --- 6. Count / Copy If ---
    printf("--- 6. Count & Copy If ---\n");
    {
        int count = thrust::count_if(d_vec.begin(), d_vec.end(), isPositive());
        printf("  正数个数: %d / %d\n", count, N);

        thrust::device_vector<float> d_filtered(N);
        auto new_end = thrust::copy_if(d_vec.begin(), d_vec.end(),
                                       d_filtered.begin(), isPositive());
        int newSize = (int)(new_end - d_filtered.begin());
        printf("  过滤后大小: %d (删除了 %d 个负数)\n\n", newSize, N - newSize);
    }

    // --- 7. Transform Reduce ---
    printf("--- 7. Transform Reduce (融合操作) ---\n");
    {
        float sumOfSquares = 0.0f;
        float ms = timed([&] {
            sumOfSquares = thrust::transform_reduce(
                d_vec.begin(), d_vec.end(), squareFunctor(),
                0.0f, thrust::plus<float>());
        });
        printf("  Sum of squares: %.2f (耗时 %.3f ms)\n", sumOfSquares, ms);
        printf("  (融合了 transform + reduce, 无需中间数组)\n\n");
    }

    printf("--- Thrust 使用建议 ---\n");
    printf("  ✓ 原型开发: 快速验证算法正确性\n");
    printf("  ✓ 标准算法: sort, reduce, scan 等\n");
    printf("  ✓ 内存管理: device_vector 自动管理\n");
    printf("  ✓ 自定义 functor: 灵活扩展\n");
    printf("  ✗ 性能关键路径: 可能不如手写 kernel\n");
    printf("  ✗ 特殊数据布局: Thrust 可能不适用\n");

    printf("\n✓ Day 28 完成! Thrust 让 CUDA 编程更简洁高效。\n");
    printf("  Week 4 结束, 明天进入实战项目阶段!\n");
    return 0;
}