/*
 * CUDA 编程范式模板 ③ CUTLASS 调用流程
 * ================================================
 * CUTLASS 是 header-only 模板库，需要先 clone 并配 include 路径：
 *   git clone https://github.com/NVIDIA/cutlass ~/cutlass
 * 编译:
 *   CUTLASS=~/cutlass
 *   nvcc template_cutlass.cu -o ct -arch=sm_89 -std=c++17 \
 *        -I$CUTLASS/include -I$CUTLASS/tools/util/include
 *   ./ct
 *
 * 本文件默认是「骨架 + 两层流程注释」，不依赖 CUTLASS 也能编译跑通提示信息；
 * 配好 CUTLASS 后，取消对应 #include 和代码段的注释即可真正运行。
 */
#include <cstdio>

// ===== 配好 CUTLASS 后取消注释（简单层：device-level API，sm_89 友好）=====
// #include <cutlass/gemm/device/gemm.h>
// #include <cutlass/util/host_tensor.h>
// #include <cutlass/util/reference/host/gemm.h>

int main() {
    printf("CUTLASS 调用流程范式\n");
    printf("============================================\n\n");

    printf("【简单层】device-level API（推荐 sm_89 入门）:\n");
    printf("  using Gemm = cutlass::gemm::device::Gemm<\n");
    printf("      float, cutlass::layout::RowMajor,   // A 类型/布局\n");
    printf("      float, cutlass::layout::RowMajor,   // B\n");
    printf("      float, cutlass::layout::RowMajor>;  // C\n");
    printf("  Gemm gemm_op;\n");
    printf("  Gemm::Arguments args({M,N,K}, {dA,lda}, {dB,ldb},\n");
    printf("                       {dC,ldc}, {dC,ldc}, {alpha,beta});\n");
    printf("  cutlass::Status st = gemm_op(args);   // 执行\n\n");

    printf("【进阶层】CUTLASS 3.x CollectiveBuilder（多为 Hopper 向）:\n");
    printf("  1) CollectiveBuilder<Arch, OpClass, 类型, tile...>::CollectiveOp  // mainloop\n");
    printf("  2) DefaultEpilogue<...>                                           // epilogue\n");
    printf("  3) kernel::GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue>\n");
    printf("  4) device::GemmUniversalAdapter<GemmKernel>  // host 端 handle\n");
    printf("     gemm.can_implement(args);\n");
    printf("     gemm.initialize(args, workspace);\n");
    printf("     gemm.run();\n\n");

    printf("数据准备仍用 cudaMalloc/cudaMemcpy，与手写/ cuBLAS 相同。\n");
    printf("请 clone CUTLASS、取消 #include 注释后按上面流程补全。\n");

    // ===== 配好 CUTLASS 后，简单层真正可运行的代码示例（取消注释）=====
    // using RowMajor = cutlass::layout::RowMajor;
    // using Gemm = cutlass::gemm::device::Gemm<float,RowMajor, float,RowMajor, float,RowMajor>;
    // int M=128,N=128,K=128;
    // cutlass::HostTensor<float,RowMajor> A({M,K}), B({K,N}), C({M,N});
    // // ... 初始化 A/B ...
    // Gemm gemm_op;
    // Gemm::Arguments args({M,N,K},
    //     {A.device_data(),A.stride(0)}, {B.device_data(),B.stride(0)},
    //     {C.device_data(),C.stride(0)}, {C.device_data(),C.stride(0)}, {1.f,0.f});
    // cutlass::Status st = gemm_op(args);
    // if (st != cutlass::Status::kSuccess) printf("CUTLASS 失败\n");

    return 0;
}
