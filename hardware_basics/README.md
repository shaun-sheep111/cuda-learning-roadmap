# GPU 硬件基础

> **写 CUDA 前应该先懂的硬件知识。**
> 理解 GPU 的物理结构，才能明白"为什么这样写快、那样写慢"——
> 线程为什么按 32 个一组？共享内存为什么快？为什么要合并访问？答案都在硬件里。

建议在开始 Week 1 之前或同期阅读本模块，它是所有性能优化的地基。

**章节导航**：上一章：[第 0 章 并行计算思维](../parallel_thinking/README.md) ｜ 第 1 章 · GPU 硬件基础 ｜ 下一章：[第 2 章 CUDA 基础](../week1_basics/)

## 学习入口

本目录现在分成两条互补的线：

- [并行计算思维](../parallel_thinking/)：从任务拆分、依赖、通信、同步和性能模型出发设计算法。
- [GPU 硬件工作流程](architecture_workflow.md)：跟踪 kernel 从 CPU 入队、block 分发、warp 发射、内存访问到结果写回的全过程。
- [GPU 工作流程动画](gpu_workflow_animation.html)：用播放和单步操作观察一次向量加法的完整数据路径。
- [从线程索引到显存地址](memory_address_pipeline.md)：理解 `blockIdx/blockDim/threadIdx → idx → 字节地址 → L1/L2/VRAM`。
- [内存地址 3D 动画](memory_address_3d.html)：用 3D token 观察地址和内存层级的变化。本地双击打开即可；也可[在线查看（htmlpreview）](https://htmlpreview.github.io/?https://github.com/shaun-sheep111/cuda-learning-roadmap/blob/main/hardware_basics/memory_address_3d.html)，拖动旋转、滚轮缩放、方向键步进。
- [同步 vs 异步传输](sync_async_transfer.md)：讲清同步 / 单 stream 异步 / 双 stream pipeline 三种传输方式、各需哪些 GPU 结构、什么场景用哪种（Day 13 配套）。配套[动画（在线查看）](https://shaun-sheep111.github.io/cuda-learning-roadmap/hardware_basics/sync_async_transfer.html)可播放/单步看传输与计算如何重叠。
- [分块矩阵乘法](tiled_matmul.md)：讲清分块(tiling)为什么快、朴素/基础tiled/8×8 register 三版本对比、以及 TILE_SIZE/BM/BN/BK/每线程元素等参数如何设计（Day 14 配套）。配套[动画（在线查看）](https://shaun-sheep111.github.io/cuda-learning-roadmap/hardware_basics/tiled_matmul.html)用"数内存访问次数"看清复用带来的加速。
- [SGEMM 优化链（进阶）](sgemm_optimize.md)：从 tiled 继续推进到接近 cuBLAS —— BM×BN/TM×TN 两层分块、float4 向量化、双缓冲、消除 bank conflict，含 RTX 4090 的寄存器/共享内存/occupancy 定量分析。配套[动画（在线查看）](https://shaun-sheep111.github.io/cuda-learning-roadmap/hardware_basics/sgemm_optimize.html)逐步点看每步加速与代价。
- [Blelloch Scan（前缀和）](https://shaun-sheep111.github.io/cuda-learning-roadmap/hardware_basics/blelloch_scan.html)：工作高效前缀和动画（Day 19 流压缩配套）。逐层演示 up-sweep/down-sweep，看清 `index=(tid+1)*stride*2-1` 如何把连续线程映射到树节点、以及为什么要跳步。

推荐顺序：先读并行计算思维，再读硬件工作流程，最后运行本目录的查询和带宽实验。

---

## 一、GPU vs CPU 的设计哲学

| | CPU | GPU |
|---|-----|-----|
| 核心数 | 少（几个~几十） | 极多（数千 CUDA core） |
| 单核 | 强，低延迟，大缓存 | 弱，高吞吐 |
| 擅长 | 复杂逻辑、串行、分支 | 大规模数据并行 |
| 类比 | 几个博士 | 成千上万小学生同时算加减法 |

> 一句话：**CPU 优化延迟，GPU 优化吞吐。** GPU 靠"海量线程 + 快速切换"来隐藏内存延迟。

## 二、硬件层次结构（自顶向下）

```
GPU
 └─ 多个 SM (Streaming Multiprocessor, 流多处理器)   ← RTX 4090 有 128 个
     └─ 多个 CUDA Core + Tensor Core + 寄存器 + 共享内存/L1
         └─ Warp (32 个线程为一组, 硬件调度单位)
             └─ Thread (单个线程)
```

**软件层次（编程模型）与之对应：**

```
Grid (整个 kernel 的所有线程)
 └─ Block (线程块, 被分配到某个 SM 上执行)
     └─ Warp (32 线程, SIMT 执行)
         └─ Thread
```

> 关键映射：**一个 Block 跑在一个 SM 上；一个 Block 被拆成若干 Warp 调度。**

这张图是编程模型到硬件资源的第一层映射。实际执行还会经过命令队列、工作分发器、warp scheduler、各类计算管线、L1/L2 和 DMA 引擎；完整时间线见 [GPU 硬件工作流程](architecture_workflow.md)。

## 三、Warp —— 最重要的概念

- GPU 以 **32 个线程为一个 warp** 一起执行（SIMT：同指令多线程）
- **一个 warp 里所有线程同一时刻执行同一条指令**
- 由此产生两个核心性能问题：
  - **线程束分化 (warp divergence)**：warp 内线程走了不同的 if 分支 → 串行执行各分支 → 变慢。所以 block size 常取 32 的倍数。
  - **合并访问 (coalescing)**：warp 内 32 个线程访问**连续**内存 → 合并成一次事务 → 快；访问分散 → 多次事务 → 慢。

## 四、内存层次（速度 快→慢，容量 小→大）

| 内存 | 速度 | 作用域 | 关键点 |
|------|------|--------|--------|
| 寄存器 Register | 最快 | 单线程 | 每线程私有，用超会 spill 到 local |
| 共享内存 Shared | 很快 | Block 内共享 | 手动管理的缓存，注意 bank conflict |
| L1 / L2 缓存 | 快 | SM / 全局 | 硬件自动 |
| 全局内存 Global | 慢(相对) | 所有线程 | 显存(GDDR6X)，要合并访问 |
| 常量/纹理 | 有缓存 | 只读 | 广播/空间局部性场景 |

> 优化的本质：**尽量把数据放在快的内存里、减少访问慢内存的次数**。
> 这就是 Week 2「内存模型」和分块(tiling)技术的意义。

## 五、Tensor Core（RTX 4090 第 4 代）

- 专门做 **矩阵乘加 (D = A·B + C)** 的硬件单元，比普通 CUDA core 做矩阵乘快数倍~数十倍
- 支持 FP16 / BF16 / TF32 / INT8 / FP8 等低精度
- 深度学习、CUTLASS/cuBLAS 的高性能 GEMM 都靠它
- 编程接口：WMMA（Week 7）、或经由 CUTLASS/cuBLAS 间接使用

---

## 本目录代码

| 文件 | 主题 | 说明 |
|------|------|------|
| `query_device.cu` | 查询本机 GPU 硬件参数 | SM 数、warp size、各类内存大小、计算能力 |
| `bandwidth_test.cu` | 实测显存带宽 | H2D/D2H/D2D 带宽，理解"带宽墙" |

编译运行：
```bash
nvcc -O2 -std=c++17 -arch=sm_89 hardware_basics/query_device.cu -o query && ./query
nvcc -O2 -std=c++17 -arch=sm_89 hardware_basics/bandwidth_test.cu -o bw && ./bw
```

## 参考

- CUDA C++ Programming Guide — Hardware Model：https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-implementation
- CUDA C++ Best Practices Guide：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- NVIDIA Ada Lovelace 架构白皮书（RTX 4090）：https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/
