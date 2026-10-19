# 从线程索引到显存地址

> 本章把一行 CUDA 代码拆成四层：代码层、逻辑层、地址层、硬件层。

配套 3D 动画：[`memory_address_3d.html`](memory_address_3d.html)。动画中的 token 会从线程索引出发，经过地址计算和缓存层级，最后到达 VRAM。

## 1. 先看完整链路

```text
【代码层】 blockIdx.x, blockDim.x, threadIdx.x   (三个整数)
                 │
                 │ blockIdx.x * blockDim.x + threadIdx.x
                 ▼
【逻辑层】 idx  （一维数组下标，通常暂存在寄存器）
                 │
                 │ input + idx * sizeof(float)
                 ▼
【地址层】 线性字节地址 = base_addr + idx * 4
                 │
                 │ 编译器生成的 load/store 指令
                 │ （例如 SASS 中可能看到 LDG.E / STG.E）
                 ▼
【硬件层】 地址翻译与缓存查找 → L1 → L2 → 显存分区 / VRAM
```

`LDG.E` / `STG.E` 是某些架构和编译选项下可以在 SASS 中观察到的指令形式，不应当作为 CUDA 源码接口硬编码。CUDA 源码只表达 load/store，具体指令由编译器和目标架构决定。

## 2. 代码如何得到一个地址

典型的一维 kernel：

```cpp
__global__ void copy_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}
```

假设 `blockDim.x = 256`、`blockIdx.x = 3`、`threadIdx.x = 7`：

```text
idx = 3 * 256 + 7 = 775
input[775]  = input_base  + 775 * sizeof(float)
output[775] = output_base + 775 * sizeof(float)
```

这里的 `idx` 是逻辑下标；`input_base + idx * 4` 才是一个字节地址。GPU 内部使用虚拟地址和内存管理单元完成映射，最终请求会被分发到相应的 L1、L2 和显存分区。

## 3. 内存层级量级

```text
┌─────────────────────────────────────────────────────────────┐
│ 寄存器 Registers       每线程上限常见为 255 个  ~1 周期       │
│                       SM 内寄存器文件总量约数百 KB           │
├─────────────────────────────────────────────────────────────┤
│ 共享内存 Shared Mem    每 block 常见 48~99 KB  ~20~30 周期   │
│                       片上，由程序显式管理                  │
├─────────────────────────────────────────────────────────────┤
│ L1 / L2 Cache          L2 常见为几十 MB       数十~数百周期  │
│                       L2 是整个 GPU 共享                    │
├─────────────────────────────────────────────────────────────┤
│ 全局显存 VRAM          24 GB / 48 GB          数百周期        │
│                       HBM / GDDR，nvidia-smi 可见容量        │
└─────────────────────────────────────────────────────────────┘
```

这些数字是教学用量级，不是所有 GPU 的固定规格：寄存器上限、shared memory 配置、L2 容量和访问延迟会随架构、数据类型、命中情况、访问并发和时钟变化。`nvidia-smi` 主要显示设备级显存容量和使用量，不显示每个线程的寄存器内容或每个 SM 的 shared memory 使用量。

| 层级 | 作用域 | 典型用途 | 主要风险 |
|---|---|---|---|
| 寄存器 | 单线程 | `idx`、局部变量、累加器 | 用量过高导致 spill 到 local memory |
| Shared memory | Block | tile、halo、block 内复用 | 同步、bank conflict、占用率下降 |
| L1 | SM 附近的硬件缓存 | 自动缓存近期访问 | 命中率受访问模式影响 |
| L2 | 整个 GPU | 跨 SM 的共享缓存层 | 容量和带宽有限，命中仍有延迟 |
| VRAM | 全局设备内存 | 大数组、权重、KV cache | 延迟高，必须关注合并访问和搬运量 |

## 4. 一个 warp 如何访问连续数组

对于 `float input[]`，若一个 warp 的线程连续执行 `idx = base + lane`：

```text
lane 0  → base + 0 * 4
lane 1  → base + 1 * 4
lane 2  → base + 2 * 4
...
lane 31 → base + 31 * 4
```

这些地址相邻，内存系统有机会把访问合并成较少的事务。若写成 `idx = base + lane * stride`，或者线程访问随机索引，事务数量和有效带宽可能变差。判断合并访问时，要看一个 warp 的 32 个 lane 一起访问什么地址，而不是只看单个线程的指针。

## 5. 从地址链路推导优化动作

```text
idx 不连续
  → warp 地址不连续
  → cache line / memory transaction 浪费
  → 有效带宽下降
  → 重新布局数据、调整映射或使用 tile

中间结果反复写 VRAM
  → 读写字节数增加
  → 内存受限
  → 融合 kernel、放入寄存器/shared、使用异步流水

每线程寄存器过多
  → SM 能驻留的 warp/block 变少
  → 隐藏内存延迟的能力下降
  → 减少临时变量、调整 tile、检查 spill
```

这条链路可以直接连接到 AI Infra：FlashAttention 减少 attention 中间矩阵写入 VRAM；vLLM 用分页 block 管理 KV cache 的物理地址；CUTLASS 用 tile 和流水让 Tensor Core 持续获得寄存器/shared 中的数据。

## 6. 建议观察

1. 编译一个简单 kernel，使用 `cuobjdump --dump-sass` 或 Nsight Compute 观察实际 load/store 指令；不要假设源码中的 `input[idx]` 必然对应某一个固定 SASS 指令。
2. 修改 `idx` 的 stride，比较 `week2_memory/day12_global_coalescing.cu` 的有效带宽。
3. 对比 global memory 直接访问和 shared memory tile，记录数据复用次数、shared memory 使用量和 occupancy。
4. 打开 [`memory_address_3d.html`](memory_address_3d.html)，逐步观察 `idx`、字节地址和缓存层级的关系。
