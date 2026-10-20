# 分块矩阵乘法：原理与参数设计

> Day 14 配套讲解。回答两个问题：**① 分块(tiling)为什么能把矩阵乘法加速几十倍？ ② TILE_SIZE / BM / BN / BK / 每线程算几个元素，这些参数怎么设计才更快？**
> 配套动画：[tiled_matmul.html](tiled_matmul.html)（本地双击打开；或 [在线查看（GitHub Pages）](https://shaun-sheep111.github.io/cuda-learning-roadmap/hardware_basics/tiled_matmul.html)）
> 配套代码：[week2_memory/day14_week2_review.cu](../week2_memory/day14_week2_review.cu)

---

## 一、先理解瓶颈：矩阵乘法卡在哪？

矩阵乘法 `C = A × B`（A 是 M×K，B 是 K×N，C 是 M×N）。算 C 的每一个元素，需要 A 的一行 × B 的一列，共 K 次乘加。

**瓶颈不是算力，而是内存带宽。** GPU 算力极强，但数据从**全局内存（显存）**读出来很慢。如果每次乘加都去读一次全局内存，GPU 大部分时间在"等数据"，算力浪费。

> 关键指标叫**算术强度 (arithmetic intensity)** = 计算量 / 访存量。
> 这个值越高，越不受带宽限制、越能发挥算力。**所有优化本质都是在提高算术强度。**

---

## 二、三个版本：一步步减少慢内存访问

GPU 的存储速度：**寄存器 > 共享内存 > L2 > 全局内存**，每层快约一个数量级。
优化的套路就是：**把数据从慢内存搬到快内存，然后尽量多复用。**

### ① 朴素版 `matMulNaive` —— 零复用，最慢

```cpp
int row = blockIdx.y*blockDim.y + threadIdx.y;
int col = blockIdx.x*blockDim.x + threadIdx.x;
float sum = 0;
for (int i = 0; i < K; i++)
    sum += A[row*K+i] * B[i*N+col];   // 每次都读全局内存
C[row*N+col] = sum;
```

- 每个线程算 C 的 **1 个**元素，要从全局内存读 A 一整行 + B 一整列。
- **相邻线程重复读同样的数据**（算 C[0][0] 和 C[0][1] 都要读 A 第 0 行），全局内存被反复读。
- 全局内存访问 ≈ 2·M·N·K，带宽直接打满 → 慢。

### ② 基础 tiled 32×32 `matMulTiled` —— 共享内存复用

```cpp
__shared__ float s_A[32][32];
__shared__ float s_B[32][32];
for (int t = 0; t < numTiles; t++) {
    s_A[ty][tx] = A[...];   // 整块搬进共享内存，每格只读一次全局
    s_B[ty][tx] = B[...];
    __syncthreads();        // 等所有线程搬完
    for (int i = 0; i < 32; i++)
        sum += s_A[ty][i] * s_B[i][tx];  // 从共享内存读，快
    __syncthreads();
}
```

- 把 A、B 切成 `32×32` 的 **tile**，整块搬进**共享内存**一次。
- block 内 **32 个线程共享**这块 tile —— 同一份数据从全局内存只读 1 次，却被复用了 32 次。
- 全局内存读降到朴素版的约 **1/32**。这是 tiling 的核心价值。

### ③ 8×8 register tiling `matMulTiled8x8` —— 再推到寄存器层

```cpp
#define BM 128   // 每 block 算 C 的 128 行
#define BN 128   // 每 block 算 C 的 128 列
#define BK 8     // K 方向每次 tile 厚度
float acc[8][8] = {0};   // 每个线程 64 个累加器，放在寄存器

for (每个 kk)
    for i in 0..8: a[i] = As[...];   // 从共享内存读 8 个到寄存器
    for j in 0..8: b[j] = Bs[...];   // 再读 8 个
    for i,j: acc[i][j] = fma(a[i], b[j], acc[i][j]);  // 8×8=64 次 FMA
```

- 每个线程改算 **8×8 = 64 个**输出元素，中间结果全放在**寄存器**里。
- 从共享内存读一次 a/b（8+8 个值），就复用在 **64 次 FMA** 上 —— 算术强度大增。
- 三层复用：全局 → 共享 → 寄存器，GFLOPS 最高，逼近 GPU 峰值算力。

**动画板块一**把这三种方法的内存访问次数直接数给你看，对比最直观。

---

## 三、参数怎么设计才更快

**核心矛盾一句话**：
> **tile / 每线程工作量 越大 → 数据复用越多（省带宽）；但占用的寄存器、共享内存越多 → 能同时驻留在 SM 上的 block 越少（occupancy 降低，隐藏延迟能力变差）。**

调参就是在**复用收益**和 **occupancy** 之间找平衡。

| 参数 | 含义 | 调大的好处 | 调大的代价 | day14 取值 |
|---|---|---|---|---|
| `TILE_SIZE` | 共享内存 tile 边长 | 全局内存读更少 | 共享内存/block 占用↑，occupancy↓ | **32**（warp 整数倍，利于合并访问） |
| `BM × BN` | 每 block 负责的 C 区域 | 复用↑、启动开销摊薄 | 寄存器+共享内存压力↑ | **128×128** |
| `BK` | K 方向 tile 厚度 | tile 小、换入快、同步少 | 太小则循环/同步开销占比↑ | **8** |
| 每线程输出 | register tiling 粒度 | **算术强度↑** | 寄存器用量↑，用超会 spill 到本地内存变慢 | **8×8=64** |
| block 线程数 | 一个 block 多少线程 | 并行度↑ | 须 32 倍数，受寄存器/共享内存上限约束 | **16×16=256** |

### 实用的设计经验

1. **tile 边长取 warp（32）的整数倍** —— 让同一 warp 的 32 个线程访问连续地址，触发**合并访问**，一次内存事务搞定。
2. **每线程多算几个元素（register tiling）是提升算术强度最有效的手段** —— 但盯住寄存器用量，一旦 spill（寄存器不够溢出到本地内存）反而变慢。
3. **别让共享内存/寄存器吃满单个 block** —— 否则一个 SM 上同时只能跑 1~2 个 block，occupancy 太低，无法用并行隐藏访存延迟。
4. **没有万能最优值** —— 最优参数取决于 GPU 型号（SM 的寄存器/共享内存容量）和矩阵尺寸。**最终靠 Nsight Compute 实测 occupancy 和 GFLOPS 来定**，不要凭空猜。

---

## 四、动手验证

```bash
nvcc week2_memory/day14_week2_review.cu -o day14 -arch=sm_89
./day14
```

输出会打印三个 kernel 的 GFLOPS：
```
朴素:          X.XX ms   XXXX GFLOPS
基础 tiled:    X.XX ms   XXXX GFLOPS    ← 比朴素快几倍
8×8 register:  X.XX ms   XXXX GFLOPS    ← 最快，逼近峰值
```

亲眼看到 GFLOPS 一级级往上跳，就真正理解了"少读慢内存、多复用快内存"的威力。

> 进阶方向：共享内存 bank conflict 消除、double buffering（预取下一块 tile 与计算重叠）、用 Tensor Core（WMMA）—— 这些是 Week 6/7 的内容，也是 cuBLAS/CUTLASS 跑得快的秘密。

---

**延伸阅读**：[同步 vs 异步传输](sync_async_transfer.md) ｜ [GPU 硬件工作流程](architecture_workflow.md) ｜ [返回硬件基础](README.md)
