# SGEMM 优化链：从 tiled 到接近 cuBLAS

> day14 的 register tiling 只是起点。本文沿着经典的 SGEMM 优化路径，逐步加上
> **block/thread 两层分块、float4 向量化、双缓冲、消除 bank conflict**，
> 每一步给出**代码片段 + RTX 4090 上的寄存器/共享内存/occupancy 定量分析**。
>
> 配套动画：[sgemm_optimize.html](sgemm_optimize.html)（本地双击；或 [在线查看（GitHub Pages）](https://shaun-sheep111.github.io/cuda-learning-roadmap/hardware_basics/sgemm_optimize.html)）
> 代码起点：[week2_memory/day14_week2_review.cu](../week2_memory/day14_week2_review.cu)

---

## RTX 4090 (sm_89) 硬件预算

优化前先记住"钱包"有多大——所有参数都受这些约束：

| 资源（每 SM） | 上限 | 约束什么 |
|---|---|---|
| 寄存器文件 | **65536** 个 | 每线程寄存器 × 线程数 ≤ 此值 |
| 每线程最大寄存器 | **255** | 超过 → spill 到本地内存（慢） |
| 共享内存 | **100 KB** | tile 大小 + 双缓冲 |
| 最大并发 warp | **48**（1536 线程） | block线程 × 同驻block 的上限 |
| 最大 block / SM | **24** | — |
| warp 大小 | **32** | tile 边长取它的整数倍 |

---

## 优化链逐步拆解

### ② 基础 tiled 32×32（day14 的 matMulTiled）

```cpp
__shared__ float sA[32][32], sB[32][32];
sA[ty][tx]=A[...]; sB[ty][tx]=B[...];
__syncthreads();
for(int i=0;i<32;i++) sum += sA[ty][i]*sB[i][tx];
```

- 全局读降到朴素的 1/32。
- **定量**：block 1024 线程(32 warp)，共享内存 8KB，寄存器 ~40/线程。
  受"每 SM 1536 线程"限制，一个 SM 只放得下 1 个 block → **occupancy ~67%**。

### ③ 两层分块 BM×BN + TM×TN（register tiling，day14 的 matMulTiled8x8）

两层分块是高性能 SGEMM 的骨架：
- **block tile `BM×BN`**：一个 block 负责 C 的 128×128 区域。
- **thread tile `TM×TN`**：每个线程负责其中 8×8=64 个元素，累加器放寄存器。

```cpp
#define BM 128
#define BN 128
#define BK 8
float acc[8][8] = {0};           // TM×TN，64 个寄存器累加器
for (int t=0; t<numTiles; t++){
    // 加载 BM×BK 和 BK×BN 到共享内存 ...
    __syncthreads();
    for (int kk=0; kk<BK; kk++){
        float a[8], b[8];        // 从共享内存读到寄存器
        for(int i=0;i<8;i++) a[i]=sA[ty*8+i][kk];
        for(int j=0;j<8;j++) b[j]=sB[kk][tx*8+j];
        for(int i=0;i<8;i++)
          for(int j=0;j<8;j++)
            acc[i][j]=fmaf(a[i],b[j],acc[i][j]);  // 8+8 读 → 64 次 FMA
    }
    __syncthreads();
}
```

- **算术强度**：读 8+8=16 个值，做 64 次 FMA。访存/计算比大幅改善。
- **定量**：block 256 线程(8 warp)，寄存器 ~100/线程。
  `65536 / (100×256) ≈ 2` → 同驻 2 block，16 warp → **occupancy ~33%**。

> ⚠️ **最反直觉的一点**：occupancy 从 67% **降到 33%**，速度却**大涨**。
> 因为瓶颈从"访存"变成了"计算"，此时决定性能的是**算术强度和 ILP（指令级并行）**，
> 不是同时跑多少 warp。每线程 64 个独立 FMA 就足以填满计算管线。
> **"occupancy 越高越快"是常见误解**——要看瓶颈到底在哪。

### ④ float4 向量化加载

```cpp
// 一条指令读 128bit = 4 个 float
float4 v = *reinterpret_cast<const float4*>(&A[idx]);
reinterpret_cast<float4*>(sA)[k] = v;
```

- 访存**指令数减到 1/4**，更好地利用内存事务宽度（128B/事务），降低指令发射压力。
- **前提**：地址必须 **16 字节对齐**，且元素连续。
- **定量**：寄存器略降(~96)，occupancy 不变(~33%)，但访存效率↑ → 再提速 ~1.3×。

### ⑤ double buffering（双缓冲 / 预取）

```cpp
__shared__ float sA[2][BM][BK];   // 开两份
int cur = 0;
load(sA[cur]);                    // 先装第一块
for (int t=0; t<numTiles; t++){
    if (t+1<numTiles) prefetch(sA[cur^1]);  // 预取下一块
    __syncthreads();
    compute(sA[cur]);             // 同时计算当前块
    cur ^= 1;                     // 切换缓冲
}
```

- **目的**：让"加载下一块 tile"与"计算当前块"**重叠**，隐藏共享内存/全局加载延迟，并减少一次同步等待。
- **代价**：共享内存**翻倍**（8KB → 16KB）。
- **定量**：共享内存 16KB/block，`100KB/16KB ≈ 6 block`，但仍受寄存器限制为 2 block → occupancy 维持 ~33%，但停顿减少 → 再提速。

### ⑥ 消除 bank conflict

共享内存分成 **32 个 bank**，一个 warp 的 32 个线程若访问**同一 bank 的不同地址**，会被**串行化**（N-way conflict 慢 N 倍）。

列方向访问 `sA[k][ty*8+i]` 时，多个线程常落在同一 bank。解决：

```cpp
// padding：每行多留几列，错开 bank 映射
__shared__ float sA[BK][BM + 4];   // +4 让列访问落到不同 bank
```

- 也可用 **swizzle**（异或打散地址）达到同样效果。
- **效果**：共享内存有效吞吐 **≈ ×2**，消除串行停顿。
- **代价**：少量 padding 占用。

---

## 完整优化效果（相对趋势）

| 阶段 | 相对速度 | 关键技术 | occupancy | 瓶颈变化 |
|---|---|---|---|---|
| ① 朴素 | 1× | — | 高 | 全局带宽 |
| ② tiled 32×32 | ~6× | 共享内存复用 | ~67% | 带宽缓解 |
| ③ BM×BN + TM×TN | ~20× | 寄存器复用 | ~33% | **转为计算** |
| ④ + float4 | ~28× | 向量化访存 | ~33% | 访存指令 |
| ⑤ + double buffer | ~36× | 加载/计算重叠 | ~33% | 加载延迟 |
| ⑥ + bank conflict free | ~42× | 共享内存吞吐 | ~33% | 共享内存停顿 |

> 数字为**典型趋势示意**，真实加速比取决于矩阵尺寸、GPU 和 profile 结果。
> cuBLAS / CUTLASS 在此基础上还有：warp tiling、异步拷贝（`cp.async`）、Tensor Core（WMMA/MMA）、按架构定制的调度等。

---

## 参数设计总结（一张决策图）

```
目标：最大化 GFLOPS
  │
  ├─ 提升算术强度（最重要）
  │    └─ 增大 TM×TN（每线程更多元素）→ 吃寄存器，盯住 255 上限、别 spill
  │
  ├─ 提升访存效率
  │    ├─ tile 边长取 32 倍数 → 合并访问
  │    ├─ float4 向量化 → 减少访存指令
  │    └─ 消除 bank conflict → padding/swizzle
  │
  ├─ 隐藏延迟
  │    └─ double buffering → 共享内存翻倍换取重叠
  │
  └─ 平衡 occupancy
       └─ 不必追求高 occupancy；算力受限时 33% 也能满速
          用 Nsight Compute 看瓶颈（访存 or 计算）再决定
```

**黄金法则**：不要凭空调参。改一处 → 用 **Nsight Compute** 看 occupancy、内存吞吐、SM 利用率，哪个是瓶颈就优化哪个。

---

**延伸阅读**：[分块矩阵乘法(基础)](tiled_matmul.md) ｜ [同步 vs 异步传输](sync_async_transfer.md) ｜ [GPU 硬件工作流程](architecture_workflow.md) ｜ [返回硬件基础](README.md)
