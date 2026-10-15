# 并行计算思维

> CUDA 是表达 GPU 程序的一种方式。真正需要掌握的是：如何拆分工作、如何安排数据、哪些地方需要通信，以及通信和同步要付出什么代价。

本模块建议在 Week 1 前阅读。读完后，看到一个算法应该先能画出任务依赖图，再决定使用哪个 CUDA API。

**章节导航**：第 0 章 · 并行计算思维 ｜ 下一章：[第 1 章 GPU 硬件基础](../hardware_basics/README.md)

## 1. 先建立正确的问题模型

### 1.1 从“一个大循环”改写成“很多独立任务”

串行代码通常描述“先做第一个元素，再做第二个元素”。并行设计要改问四个问题：

1. 一个最小任务是什么？例如一个输出元素、一个像素、一个矩阵 tile 或一个粒子。
2. 任务之间是否独立？如果不独立，依赖发生在什么阶段？
3. 每个任务需要读哪些数据、写哪些数据？谁拥有这份数据？
4. 任务完成后，哪些结果必须被别的任务看到？

例如向量加法可以写成：

```text
任务 i: C[i] = A[i] + B[i]
依赖: 无
通信: 只读 A/B，各自写 C
同步: kernel 内不需要线程间同步
```

而前缀和不能让每个元素独立完成，因为 `out[i]` 依赖 `out[i-1]`。它需要把问题改造成多个阶段的 scan，并在阶段之间交换部分和。

### 1.2 数据并行、任务并行和流水并行

| 模式 | 问题形状 | 常见例子 | 主要风险 |
|------|----------|----------|----------|
| 数据并行 | 同一操作作用于大量元素 | 向量加法、逐像素变换 | 访存不合并、负载不均 |
| 任务并行 | 不同任务执行不同工作 | 不同阶段的预处理 | 任务粒度太小、调度开销 |
| 流水并行 | 阶段可以重叠执行 | H2D → kernel → D2H | 缓冲区复用和同步错误 |
| 分层并行 | 不同层级使用不同粒度 | warp 归约、block tile、grid 归约 | 层级间通信代价 |

GPU 程序通常是组合：grid 负责大量数据并行，block 负责共享 tile，warp 负责细粒度协作，stream 负责阶段流水。

## 2. 五种最常用的并行模式

### Map

每个输出元素由一个任务独立计算：

```text
for i in parallel:
    y[i] = f(x[i])
```

适合向量运算、激活函数和逐像素处理。优先检查 warp 是否连续访问内存。

### Reduce

很多输入合并成一个或少量结果：

```text
sum = reduce(x, +)
```

先在寄存器中累积，再用 warp shuffle 或 shared memory 合并，最后用多级 kernel 完成全局归约。原子操作是工具，不是默认方案：热点地址上的原子会串行化。

### Scan

输出保留前缀关系：

```text
out[i] = x[0] + x[1] + ... + x[i]
```

Scan 是 compact、stream compaction 和 radix sort 的基础。它用更多工作换取更短的关键路径，因此要区分总工作量（work）和依赖链长度（span）。

### Stencil / Tiling

每个输出读取邻域数据，例如卷积和有限差分。相邻任务会重复读取相同输入，适合把 tile 搬进 shared memory，再重复使用。

```text
global memory -> shared tile (+ halo) -> registers -> output
```

tile 不是越大越好：它会占用 shared memory 和寄存器，降低同时驻留的 block 数。

### Gather / Scatter

- **gather**：输出任务根据索引读取输入，写入位置规则。
- **scatter**：输入任务根据索引写入输出，写入位置可能冲突。

Scatter 常需要排序、分段、私有化或原子操作来解决写冲突。先确认冲突是否真的存在，再选择同步手段。

## 3. 把通信当成算法的一部分

一次 GPU 计算不是只有“算术指令”，而是：

```text
准备数据 -> 搬运数据 -> 执行指令 -> 写回结果 -> 等待依赖
```

通信路径和可见范围决定了同步方式：

| 数据交换范围 | 典型路径 | 适合的同步 |
|--------------|----------|------------|
| 同一线程 | register | 不需要同步 |
| 同一 warp | register shuffle / vote | warp intrinsic |
| 同一 block | shared memory / L1 | `__syncwarp`, `__syncthreads` |
| 不同 block | L2 / global memory | 分阶段 kernel、原子、cooperative launch |
| CPU ↔ GPU | PCIe / NVLink + DMA | stream、event、显式同步 |
| GPU ↔ GPU | P2P、NVLink 或 PCIe | P2P copy、NCCL、event |

经验法则：**能在更小范围完成的通信，不要提升到更大范围。** 让一个 block 内完成 tile 协作，通常比让多个 block 频繁访问全局内存更便宜。

## 4. 用性能模型指导取舍

### 4.1 延迟、吞吐和并发

- 延迟：完成一次操作需要多久。
- 吞吐：单位时间完成多少数据或指令。
- 并发：同时在飞的线程、内存事务或 kernel 数量。

GPU 主要优化吞吐。单次全局内存访问可能很慢，但大量可运行 warp 可以在等待时被调度，以隐藏延迟。因此 occupancy 是“隐藏延迟的资源”，不是越高越快的分数。

### 4.2 算术强度和 Roofline

```text
算术强度 = 浮点/整数操作数 ÷ 从显存搬运的字节数
可达性能 <= min(峰值计算吞吐, 算术强度 × 有效内存带宽)
```

低算术强度的 kernel 通常是内存受限：优先优化访问合并、数据复用和搬运次数。高算术强度的 kernel 通常是计算受限：再看指令吞吐、Tensor Core、寄存器压力和发散。

### 4.3 负载均衡

平均每个线程工作量相同，不代表执行时间相同。要检查：

- 是否有少数任务远比其他任务重；
- 一个 warp 内是否出现大量不同分支；
- 是否因为原子热点让线程排队；
- 最后一个 block 是否只有很少有效元素。

不规则工作可使用 grid-stride loop、任务队列、分桶、分段扫描或两阶段算法重新分配。

## 5. 一张并行设计工作表

实现 kernel 前先写下这些答案：

```text
输入/输出:     每个张量的形状、类型和布局是什么？
最小任务:      一个线程、warp 还是 block 完成什么？
映射:          哪个维度映射到 x/y/z，是否需要 grid-stride loop？
依赖:          哪些值必须先产生？依赖跨越哪个层级？
通信:          数据在哪里交换，传输多少字节，能否复用？
同步:          最小同步范围是什么？是否需要分阶段 kernel？
访存:          一个 warp 的地址是否连续？是否会重复读？
负载:          每个任务的工作量是否均匀？
瓶颈假设:      预计受 PCIe、显存、指令还是同步限制？
验证:          并行结果如何与 CPU/reference 对比？
```

这张表可以直接对应仓库中的练习：Map 对应 Day 03，Reduce 对应 Day 06/15，Stencil/Tiling 对应 Day 14/17/29，Gather/Scatter 对应 Day 18/19，流水对应 Day 25/44/56。

## 6. 常见误区

1. **CUDA thread 不是 CPU thread 的缩小版**：线程以 warp 为单位发射指令，单个线程的独立控制流会受到 SIMT 约束。
2. **CUDA Core 数量不是可直接创建的线程数**：硬件会把大量线程分批调度到 SM，真正的上限还受寄存器、shared memory 和 block 限制。
3. **更多 block 不一定更快**：如果已经受显存带宽限制，增加线程只会增加竞争。
4. **同步不是免费的**：同步点会等待最慢线程，并可能阻止流水重叠。
5. **先测量再优化**：用 Nsight Systems 看时间线，用 Nsight Compute 看 kernel 内部指标，再验证优化是否改善端到端时间。

## 7. 建议练习顺序

1. 用 `week1_basics/day03_vector_ops.cu` 改写一个 grid-stride Map。
2. 用 `week1_basics/day06_dot_product.cu` 画出 block 内 Reduce 的数据流。
3. 用 `week2_memory/day14_week2_review.cu` 标出 global → shared → register 的每次搬运。
4. 用 `week3_patterns/day19_compact.cu` 找出 predicate、scan、scatter 三个阶段的边界。
5. 用 `week4_advanced/day25_streams_events.cu` 把 H2D、kernel、D2H 画成时间线，判断哪些阶段可以重叠。

完成这些练习后，再进入 `hardware_basics/architecture_workflow.md`，把算法层的选择和硬件层的执行逐一对上。

## 8. 面向 AI Infra 的并行思维

CUDA kernel 是底层表达，AI Infra 设计需要在更高层决定“工作如何切分、数据在哪里、通信何时发生”。vLLM、FlashAttention 等系统的核心不是某个 API，而是把模型计算重新组织成适合 GPU 硬件的并行和内存访问模式。

### 8.1 四种分布式并行方式

| 方式 | 切分对象 | 典型通信 | 适合的问题 |
|---|---|---|---|
| 数据并行 | batch / request | 梯度或结果同步 | 多个样本独立，模型可放入单卡 |
| 张量并行 | 一个层的权重或矩阵维度 | all-reduce / all-gather | 单层模型太大，或需要多卡共同完成 GEMM |
| 流水并行 | 网络层或阶段 | activation 发送、pipeline bubble | 模型跨卡放置，批量足够大 |
| 序列/上下文并行 | token、序列维度 | attention 相关的分片通信 | 长上下文和显存压力较大 |

选择方法不是“并行越多越好”，而是先比较：每张卡的计算量、显存占用、跨卡字节数、同步次数和流水空洞。通信量接近计算量时，增加 GPU 可能不会带来线性加速。

### 8.2 FlashAttention 的思维方式

朴素 attention 先物化完整的 `S = QK^T` 和概率矩阵，带来巨大的显存读写。FlashAttention 把问题改写成 block/tile 流水：

```text
Q tile + K tile -> 寄存器/shared 中计算局部分数
              -> online softmax 维护行最大值和归一化因子
              -> 直接累加 V tile 的输出
              -> 不把完整注意力矩阵写回显存
```

这里的关键并行判断是：

1. 哪些数据可以在一个 tile 内复用？
2. 哪些中间结果不值得写入 global memory？
3. softmax 的归约依赖如何用 online 算法变成可流水的局部状态？
4. shared memory、寄存器和 Tensor Core 的容量如何决定 tile 形状？

这就是“算法重排 → 减少通信 → 提高算术强度”，而不只是把原始公式翻译成 CUDA。

### 8.3 vLLM 的思维方式

LLM 推理的瓶颈不仅是矩阵乘，也包括不同长度请求带来的调度和 KV cache 管理。vLLM 的 PagedAttention 把 KV cache 从“大块连续内存”改成按 block 管理的页式存储：

```text
请求 token 序列
    ↓ 按固定 token block 切分
物理 KV blocks ← block table ← 逻辑序列位置
```

这样做的并行和系统设计收益是：

- 不需要为每条请求预留一大段连续显存，减少碎片和浪费；
- 多个请求可以共享前缀 block，减少重复计算和存储；
- 调度器可以把不同长度请求组成 continuous batching，让 GPU 及时填满；
- kernel 通过 block table gather KV，而不是假设每条序列物理连续。

这里的核心不是“写一个更快的 kernel”，而是同时设计调度、内存布局、批处理粒度和 kernel 接口。

### 8.4 从算法到系统的通用设计循环

```text
定义工作单元
  → 计算依赖和可交换的维度
  → 选择数据布局与所有权
  → 估算计算量、显存量和通信字节数
  → 选择 tile / batch / shard / pipeline 粒度
  → 设计最小同步范围
  → 用时间线和 kernel 指标验证
  → 根据瓶颈重排算法，而不是只调 block size
```

这个循环可以从单卡 CUDA 延伸到多卡推理、训练和服务系统。理解它之后，CUDA、Triton、CUTLASS、NCCL、vLLM 都可以看作同一套并行思维在不同抽象层的实现。
