# 同步 vs 异步传输：GPU 结构与场景选择

> Day 13 配套讲解。面向**完全没接触过 stream / 异步**的新手。
> 配套动画：[sync_async_transfer.html](sync_async_transfer.html)
> （本地双击打开；或 [在线查看（GitHub Pages）](https://shaun-sheep111.github.io/cuda-learning-roadmap/hardware_basics/sync_async_transfer.html)）
> 配套代码：[week2_memory/day13_pinned_memory.cu](../week2_memory/day13_pinned_memory.cu)

---

## 一、先搞懂一个最基本的问题：数据怎么从 CPU 到 GPU

GPU 自己有一块显存（VRAM），CPU 有自己的内存。一次典型的 GPU 计算分三步：

```
① H2D  把输入数据从 CPU 内存 → 搬进 GPU 显存   (Host to Device)
② 计算  GPU 的 SM 执行 kernel
③ D2H  把结果从 GPU 显存 → 搬回 CPU 内存       (Device to Host)
```

搬数据的活**不是 SM 干的**，而是一个专门的硬件单元 —— **Copy Engine（拷贝引擎 / DMA 引擎）** 干的。
现代 GPU 通常有 **两个 Copy Engine**，可以同时做一个 H2D 和一个 D2H。这个细节后面很关键。

---

## 二、三种传输方式（从慢到快）

### ① 同步串行 —— 最简单也最慢

```cpp
cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);   // 传入，CPU 在这里等
kernel<<<grid, block>>>(d_in, d_out, n);                // 计算
cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost); // 传回
```

- 三步**一个接一个**，每一步都要等上一步完全结束。
- 任何时刻**只有一个部件在干活**：传数据时 SM 空着，计算时两个 Copy Engine 都空着。
- 总时间 = 传入 + 计算 + 传回（三者相加）。

> 就像一个人洗衣服：洗完才能烘，烘完才能叠，全程只有一台机器在转。

### ② 单 stream 异步 —— CPU 不等了，但 GPU 没加速

```cpp
cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_in, d_out, n);
cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, stream);
```

- `cudaMemcpyAsync` 发出后**立刻返回**，CPU 不用傻等，可以去干别的事。
- **但是**：这三步都在**同一条 stream**里。同一条 stream 内任务严格有序（必须传完才能算），所以**三步仍然不重叠**，GPU 这边的时间线和同步几乎一样。
- **前提**：异步传输**必须用 pinned（固定）内存**（`cudaMallocHost`），否则会退化成同步。

> 收益在 **CPU 侧**（腾出 CPU），不在 GPU 侧。它更大的意义是**为 pipeline 打基础**。

### ③ 双 stream pipeline —— 真正的重叠，最快

```cpp
// 把数据切成 N 块，轮流分给 2 条 stream
for (int i = 0; i < NUM_CHUNKS; i++) {
    cudaStream_t st = s[i & 1];                 // 两条 stream 交替
    size_t off = i * chunk;
    cudaMemcpyAsync(d_in+off,  h_in+off,  chunkBytes, H2D, st);
    kernel<<<gridChunk, block, 0, st>>>(d_in+off, d_out+off, chunk);
    cudaMemcpyAsync(h_out+off, d_out+off, chunkBytes, D2H, st);
}
```

- 把大数据**切成小块**，分到**两条 stream**。
- **关键**：当 A 块在 SM 上**计算**时，B 块已经在**传入**了；A 块**回传**时，C 块在**传入**、B 块在**计算**…… 三条泳道**同时忙**。
- 这就是**重叠（overlap）**：传输和计算并行。总时间从"三步相加"缩短到"最长那条泳道"。

> 回到洗衣比喻：有两台洗衣机 + 一台烘干机，第一批在烘时，第二批已经在洗 —— 流水线。

**动画里能清楚看到**：pipeline 模式下 t=8 时刻，H2D、计算、D2H 三条泳道同时亮起。

---

## 三、同步 / 异步分别需要哪些 GPU 结构

| 结构 / 机制 | 它是干嘛的 | 同步 | 单 stream | 双 stream pipeline |
|---|---|---|---|---|
| **Copy Engine (DMA)** | 不占用 SM 就能搬数据 | 用 1 个 | 用 1 个 | **用 2 个**（H2D、D2H 同时） |
| **Pinned 内存** | 锁页内存，DMA 能直接访问 | 可选 | **必须** | **必须** |
| **CUDA Stream** | 任务队列，同队内有序、跨队可并行 | 默认流 | 1 条 | **≥2 条** |
| **SM 计算单元** | 真正算 kernel 的地方 | 有空档 | 有空档 | **尽量填满** |
| **数据分块 (chunk)** | 切小才能错峰重叠 | 不切 | 不切 | **必须切** |

一句话：**重叠 = 多 stream + 分块 + pinned 内存 + 两个 Copy Engine 协同**。缺一个都重叠不起来。

---

## 四、什么场景用哪种？

| 场景 | 推荐方式 | 原因 |
|---|---|---|
| 调试、学习、跑一次性脚本 | ① 同步 | 代码最简单，不易出错 |
| 数据量很小（传输时间可忽略） | ① 同步 | 重叠带来的收益还不够抵消复杂度 |
| CPU 侧还有别的活要并行做 | ② 单 stream 异步 | 让 CPU 不被传输阻塞 |
| **大数据、频繁传输、追求吞吐** | ③ 双 stream pipeline | 传输与计算重叠，吞吐最高 |
| 深度学习数据加载 / 流式处理 | ③ pipeline | 一边喂下一批数据，一边算当前批 |

> 经验法则：**先用同步把功能写对，确认慢在"传输 ↔ 计算切换"上，再上 pipeline 优化。** 不要一上来就写复杂的多 stream。

---

## 五、动手验证

跑一遍 Day 13 代码，看三种方式的真实耗时对比：

```bash
nvcc week2_memory/day13_pinned_memory.cu -o day13 -arch=sm_89
./day13
```

输出里会打印：
```
同步 memcpy 串行:         X.XXX ms
单 stream 异步 (退化):     X.XXX ms      ← 和同步差不多，印证"异步≠重叠"
双 stream pipeline (重叠): X.XXX ms      ← 明显更快
```

亲眼看到 `pipeline / 同步` 的加速比，就真正理解重叠的价值了。

---

**延伸阅读**：[GPU 硬件工作流程](architecture_workflow.md) ｜ [返回硬件基础](README.md)
