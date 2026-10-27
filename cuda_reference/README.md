# CUDA 编程范式与 API 速查

> 一份完整的 CUDA 编程"范式"参考：从**手写核函数**到**调用 cuBLAS / CUTLASS** 的整套流程与关键函数。
> 不只是查单个函数，而是把它们**串成一次完整程序该怎么写**。
>
> 配套动画：[cuda_flow.html](cuda_flow.html)（[在线查看（GitHub Pages）](https://shaun-sheep111.github.io/cuda-learning-roadmap/cuda_reference/cuda_flow.html)）
> 可编译模板：[手写](template_handwritten.cu) ｜ [cuBLAS](template_cublas.cu) ｜ [CUTLASS](template_cutlass.cu)

---

## 零、一次完整 CUDA 程序的 7 步骨架

几乎所有 CUDA 程序都是这个流程（动画里逐步演示）：

```
① 分配设备内存      cudaMalloc
② 拷入数据 H2D      cudaMemcpy(..., cudaMemcpyHostToDevice)
③ 启动核函数        kernel<<<grid, block>>>(args)
④ 等待 / 计时       cudaDeviceSynchronize / cudaEvent*
⑤ 拷回结果 D2H      cudaMemcpy(..., cudaMemcpyDeviceToHost)
⑥ 释放设备内存      cudaFree
⑦ 错误检查         cudaGetLastError / CUDA_CHECK
```

记住这 7 步，CUDA 程序就不会乱。下面逐块展开。

---

## 一、核函数（Kernel）—— GPU 上跑的代码

### 函数限定符

| 限定符 | 在哪执行 | 谁能调用 | 说明 |
|---|---|---|---|
| `__global__` | **设备(GPU)** | 主机调用 | 核函数入口，用 `<<<>>>` 启动，返回必须 `void` |
| `__device__` | 设备 | 设备（其他 kernel/device 函数） | GPU 上的辅助函数 |
| `__host__` | 主机(CPU) | 主机 | 普通 CPU 函数（默认） |
| `__host__ __device__` | 两者 | 两者 | 同一函数编译出 CPU 和 GPU 两版 |

### 核函数示例

```cpp
// __global__：这就是一个核函数，每个线程执行一遍
__global__ void scanInclusiveNaive(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 当前线程的全局编号
    if (idx < n) {                                      // 边界检查（线程常多于数据）
        float sum = 0;
        for (int i = 0; i <= idx; i++) sum += in[i];    // 朴素前缀和
        out[idx] = sum;
    }
}
```

### 内置变量（核函数里直接可用）

| 变量 | 含义 | 类型 |
|---|---|---|
| `threadIdx.x/y/z` | 线程在 block 内的坐标 | uint3 |
| `blockIdx.x/y/z` | block 在 grid 内的坐标 | uint3 |
| `blockDim.x/y/z` | 每个 block 的线程数 | dim3 |
| `gridDim.x/y/z` | grid 的 block 数 | dim3 |

**最常用的一行**——算出当前线程负责第几个数据：
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 共享内存与同步

```cpp
__global__ void reduce(const float *in, float *out, int n) {
    __shared__ float s[256];              // __shared__：block 内共享的高速内存
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    s[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();                       // 等 block 内所有线程都写完 s[]，再往下

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();                   // 每轮归约后都要同步
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}
```

- `__shared__`：声明共享内存，block 内所有线程可见，比全局内存快一个量级。
- `__syncthreads()`：**block 内屏障**——所有线程都到这才继续。用于共享内存读写之间，防数据竞争。
  > ⚠️ 不能放在 `if` 分支里只让部分线程执行，否则死锁。

---

## 二、核函数启动语法 `<<<>>>`

```cpp
kernel<<< gridDim, blockDim, sharedMemBytes, stream >>>(args...);
//         ①网格大小  ②块大小    ③动态共享内存(可选) ④流(可选)
```

```cpp
int n = 1 << 20;
int block = 256;                          // 每 block 256 线程（取 32 的倍数）
int grid  = (n + block - 1) / block;      // 向上取整，保证覆盖所有数据
myKernel<<<grid, block>>>(d_in, d_out, n);
```

二维示例（如矩阵）：
```cpp
dim3 block(16, 16);
dim3 grid((N + 15)/16, (M + 15)/16);
matKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

> 核函数启动是**异步**的：CPU 发出后立刻返回，不等 GPU 算完。要等结果必须显式同步（见下）。

---

## 三、内存管理

| 函数 | 作用 | 用法 |
|---|---|---|
| `cudaMalloc(&ptr, bytes)` | 在显存分配 | `cudaMalloc(&d_A, n*sizeof(float))` |
| `cudaMemcpy(dst, src, bytes, kind)` | 主机↔设备拷贝 | kind 见下 |
| `cudaFree(ptr)` | 释放显存 | `cudaFree(d_A)` |
| `cudaMallocHost(&ptr, bytes)` | 分配**固定(pinned)内存** | 异步传输必需，更快 |
| `cudaFreeHost(ptr)` | 释放固定内存 | — |
| `cudaMemset(ptr, val, bytes)` | 显存置值 | 常用来清零 |

**`cudaMemcpy` 的方向（kind）：**
- `cudaMemcpyHostToDevice` —— H2D，拷入
- `cudaMemcpyDeviceToHost` —— D2H，拷回
- `cudaMemcpyDeviceToDevice` —— 设备内部

```cpp
float *d_A;
cudaMalloc(&d_A, n * sizeof(float));                       // ① 分配
cudaMemcpy(d_A, h_A, n*sizeof(float), cudaMemcpyHostToDevice);  // ② 拷入
// ... 启动 kernel ...
cudaMemcpy(h_A, d_A, n*sizeof(float), cudaMemcpyDeviceToHost);  // ⑤ 拷回
cudaFree(d_A);                                             // ⑥ 释放
```

---

## 四、同步与计时

### 同步

| 函数 | 作用 |
|---|---|
| `cudaDeviceSynchronize()` | 阻塞 CPU，直到 GPU 上**所有**任务完成 |
| `cudaStreamSynchronize(stream)` | 只等某条 stream |
| `__syncthreads()` | （设备端）block 内线程屏障 |

### 计时（CUDA Event，测 GPU 真实耗时的标准方法）

```cpp
cudaEvent_t start, stop;                   // ① 声明事件
cudaEventCreate(&start);                   // ② 创建
cudaEventCreate(&stop);

cudaEventRecord(start);                    // ③ 记录起点（插入 GPU 任务流）
myKernel<<<grid, block>>>(d_in, d_out, n); // 要测的操作
cudaEventRecord(stop);                     // ④ 记录终点
cudaEventSynchronize(stop);                // ⑤ 等 stop 事件真正完成

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);    // ⑥ 算出 start→stop 的毫秒数
printf("耗时 %.3f ms\n", ms);

cudaEventDestroy(start);                   // ⑦ 销毁
cudaEventDestroy(stop);
```

> 为什么不用 CPU 的计时器？因为 kernel 启动是异步的，CPU 计时器量不准 GPU 实际耗时。
> CUDA Event 插在 GPU 任务流里，测的是 GPU 墙上时间，准确。
> 测性能时常循环多次取平均：`cudaEventElapsedTime` 得到总时间再除以次数。

---

## 五、错误处理

CUDA API 返回 `cudaError_t`，但很多错误是**异步**发生的（kernel 内部），需要专门检查。

```cpp
// 包一个宏，每个 CUDA 调用都检查
#define CUDA_CHECK(call) do {                                   \
    cudaError_t e = (call);                                     \
    if (e != cudaSuccess) {                                     \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
               cudaGetErrorString(e));                          \
        exit(1);                                                \
    }                                                           \
} while(0)

CUDA_CHECK(cudaMalloc(&d_A, bytes));

myKernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());        // 查 kernel 启动是否出错（配置非法等）
CUDA_CHECK(cudaDeviceSynchronize());   // 查 kernel 执行期间是否出错
```

| 函数 | 作用 |
|---|---|
| `cudaGetLastError()` | 取并清除最后一个错误（查 kernel 启动错误） |
| `cudaPeekAtLastError()` | 取但不清除 |
| `cudaGetErrorString(err)` | 错误码转可读字符串 |

---

## 六、三条路线对比：手写 / cuBLAS / CUTLASS

同样做矩阵乘法 `C = A×B`，三种写法的定位完全不同：

| | 手写 kernel | cuBLAS | CUTLASS |
|---|---|---|---|
| 是什么 | 自己写 `__global__` | 官方**闭源**成品库 | 开源 C++ **模板**库 |
| 代码量 | 多，要自己管 tiling | 几行 API | 中等，模板组装 |
| 灵活性 | 完全可控 | 固定运算 | 可定制/融合 |
| 性能 | 看你水平 | 很高 | 可达/超 cuBLAS |
| 何时用 | 学习、特殊算子 | 标准运算、快速上线 | 定制、研究、融合 |

### ① 手写路线（本仓库 week1~ 的主线）
就是上面第零~五节的完整流程。见 [template_handwritten.cu](template_handwritten.cu)。

### ② cuBLAS 路线

```cpp
#include <cublas_v2.h>
cublasHandle_t handle;
cublasCreate(&handle);                     // 创建句柄（一次）

const float alpha = 1.f, beta = 0.f;
// C = alpha*A*B + beta*C ；注意 cuBLAS 是【列主序】
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha,
            d_A, M, d_B, K, &beta, d_C, M);

cublasDestroy(handle);                     // 销毁句柄
```
关键：`cublasCreate/Destroy` 管句柄生命周期；数据仍用 `cudaMalloc/cudaMemcpy` 准备；**列主序**是最大的坑。见 [template_cublas.cu](template_cublas.cu)。

### ③ CUTLASS 路线（两个层次）

**简单层（device-level API，sm_89 友好）：**
```cpp
#include <cutlass/gemm/device/gemm.h>
using Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,   // A
    float, cutlass::layout::RowMajor,   // B
    float, cutlass::layout::RowMajor>;  // C
Gemm gemm_op;
Gemm::Arguments args({M,N,K}, {d_A,lda}, {d_B,ldb}, {d_C,ldc}, {d_C,ldc}, {alpha,beta});
gemm_op(args);                          // 执行
```

**进阶层（CUTLASS 3.x CollectiveBuilder，多为 Hopper 向，流程示意）：**
```cpp
// 1) 构建 collective mainloop（自动选 tile/流水线深度）
using Mainloop  = cutlass::gemm::collective::CollectiveBuilder</*ArchTag, 类型, tile...*/>::CollectiveOp;
// 2) 构建 epilogue
using Epilogue  = cutlass::epilogue::collective::DefaultEpilogue</*...*/>;
// 3) 组装 kernel
using GemmKernel= cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue>;
// 4) 包成 host 端 handle
using Gemm      = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
Gemm gemm;
gemm.can_implement(args); gemm.initialize(args, workspace); gemm.run();
```
见 [template_cutlass.cu](template_cutlass.cu) 的详细注释。

---

## 七、把流程记成一张图

```
         ┌───────────────── Host (CPU) ─────────────────┐
         │  malloc h_A/h_B           准备输入            │
         │  cudaMalloc d_A/d_B/d_C   ① 分配显存          │
         │  cudaMemcpy H2D           ② 拷入 ──────┐      │
         │  cudaEventRecord(start)   计时起点      │      │
         │  kernel<<<grid,block>>>   ③ 启动 ───┐   │      │
         │     (cuBLAS: cublasSgemm)           │   ▼      │
         │     (CUTLASS: gemm_op(args))    ┌── Device (GPU) ──┐
         │  cudaEventRecord(stop)          │ 各线程并行执行     │
         │  cudaEventSynchronize           │ __shared__/__sync │
         │  cudaEventElapsedTime 计时      └──────┬───────────┘
         │  cudaMemcpy D2H           ⑤ 拷回 ◄─────┘      │
         │  cudaFree / cudaEventDestroy ⑥⑦ 清理          │
         └───────────────────────────────────────────────┘
```

---

## 八、Warp 级编程语法

Warp 是 GPU 调度线程的基本单位，当前 CUDA GPU 的 `warpSize` 为 **32**。同一 block 内，相邻的 32 个线性线程组成一个 warp；warp 内可以直接交换寄存器数据、投票和同步。

### 1. 线程编号：lane 与 warp

```cpp
// kernel 内使用；适用于一维、二维、三维 block（x 方向变化最快）
unsigned tid = threadIdx.x + blockDim.x *
               (threadIdx.y + blockDim.y * threadIdx.z);
unsigned lane = tid % warpSize;       // 当前线程在 warp 内的位置：0~31
unsigned warp_id = tid / warpSize;    // 当前 warp 在 block 内的编号
constexpr unsigned FULL_MASK = 0xffffffffu;
```

`blockDim.x = 256` 的一维 block 有 8 个完整 warp。若 block 的总线程数不是 32 的倍数，最后一个 warp 不完整，不能直接套用要求完整 warp 的示例。

### 2. 参与掩码 mask：哪些线程一起执行

`mask` 是 32 位无符号整数，第 `i` 位为 1 表示 lane `i` 参与。**调用线程自身必须在 mask 中；mask 中所有尚未退出的线程必须执行同一个 intrinsic，并传入相同 mask。** mask 不会自动把分支外的线程拉进来，也不会把缺失的源 lane 补成 0。

```cpp
// 前提：完整 warp 的 32 个线程都执行这里
bool valid = idx < n;                         // idx 为当前线程的数据索引
unsigned members = __ballot_sync(FULL_MASK, valid);
if (valid) {
    // members 在分支前确定，标记所有 valid 线程
    int leader = __ffs(members) - 1;          // 第一个参与 lane
    float first = __shfl_sync(members, value, leader);
    // first 为 leader 的 value；所有 members 线程都要执行上面的 shuffle
}
```

- `__activemask()`：返回**执行到此处时**的活跃 lane 掩码，不是屏障。独立线程调度下，它不能替代分支前的 `__ballot_sync` 来确定逻辑上的参与集合。
- `__ffs(mask)`：返回最低置位位的位置（从 1 开始）；`__popc(mask)`：返回置位数，可用来统计参与线程数。
- `FULL_MASK` 适用于完整 warp 全员执行同一操作的情形；稀疏 mask 的 lane 编号仍是物理编号，不会重新连续编号。

### 3. Shuffle：直接读取其他 lane 的寄存器

以下 `v` 是每个线程自己的值；`width` 可省略，默认 32，也可取 2、4、8、16，将 warp 分成连续等宽子组。

| 语法 | 含义 |
|---|---|
| `__shfl_sync(mask, v, srcLane, width)` | 读取所在子组内编号为 `srcLane` 的 lane 的值，常用于广播 |
| `__shfl_up_sync(mask, v, delta, width)` | 读取所在子组内编号比自己小 `delta` 的 lane 的值 |
| `__shfl_down_sync(mask, v, delta, width)` | 读取所在子组内编号比自己大 `delta` 的 lane 的值 |
| `__shfl_xor_sync(mask, v, laneMask, width)` | 读取所在子组内编号与自己异或 `laneMask` 的 lane 的值，常用于蝶形交换 |

```cpp
// 前提：完整 warp 全员执行，v 为 float 或支持的整数等标量类型
float broadcast = __shfl_sync(FULL_MASK, v, 0);  // 广播 lane 0 的 v
float neighbor  = __shfl_xor_sync(FULL_MASK, v, 1); // 0↔1、2↔3……

// 每 8 个 lane 一组，分别广播组内第 0 个 lane 的值
float subgroup = __shfl_sync(FULL_MASK, v, 0, 8);
```

源 lane 必须活跃且参与该次 shuffle，否则读到的值未定义。`up/down` 的源位置越过子组边界时返回调用者自己的值。**Shuffle 不提供共享内存或全局内存的读写顺序保证。**

### 4. 投票、匹配与硬件归约

| 语法 | 返回值 / 用途 | 硬件要求 |
|---|---|---|
| `__all_sync(mask, pred)` | 所有参与线程的 `pred` 都非零时返回非零 | 计算能力 5.0+ |
| `__any_sync(mask, pred)` | 任一参与线程的 `pred` 非零时返回非零 | 计算能力 5.0+ |
| `__ballot_sync(mask, pred)` | 返回参与线程中 `pred` 非零的 lane 位图 | 计算能力 5.0+ |
| `__match_any_sync(mask, value)` | 返回参与线程中与自己 `value` 相同的 lane 位图 | 计算能力 7.0+ |
| `__match_all_sync(mask, value, &pred)` | 全员值相同则返回 mask 且 `pred=1`，否则返回 0 且 `pred=0` | 计算能力 7.0+ |
| `__reduce_add_sync(mask, value)` | 参与线程的整数和，每个参与线程都获得结果 | 计算能力 8.0+ |
| `__reduce_min_sync` / `__reduce_max_sync` | 参与线程的整数最小值 / 最大值，参数同上 | 计算能力 8.0+ |
| `__reduce_and_sync` / `__reduce_or_sync` / `__reduce_xor_sync` | 参与线程的无符号整数按位归约，参数同上 | 计算能力 8.0+ |

硬件归约的 `add/min/max` 支持 32 位 `int` / `unsigned int`，不能直接传 `float`。浮点归约可用下面的 shuffle 写法。编译时需选择覆盖对应计算能力的目标架构。

### 5. 可复用示例：每个 warp 输出一个浮点和

尾部不足 32 个数据时，仍让完整 warp 执行归约，越界数据填 0。这样每次 shuffle 的源 lane 都存在。

```cpp
#include <cuda_runtime.h>

__device__ float warp_sum(float v) {
    constexpr unsigned FULL_MASK = 0xffffffffu;
    // 调用约定：完整 warp 的所有 32 个线程执行此函数
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(FULL_MASK, v, offset);
    return v;  // 仅 lane 0 持有整个 warp 的和
}

__global__ void sum_per_warp(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    float v = (idx < n) ? in[idx] : 0.0f;  // 不要提前 return
    float sum = warp_sum(v);
    if (lane == 0 && idx < n)
        out[idx / warpSize] = sum;
}

// host 端启动片段：d_in / d_out 已分配，n 为非负数
// d_out 至少容纳 (n + 31) / 32 个 float
int block = 256;                           // 一维 block，线程数必须是 32 的倍数
if (n > 0) {
    int grid = (n + block - 1) / block;
    sum_per_warp<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());        // 使用第五节的宏
    CUDA_CHECK(cudaDeviceSynchronize());
}
```

例如 `n = 35`，输出为前 32 个元素之和、最后 3 个元素之和。若需要整个数组的总和，还要继续归约这些输出。浮点加法顺序与 CPU 串行求和不同，结果可能有舍入差异。

### 6. Warp 同步与 Cooperative Groups

`__syncwarp(mask)` 是指定 warp 成员之间的屏障，并保证参与线程在屏障前的内存访问对屏障后的参与线程有序可见。以下示例要求完整 warp 全员执行：

```cpp
// kernel 内，使用第一小节的 tid / lane
// block 总线程数必须是 32 的倍数且不超过 256
__shared__ float scratch[256];
scratch[tid] = value;
__syncwarp(0xffffffffu);                    // 等待本 warp 的写入
float peer = scratch[tid ^ 1u];            // 读取同 warp 相邻 lane 的数据
__syncwarp(0xffffffffu);                    // 后续复用 scratch 前，等待本轮读取完成
```

跨 warp 交换共享内存数据需要 `__syncthreads()` 等 block 级同步。Volta 起支持独立线程调度，不应依赖“同 warp 一定隐式同步”的旧写法；`volatile` 也不能替代同步。

也可以用 Cooperative Groups 显式表达 32 线程子组：

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cg_warp_sum(const float *in, float *out, int n) {
    // 一维 block，线程数是 32 的倍数；整个 block 都必须执行分组操作
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (idx < n) ? in[idx] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        v += tile.shfl_down(v, offset);
    if (tile.thread_rank() == 0 && idx < n)
        out[idx / 32] = v;
    // tile.sync() 可用于该子组内需要内存同步的场景
}
```

普通 kernel 内的 `tiled_partition<32>` 不需要协作式 kernel 启动。分组与组内集合操作仍必须由相应组的所有成员共同执行。

**常见错误速查：**

- 在 `if (idx < n)` 内用 `FULL_MASK`，却让其他尚未退出的 lane 跳过同一次调用：违反参与约定。
- 对任意稀疏 mask 直接套用 `offset = 16, 8, 4, 2, 1` 的归约：可能读取不参与的源 lane；优先采用上面的完整 warp 补零方案。
- 认为 `__shfl_*_sync` 的 `_sync` 等价于内存屏障：共享内存通信仍需 `__syncwarp` 或更大范围的同步。
- 使用旧的 `__shfl`、`__ballot` 等无 `_sync` 接口：新代码应使用带 mask 的同步接口。

参考：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) 中的 Warp Shuffle、Warp Vote、Warp Match、Warp Reduce 和 Synchronization Functions 说明。

---

**延伸**：[分块矩阵乘法](../hardware_basics/tiled_matmul.md) ｜ [SGEMM 优化链](../hardware_basics/sgemm_optimize.md) ｜ [同步vs异步传输](../hardware_basics/sync_async_transfer.md) ｜ [Warp 级编程练习](../week4_advanced/day22_warp_programming.cu)
