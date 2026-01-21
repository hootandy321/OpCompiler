# DeepSeek V3 模型推理实现核心文档

DeepSeek V3 是一个混合专家模型(MoE)，实现了多层级 Transformer 架构，结合了密集层和稀疏层，支持多 GPU 并行推理。该模块提供了完整的模型权重加载、KV 缓存管理、批处理推理等功能。

## 1. 模块结构

- **`deepseek_v3_impl.hpp`**: 核心数据结构定义，包括权重结构、缓存结构、设备资源、推理状态等
- **`deepseek_v3.cpp`**: 主要推理实现，包含前向传播、MoE 计算、多设备并行调度逻辑
- **`deepseek_v3_weight.cpp`**: 权重加载和初始化，实现量化线性层权重管理、RoPE 位置编码表生成
- **`deepseek_v3_cache.cpp`**: KV 缓存的创建和销毁，管理多设备多层的 Key-Value 缓存

## 2. 核心数据结构

### `DeepSeekV3Model`
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: 完整的 DeepSeek V3 模型实例，管理多设备资源和推理线程
- **核心成员**:
  - `meta`: `DeepSeekV3Meta` - 模型元信息(维度、层数、专家数等超参数)
  - `dev_ids`: `std::vector<int>` - 设备 ID 列表
  - `dev_resources`: `std::vector<DeepSeekV3DeviceResource>` - 每个设备的计算资源
  - `states`: `std::vector<InferState>` - 每个设备的推理状态(线程同步)
  - `threads`: `std::vector<std::thread>` - 每个设备对应的推理线程
  - `req`: `InferRequest` - 当前批次推理请求
- **生命周期**:
  1. 构造函数初始化多设备资源和通信域
  2. 为每个设备创建推理线程并进入等待循环
  3. 接收推理请求后通过条件变量唤醒所有设备线程
  4. 销毁时设置退出标志并等待所有线程结束

### `QuantLinearWeight`
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: 量化线性层权重，支持 W8A8 量化(8-bit 权重，8-bit 激活)
- **核心成员**:
  - `w`: `std::shared_ptr<Tensor>` - 量化权重张量(int32)，形状为 [in_dim, out_dim/8]
  - `s`: `std::shared_ptr<Tensor>` - 量化尺度张量，形状为 [in_dim/64, out_dim]
  - `z`: `std::shared_ptr<Tensor>` - 零点偏移张量(int32)，形状为 [in_dim/64, out_dim/8]
- **量化策略**: 每 64 个输入通道共享一组尺度和零点，权重按 8 位打包存储

### `MLAWeight` (Multi-head Latent Attention)
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: MLA 注意力机制的权重，实现低秩键值压缩
- **核心成员**:
  - `q_a_norm`, `kv_a_norm`: RMS Layer Norm 权重
  - `q_a_proj`: Q 向量第一层投影 [d → r_q]
  - `q_b_proj`: Q 向量第二层投影 [r_q → nh/ndev * d_qk]
  - `kv_a_proj`: KV 向量压缩投影 [d → r_kv + d_rope]
  - `kv_b_proj`: K 向量解压投影 [r_kv → nh/ndev * (d_nope + d_v)]
  - `o_proj`: 输出投影 [nh/ndev * d_v → d]
- **设计理念**: 通过低秩分解减少 KV Cache 显存占用，将 K/V 从 [d_model] 压缩到 [r_kv]

### `GateWeight`
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: MoE 路由门网络权重
- **核心成员**:
  - `w`: 门控权重矩阵，形状 [nexperts, d] (转置存储，实际计算为 [d, nexperts])
  - `b`: 门控偏置向量，形状 [nexperts]

### `MLPWeight`
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: MLP 层的量化权重 (SwiGLU 激活函数)
- **核心成员**:
  - `gate`: 门控投影权重 [d → di]
  - `up`: 上投影权重 [d → di]
  - `down`: 下投影权重 [di → d]

### `LayerWeight`
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: 单个 Transformer 层的完整权重
- **核心成员**:
  - `mla_norm`: MLA 注意力前的 RMSNorm
  - `mla`: 注意力权重 (MLA 结构)
  - `mlp_norm`: MLP 前的 RMSNorm
  - `dense_mlp`: 密集层 MLP (仅前 n_dense_layer 层)
  - `route`: MoE 路由权重 (仅后 n_sparse_layer 层)
  - `share_expert`: 共享专家 MLP (仅后 n_sparse_layer 层)
  - `experts`: 路由专家列表 `std::vector<std::shared_ptr<MLPWeight>>` (仅后 n_sparse_layer 层)

### `DeepSeekV3DeviceWeights`
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: 单个设备的权重存储
- **核心成员**:
  - `w_in_embd`: 输入嵌入表 [dvoc, d]
  - `w_out_norm`: 输出层 RMSNorm [d]
  - `w_out_embd`: 输出嵌入表 [d, dvoc] (转置存储)
  - `sin_table`, `cos_table`: RoPE 位置编码表 [dctx, d_rope/2]
  - `w_layers`: 所有层的权重列表
  - `device`: 设备类型 (CPU/CUDA 等)
  - `dev_id`: 设备编号
  - `load_stream`: 权重加载流

### `DeepSeekV3Cache`
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: 单个序列的 KV 缓存，跨所有层和设备
- **核心成员**:
  - `kv_pass`: `std::vector<std::vector<std::shared_ptr<Tensor>>>` - 压缩后的 K 缓存 [ndev][nlayer][max_len, r_kv]
  - `k_rot`: `std::vector<std::vector<std::shared_ptr<Tensor>>>` - 旋转位置 K 缓存 [ndev][nlayer][max_len, d_rope]
- **存储优化**: MLA 架构将 K 压缩为 kv_pass (无位置信息) 和 k_rot (带位置信息)，大幅减少显存

### `DeepSeekV3DeviceResource`
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: 单个设备的运行时资源
- **核心成员**:
  - `device`, `device_id`: 设备信息
  - `handle`: `infiniopHandle_t` - InfiniCore 计算库句柄
  - `weights`: 该设备的权重指针
  - `stream`: 计算流
  - `comm`: `infinicclComm_t` - 多设备通信域
  - `memory_pool`: `std::shared_ptr<MemoryPool>` - 显存池

### `InferState`
- **位置**: `deepseek_v3_impl.hpp`
- **主要功能**: 单个设备推理线程的同步状态
- **核心成员**:
  - `mtx`: `std::mutex` - 互斥锁
  - `cv_load`, `cv_start`, `cv_done`: 条件变量
  - `loaded`: 是否完成初始化
  - `proceed`: 是否开始推理
  - `exit_flag`: 退出线程标志
- **同步机制**:
  - 主线程等待所有设备 `cv_load` 确保初始化完成
  - 主线程通过 `cv_start` 通知设备开始推理
  - 设备完成推理后通过 `cv_done` 通知主线程

## 3. 核心算法

### 3.1 推理主循环 (`inferDeviceBatch`)

**位置**: `deepseek_v3.cpp:51-432`

**功能**: 单个设备上的批处理前向传播

**关键步骤**:

1. **输入嵌入** (行 120-124):
   ```cpp
   // 从 w_in_embd 表中查表获取 token 嵌入
   // logits_in: [ntok, d]
   ```

2. **层迭代** (行 147-392):
   对每一层 (共 n_dense_layer + n_sparse_layer 层):

   **2.1 注意力部分** (行 148-227):
   - RMS Normalization (行 150)
   - Q 向量两阶段投影:
     - `dequant_linear(q_a_buf, logits_out, q_a_proj)` (行 152-156)
     - `rmsnorm(q_a_buf, q_a_norm)` (行 157)
     - `dequant_linear(q_buf, q_a_buf, q_b_proj)` (行 158-162)
   - RoPE 旋转位置编码 (行 163-164):
     ```cpp
     auto q_rot = q_buf->view({ntok, nh, d_qk})->slice(2, d_nope, d_rope);
     rope_v2(q_rot, q_rot, pos_ids_buf, sin_table, cos_table);
     ```
   - KV 压缩投影 (行 166-174):
     ```cpp
     dequant_linear(kv_a_buf, logits_out, kv_a_proj); // [ntok, r_kv + d_rope]
     auto kv_pass = kv_a_buf->slice(1, 0, r_kv);       // 压缩 K (无位置)
     rmsnorm(kv_pass, kv_a_norm);
     auto k_rot = kv_a_buf->slice(1, r_kv, d_rope);    // 旋转 K (带位置)
     rope_v2(k_rot, k_rot, pos_ids_buf, sin_table, cos_table);
     ```
   - **逐请求处理** (行 176-220):
     - 更新 KV 缓存:
       ```cpp
       rearrange(caches[req]->kv_pass[idev][layer]->slice(0, past_len, seq_len), kv_pass_req);
       rearrange(caches[req]->k_rot[idev][layer]->slice(0, past_len, seq_len), k_rot_req);
       ```
     - KV 解压投影:
       ```cpp
       dequant_linear(kv_b_req, cached_kv_pass, kv_b_proj);
       ```
     - K 重组 (无位置部分 + 旋转位置部分):
       ```cpp
       rearrange(full_k_pass_req, kv_b_req->slice(1, 0, nh * d_nope));
       rearrange(full_k_rot_req, k_rot_req->view_as({total_len, nh, d_rope}, {d_rope, 0, 1}));
       ```
     - 计算注意力分数:
       ```cpp
       linear(attn_score_req, q_req->permute({1, 0, 2}), full_k_req->permute({1, 2, 0}),
              1.f / sqrt(d_qk), 0.f);
       ```
     - 因果 Softmax:
       ```cpp
       causalSoftmax(attn_score_req, attn_score_req);
       ```
     - 注意力加权:
       ```cpp
       linear(attn_val_req, attn_score_req, full_v_req->permute({1, 0, 2}), 1.f, 0.f);
       ```
   - 输出投影 (行 223-227):
     ```cpp
     dequant_linear(logits_in, o_buf, o_proj->w, o_proj->s, o_proj->z,
                    1.0, 0.0, idev == 0 ? logits_in : nullptr);
     // 仅 rank 0 添加残差连接
     ```
   - 多设备 AllReduce (行 230-235):
     ```cpp
     infinicclAllReduce(logits_in, logits_in, ntok * d, INFINICCL_SUM, comm, stream);
     ```

   **2.2 MLP 部分** (行 236-391):
   - RMS Normalization (行 237)

   - **密集层** (行 239-255, 仅前 n_dense_layer 层):
     ```cpp
     dequant_linear(gate_dense, logits_out, dense_mlp->gate);
     dequant_linear(up_dense, logits_out, dense_mlp->up);
     swiglu(gate_dense, up_dense, gate_dense); // SiLU 激活 + 逐元素乘
     dequant_linear(logits_in, gate_dense, dense_mlp->down,
                    1.0, 0.0, idev == 0 ? logits_in : nullptr);
     ```

   - **稀疏 MoE 层** (行 256-383, 仅后 n_sparse_layer 层):

     **步骤 1: 共享专家计算** (行 288-304):
     ```cpp
     dequant_linear(moe_gate_buf, hidden_states, share_expert->gate);
     dequant_linear(moe_up_buf, hidden_states, share_expert->up);
     swiglu(moe_gate_buf, moe_up_buf, moe_gate_buf);
     dequant_linear(shared_states, moe_gate_buf, share_expert->down);
     ```

     **步骤 2: Top-K 路由** (行 307-317):
     ```cpp
     // 计算每个 token 对所有专家的亲和度分数
     gemm(router_logits, hidden_states, route->w, 1.0, 0.0);

     // 选择 Top-8 专家并归一化权重
     topkrouter(values_gpu, indices_gpu, router_logits, route->b,
                routed_scaling_factor, topk); // topk=8

     // 拷贝到 CPU 用于后续循环
     infinirtMemcpy(values_cpu.data(), values_gpu->data(), ...);
     infinirtMemcpy(indices_cpu.data(), indices_gpu->data(), ...);
     ```

     **步骤 3: 路由专家计算** (行 319-374):
     - **逐 Token 循环** (行 322):
       - 第一个专家: `C = alpha * Expert(x)` (行 330-351)
       - 后续专家: `C += alpha * Expert(x)` (行 354-373)
     ```cpp
     for (size_t itok = 0; itok < ntok; ++itok) {
         for (size_t k = 0; k < topk; ++k) {
             int index = indices_cpu[itok * topk + k]; // 专家 ID
             float alpha = values_cpu[itok * topk + k]; // 加权系数

             dequant_linear(moe_gate_buf_i, hidden_states_i, experts[index]->gate);
             dequant_linear(moe_up_buf_i, hidden_states_i, experts[index]->up);
             swiglu(moe_gate_buf_i, moe_up_buf_i, moe_gate_buf_i);

             if (k == 0)
                 dequant_linear(router_states_sum_i, moe_gate_buf_i, experts[index]->down,
                                alpha, 0.0, nullptr, nullptr);
             else
                 dequant_linear(router_states_sum_i, moe_gate_buf_i, experts[index]->down,
                                alpha, 0.0, router_states_sum_i, nullptr);
         }
     }
     ```

     **步骤 4: 合并输出** (行 377-382):
     ```cpp
     add(shared_states, shared_states, router_states_sum); // 共享 + 路由
     add(logits_in, shared_states, logits_in);             // 残差连接
     ```

   - 多设备 AllReduce (行 386-391)

3. **输出采样** (行 394-431, 仅 rank 0):
   - RMS Norm + 输出投影:
     ```cpp
     rmsnorm(logits_out, logits_in, w_out_norm, epsilon);
     linear(last_logits_buf, logits_out, w_out_embd, 1.0, 0.0);
     ```
   - 随机采样 (行 416-422):
     ```cpp
     float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
     randomSample(result_buf, prob_buf, random_val, topp, topk, temperature);
     ```

**算法复杂度**:
- 注意力: O(ntok² * d_qk * nh) (序列长度平方)
- MoE 路由: O(ntok * d * nexperts)
- MoE 计算: O(ntok * topk * di * d) ≈ O(ntok * 8 * di * d)

### 3.2 量化线性层反量化 (`dequant_linear`)

**原理**: W8A8 量化，公式: `output = (input @ (weight * scale + zero_point).T) * alpha + beta`

**参数**:
- `input`: 输入张量 [M, K]
- `w`: 量化权重 [K, N/8] (int32, 8 个值打包为一个 int32)
- `s`: 尺度 [K/64, N] (float)
- `z`: 零点 [K/64, N/8] (int32)
- `alpha`, `beta`: 输出缩放和偏移
- `bias`: 可选偏置 (本代码中未使用)

**实现**: 由 InfiniCore 库提供内核优化

### 3.3 RoPE v2 位置编码 (`rope_v2`)

**位置**: `deepseek_v3.cpp:164, 174`

**公式**:
```
对于位置 p 和维度 i (i ∈ [0, d_rope/2)):
  θ_i = 1 / (rope_theta ^ (2i / d_rope))
  rot_pos = p * θ_i

  q[2i]   = q[2i]   * cos(rot_pos) - q[2i+1] * sin(rot_pos)
  q[2i+1] = q[2i]   * sin(rot_pos) + q[2i+1] * cos(rot_pos)
```

**实现细节**:
- 预计算 sin_table 和 cos_table: [dctx, d_rope/2]
- 支持半精度 (FP16/BF16) 和单精度 (FP32)
- 仅对部分维度应用 RoPE (d_nope 到 d_rope)

### 3.4 Top-K 路由 (`topkrouter`)

**位置**: `deepseek_v3.cpp:314`

**输入**:
- `router_logits`: [ntok, nexperts] - 原始门控分数
- `gate_bias`: [nexperts] - 门控偏置
- `routed_scaling_factor`: 缩放因子 (2.5)
- `topk`: 选择的专家数量 (8)

**输出**:
- `values_gpu`: [ntok * topk] - 归一化权重 (softmax 后)
- `indices_gpu`: [ntok * topk] - 专家 ID

**算法**:
1. `logits = hidden_states @ gate_weight.T + gate_bias`
2. `scaled_logits = logits * routed_scaling_factor`
3. 对每个 token，选择分数最高的 topk 个专家
4. 对选中的专家分数应用 Softmax 归一化
5. 返回归一化权重和专家索引

## 4. API 接口

### 4.1 模型创建

```cpp
// 创建模型实例
__C struct DeepSeekV3Model *
createDeepSeekV3Model(const DeepSeekV3Meta *meta,
                      const DeepSeekV3Weights *weights);

// DeepSeekV3Meta 结构
struct DeepSeekV3Meta {
    size_t dvoc;           // 词汇表大小
    size_t d;              // 隐藏层维度
    size_t nh;             // 注意力头数
    size_t d_qk;           // 每个头的 Q/K 维度
    size_t d_v;            // 每个头的 V 维度
    size_t d_rope;         // RoPE 旋转维度
    size_t d_nope;         // 不旋转的维度
    size_t r_q;            // Q 压缩维度
    size_t r_kv;           // KV 压缩维度
    size_t n_dense_layer;  // 密集层数量
    size_t n_sparse_layer; // 稀疏层数量
    size_t di;             // 密集 MLP 中间维度
    size_t di_moe;         // MoE MLP 中间维度
    size_t nexperts;       // 专家总数
    size_t kexperts;       // 激活专家数 (topk)
    size_t dctx;           // 最大上下文长度
    float rope_theta;      // RoPE 基数
    float epsilon;         // RMSNorm epsilon
    float routed_scale;    // MoE 路由缩放
    infiniDtype_t dt_logits;      // 激活值数据类型
    infiniDtype_t dt_norm;        // Norm 层数据类型
    infiniDtype_t dt_quant_scale; // 量化尺度数据类型
    infiniDtype_t dt_gate_weight; // 门控权重数据类型
    infiniDtype_t dt_gate_bias;   // 门控偏置数据类型
};
```

**功能**: 初始化多设备推理模型，创建推理线程池

### 4.2 权重创建

```cpp
// 创建权重结构
__C DeepSeekV3Weights *
createDeepSeekV3Weights(const DeepSeekV3Meta *meta,
                        infiniDevice_t device,
                        int ndev,
                        const int *dev_ids);

// 获取权重加载器
__C DeepSeekV3WeightLoader *
createDeepSeekV3WeightLoader();

// 权重加载器接口
struct DeepSeekV3WeightLoader {
    void (*load_input_embd)(DeepSeekV3Weights *, void *cpu_ptr);
    void (*load_output_norm)(DeepSeekV3Weights *, void *cpu_ptr);
    void (*load_output_embd)(DeepSeekV3Weights *, void *cpu_ptr);
    void (*load_attn_norm)(DeepSeekV3Weights *, void *cpu_ptr, size_t layer);
    void (*load_attn_q_a_proj)(DeepSeekV3Weights *, void *w, void *s, void *z, size_t layer);
    void (*load_attn_q_a_layernorm)(DeepSeekV3Weights *, void *cpu_ptr, size_t layer);
    void (*load_attn_q_b_proj)(DeepSeekV3Weights *, void *w, void *s, void *z, size_t layer);
    void (*load_attn_kv_a_proj_with_mqa)(DeepSeekV3Weights *, void *w, void *s, void *z, size_t layer);
    void (*load_attn_kv_a_layernorm)(DeepSeekV3Weights *, void *cpu_ptr, size_t layer);
    void (*load_attn_kv_b_proj)(DeepSeekV3Weights *, void *w, void *s, void *z, size_t layer);
    void (*load_attn_o_proj)(DeepSeekV3Weights *, void *w, void *s, void *z, size_t layer);
    void (*load_mlp_norm)(DeepSeekV3Weights *, void *cpu_ptr, size_t layer);
    void (*load_mlp_dense)(DeepSeekV3Weights *, void *gate_w, void *gate_s, void *gate_z,
                          void *up_w, void *up_s, void *up_z,
                          void *down_w, void *down_s, void *down_z, size_t layer);
    void (*load_mlp_gate_weight)(DeepSeekV3Weights *, void *cpu_ptr, size_t layer);
    void (*load_mlp_gate_bias)(DeepSeekV3Weights *, void *cpu_ptr, size_t layer);
    void (*load_mlp_shared_experts)(DeepSeekV3Weights *, ...);
    void (*load_mlp_experts)(DeepSeekV3Weights *, ..., size_t layer, size_t expert_id);
};
```

**功能**: 分配 GPU 显存，提供回调函数从 CPU 加载权重

**多设备分发策略**:
- 全局权重 (嵌入表、Norm): 复制到所有设备
- 注意力投影: 按头数切分 (每个设备 `nh/ndev` 个头)
- MLP 权重: 按中间维度切分 (`di/ndev` 或 `di_moe/ndev`)

### 4.3 缓存创建

```cpp
// 创建 KV 缓存
__C struct DeepSeekV3Cache *
createDeepSeekV3Cache(const struct DeepSeekV3Model *model);

// 销毁 KV 缓存
__C void
dropDeepSeekV3Cache(const struct DeepSeekV3Model *model,
                    struct DeepSeekV3Cache *cache);
```

**功能**: 为单个序列分配 KV 缓存张量

**内存布局**:
- `kv_pass[idev][layer]`: [max_len, r_kv] - 压缩 K 缓存
- `k_rot[idev][layer]`: [max_len, d_rope] - 旋转 K 缓存

### 4.4 推理接口

```cpp
// 批量推理并采样
__C void
inferBatchDeepSeekV3(struct DeepSeekV3Model *model,
                     const uint32_t *tokens,   // 输入 token 序列 [ntok]
                     uint32_t ntok,            // token 总数
                     const uint32_t *req_lens, // 每个请求的长度 [nreq]
                     uint32_t nreq,            // 请求总数
                     const uint32_t *req_pos,  // 每个请求的起始位置 [nreq]
                     struct DeepSeekV3Cache **kv_caches, // KV 缓存指针数组 [nreq]
                     const float *temperature, // 采样温度 [nreq]
                     const uint32_t *topk,     // Top-K 采样 [nreq]
                     const float *topp,        // Top-P (nucleus) 采样 [nreq]
                     uint32_t *output);        // 输出 token [nreq]

// 批量推理并返回 logits
__C void
forwardBatchDeepSeekV3(struct DeepSeekV3Model *model,
                       const uint32_t *tokens, uint32_t ntok,
                       const uint32_t *req_lens, uint32_t nreq,
                       const uint32_t *req_pos,
                       struct DeepSeekV3Cache **kv_caches,
                       void *logits); // 输出 logits [nreq, dvoc]
```

**功能**: 多请求批处理推理，支持不同长度的序列

**调用流程**:
1. 主线程调用 `inferBatchDeepSeekV3`
2. 设置 `model->req` 并通过条件变量唤醒所有设备线程
3. 设备线程并行执行 `inferDeviceBatch`
4. 主线程等待所有设备完成 (逆序等待，确保 rank 0 最后完成)
5. 返回采样结果

### 4.5 模型销毁

```cpp
__C void
destroyDeepSeekV3Model(struct DeepSeekV3Model *model);
```

**功能**: 优雅退出，通知所有推理线程结束并回收资源

## 5. 使用示例

```cpp
#include "infinicore_infer.h"

// 1. 定义模型超参数
DeepSeekV3Meta meta = {
    .dvoc = 102400,
    .d = 7168,
    .nh = 128,
    .d_qk = 64,
    .d_v = 128,
    .d_rope = 64,
    .d_nope = 32,
    .r_q = 1536,
    .r_kv = 512,
    .n_dense_layer = 24,
    .n_sparse_layer = 37,
    .di = 18432,
    .di_moe = 7168,
    .nexperts = 256,
    .kexperts = 8,
    .dctx = 8192,
    .rope_theta = 10000.0f,
    .epsilon = 1e-6f,
    .routed_scale = 2.5f,
    .dt_logits = INFINI_DTYPE_F16,
    .dt_norm = INFINI_DTYPE_F16,
    .dt_quant_scale = INFINI_DTYPE_F16,
    .dt_gate_weight = INFINI_DTYPE_F16,
    .dt_gate_bias = INFINI_DTYPE_F16,
};

// 2. 初始化权重 (假设从文件加载)
int ndev = 4;
int dev_ids[4] = {0, 1, 2, 3};
auto weights = createDeepSeekV3Weights(&meta, INFINI_DEVICE_CUDA, ndev, dev_ids);

auto loader = createDeepSeekV3WeightLoader();

// 加载嵌入表
loader->load_input_embd(weights, cpu_input_embd_ptr);
loader->load_output_norm(weights, cpu_output_norm_ptr);
loader->load_output_embd(weights, cpu_output_embd_ptr);

// 加载层权重
for (size_t layer = 0; layer < meta.n_dense_layer + meta.n_sparse_layer; layer++) {
    loader->load_attn_norm(weights, cpu_attn_norm_ptr, layer);
    loader->load_attn_q_a_proj(weights, cpu_q_w, cpu_q_s, cpu_q_z, layer);
    loader->load_attn_q_a_layernorm(weights, cpu_q_norm_ptr, layer);
    loader->load_attn_q_b_proj(weights, cpu_qb_w, cpu_qb_s, cpu_qb_z, layer);
    loader->load_attn_kv_a_proj_with_mqa(weights, cpu_kv_w, cpu_kv_s, cpu_kv_z, layer);
    loader->load_attn_kv_a_layernorm(weights, cpu_kv_norm_ptr, layer);
    loader->load_attn_kv_b_proj(weights, cpu_kvb_w, cpu_kvb_s, cpu_kvb_z, layer);
    loader->load_attn_o_proj(weights, cpu_o_w, cpu_o_s, cpu_o_z, layer);
    loader->load_mlp_norm(weights, cpu_mlp_norm_ptr, layer);

    if (layer < meta.n_dense_layer) {
        // 密集 MLP
        loader->load_mlp_dense(weights, gate_w, gate_s, gate_z,
                               up_w, up_s, up_z,
                               down_w, down_s, down_z, layer);
    } else {
        // 稀疏 MoE
        loader->load_mlp_gate_weight(weights, cpu_gate_w_ptr, layer);
        loader->load_mlp_gate_bias(weights, cpu_gate_b_ptr, layer);
        loader->load_mlp_shared_experts(weights, s_gate_w, s_gate_s, s_gate_z,
                                       s_up_w, s_up_s, s_up_z,
                                       s_down_w, s_down_s, s_down_z, layer);
        for (size_t expert = 0; expert < meta.nexperts; expert++) {
            loader->load_mlp_experts(weights, e_gate_w, e_gate_s, e_gate_z,
                                    e_up_w, e_up_s, e_up_z,
                                    e_down_w, e_down_s, e_down_z,
                                    layer, expert);
        }
    }
}

// 3. 创建模型
auto model = createDeepSeekV3Model(&meta, weights);

// 4. 创建 KV 缓存 (假设 2 个请求)
DeepSeekV3Cache *caches[2];
caches[0] = createDeepSeekV3Cache(model);
caches[1] = createDeepSeekV3Cache(model);

// 5. 准备输入 (2 个请求，第一个长度 10，第二个长度 5)
uint32_t tokens[15] = { /* token IDs */ };
uint32_t req_lens[2] = {10, 5};
uint32_t req_pos[2] = {0, 0}; // 都是新生成的请求
float temperature[2] = {0.7f, 0.8f};
uint32_t topk[2] = {50, 50};
float topp[2] = {0.9f, 0.95f};
uint32_t output[2];

// 6. 推理
inferBatchDeepSeekV3(model, tokens, 15, req_lens, 2, req_pos, caches,
                     temperature, topk, topp, output);

printf("Generated tokens: %u, %u\n", output[0], output[1]);

// 7. 清理
dropDeepSeekV3Cache(model, caches[0]);
dropDeepSeekV3Cache(model, caches[1]);
destroyDeepSeekV3Model(model);
```

## 6. 实现细节

### 6.1 内存管理

**显存池 (MemoryPool)**:
- 每个设备维护独立的显存池
- 推理过程中所有中间张量从池中分配
- 避免频繁的 `malloc/free` 操作，提升性能

**KV 缓存优化**:
- MLA 架构将 KV Cache 从 [max_len, nh, d_k] 压缩到 [max_len, r_kv]
- 对于 DeepSeek V3 (d_model=7168, r_kv=512)，压缩比约 14 倍
- 支持 PagedAttention 风格的分块缓存 (本代码中未实现)

**量化显存节省**:
- W8A8 量化将权重显存减少 4 倍 (FP32 → INT8)
- 每 64 个输入通道共享一组量化参数，平衡精度和效率

### 6.2 并行策略

**数据并行**:
- 不同设备处理相同的请求
- 通过 AllReduce 在每层后同步梯度 (推理时同步激活值)
- 适用于大批次场景

**张量并行** (当前实现):
- 注意力头按设备切分: `nh_per_dev = nh / ndev`
- MLP 中间维度按设备切分: `di_per_dev = di / ndev`
- 每层后通过 AllReduce 聚合部分和

**通信优化**:
- 使用独立流进行权重加载和计算，隐藏 PCIe 传输延迟
- AllReduce 使用 NCCL (InfiniCCL)，支持 Ring/Tree 算法

### 6.3 性能优化

**融合内核**:
- `dequant_linear`: 反量化 + 矩阵乘法融合
- `rmsnorm`: RMS 归一化专用内核
- `swiglu`: SiLU 激活 + 逐元素乘融合
- `rope_v2`: RoPE 位置编码内核

**批处理优化**:
- 多请求合并为单个批次，提升 GPU 利用率
- 动态请求长度，支持变长序列

**缓存友好**:
- 权重按层顺序加载到 GPU，利用缓存局部性
- 中间激活值复用缓冲区，减少显存占用

### 6.4 错误处理

**宏 `RUN_INFINI`**:
```cpp
#define RUN_INFINI(expr) \
    do { \
        auto _err = (expr); \
        if (_err != INFINI_STATUS_SUCCESS) { \
            std::cerr << "Error: " << #expr << " failed with code " << _err << std::endl; \
            exit(1); \
        } \
    } while (0)
```

**类型安全**:
- 使用 `__C` 宏导出 C 兼容接口
- C++ 实现细节隐藏在内部结构中

### 6.5 线程同步

**推理线程生命周期**:
```cpp
// 启动阶段 (DeepSeekV3Model 构造函数)
for (int i = 0; i < ndev; i++) {
    threads[i] = std::thread(launchDevice, ...);
}
// 等待所有设备初始化完成
for (int i = 0; i < ndev; i++) {
    states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
}

// 推理循环 (launchDevice 函数)
while (true) {
    cv_start.wait(lock, [&] { return proceed || exit_flag; });
    if (exit_flag) break;
    inferDeviceBatch(...);
    proceed = false;
    cv_done.notify_one();
}
```

**主线程调度**:
```cpp
// inferBatchDeepSeekV3 / forwardBatchDeepSeekV3
for (size_t idev = 0; idev < ndev; idev++) {
    states[idev].proceed = true;
    states[idev].cv_start.notify_one();
}
// 逆序等待 (确保 rank 0 最后完成，避免数据竞争)
for (size_t i = ndev; i > 0; i--) {
    auto idev = i - 1;
    states[idev].cv_done.wait(lock, [&] { return !states[idev].proceed });
}
```

### 6.6 RoPE 表生成

**预计算策略** (`getSinTable`, `getCosTable`):
- 在模型初始化时一次性计算所有位置的 sin/cos 值
- 支持 FP16/BF16/FP32 格式
- 公式: `θ_i = 1 / (rope_theta ^ (2i / d_rope))`

**数据转换**:
```cpp
if (dt_logits == INFINI_DTYPE_F16) {
    ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
} else if (dt_logits == INFINI_DTYPE_BF16) {
    ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
}
```

## 7. 依赖关系

**外部依赖**:
- `infinicore_infer.h`: InfiniCore 推理库头文件
  - `infiniopHandle_t`: 计算库句柄
  - `infinirtStream_t`: CUDA/设备流
  - `infinicclComm_t`: 通信域
  - `RUN_INFINI`: 错误处理宏

**内部依赖**:
- `../../tensor.hpp`: 张量抽象层
  - `Tensor::buffer`: 分配计算缓冲区
  - `Tensor::weight`: 创建权重张量
  - `dequant_linear`, `rmsnorm`, `swiglu`, `rope_v2` 等算子
- `../../allocator.hpp`: 显存池管理
  - `MemoryPool`: 显存分配器
- `../../utils.hpp`: 工具函数
  - `dsize`: 数据类型字节数
  - `f32_to_f16`, `f32_to_bf16`: 浮点转换
- `../inference_context.hpp`: 推理上下文
  - `InferenceContext`: 算子执行的上下文
  - `CacheManager`: 缓存管理器

## 8. 设计模式

### 8.1 工厂模式
- `createDeepSeekV3Model`, `createDeepSeekV3Weights`, `createDeepSeekV3Cache`
- 封装复杂的初始化逻辑，提供简洁的 API

### 8.2 线程池模式
- 每个设备对应一个推理线程
- 主线程通过条件变量调度工作
- 线程生命周期与模型绑定

### 8.3 策略模式
- `DeepSeekV3WeightLoader` 函数表
- 允许调用方自定义权重加载流程
- 支持增量加载和延迟加载

### 8.4 RAII (资源获取即初始化)
- `DeepSeekV3DeviceResource` 使用智能指针管理权重
- 析构函数自动释放 CUDA 流、句柄、通信域

## 9. 扩展性

**支持新硬件**:
- 通过 `infiniDevice_t` 抽象，支持 CPU、CUDA、ROCm 等
- 量化内核由 InfiniCore 库提供，后端无关

**支持新量化格式**:
- 修改 `QuantLinearWeight` 结构
- 更新 `dequant_linear` 算子内核

**支持新注意力机制**:
- 修改 `MLAWeight` 结构
- 更新 `inferDeviceBatch` 中的注意力计算逻辑

**支持更大的批次**:
- 调整 `MemoryPool` 大小
- 优化中间缓冲区分配策略
