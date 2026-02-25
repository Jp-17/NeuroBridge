# proposal.md 深度审查报告

**审查日期**: 2026-02-25
**审查对象**: `cc_core_files/proposal.md` — NeuroBridge 研究提案
**参照文档**: `cc_core_files/code_research.md` + torch_brain 源代码直接验证
**审查者**: Claude Code (Opus 4.6)

---

## 总评

proposal.md 是一份结构完整、论述清晰的研究提案，研究空白的识别准确，技术路线设计有充分的理论支撑。特别是 Pre-PerceiverIO masking 的设计考量、不使用 prompt token 的理由、以及 Output Cross-Attention Readout 的表征提取方案，这三处设计备注的论证质量很高。

然而，提案在以下几个方面存在与实际代码/数据不匹配或技术上需要重新审视的问题：

---

## 第一部分：与代码实现不匹配的问题

### 1.1 "Spike-level tokenization" 与 TVSD MUA 数据的根本性矛盾

**提案声明 (多处)**:
> "spike-level tokenization + PerceiverIO 压缩架构"
> "每个 spike event 成为一个独立的 token"

**实际问题**:

TVSD 提供的核心数据是 **MUA (Multi-Unit Activity)**，即多个神经元的混合连续信号，而非经过 spike sorting 的离散 spike events。proposal.md 在第七节数据集描述中有一句提到"采集 MUA 信号"，但在整个方法设计（第五节）中仍然以 spike-level tokenization 为核心叙述框架，两者之间存在未解决的矛盾。

**深层分析**:

torch_brain 代码库中对这两种数据的处理方式截然不同：

| 特征 | POYO (spike data) | CaPOYO (calcium/continuous data) |
|------|-------------------|--------------------------------|
| 输入构造 | unit_emb + token_type_emb | input_value_map(value) || unit_emb |
| token 数量 | = spike 数量（稀疏） | = T × N（密集） |
| 时间戳 | 每个 spike 的精确时间 | 每个时间点重复 N 次 |
| 值编码 | 隐式（有 spike = 1，没有 = 不存在） | 显式 `nn.Linear(1, dim//2)` |

对于 TVSD MUA 数据，CaPOYO-style 的 tokenization 更为自然。但关键问题是 **token 数量**：
- 如果 MUA 以 1kHz 采样、1024 电极、1 秒窗口 → 1,024,000 tokens
- 这完全超出了 cross-attention 的处理能力（PerceiverIO 的 cross-attention 对 key/value 序列长度是 O(N) 的，128 个 latent tokens × 100 万个 input tokens = 1.28 亿次乘法，每个 batch 样本）

**建议**: proposal 应明确定义一个 **MUA 预处理策略**，将其降低到可处理的 token 数量。可能的方案：

1. **时间降采样 + 事件化**: 将 MUA 在时间上 bin 到 10-50ms（得到 20-100 个时间步），然后为每个 time_bin × electrode 创建一个 token (值 = bin 内平均 MUA)。1 秒 / 50ms × 1024 = 20,480 tokens，这在 PerceiverIO 的处理范围内。
2. **阈值化提取**: 对 MUA 设定阈值（如 2σ），将超阈值的时间点视为"spike event"，转化为 POYO-style 的离散 token。这保持了 spike-level tokenization 的叙事框架，但损失了亚阈值活动信息。
3. **混合方案**: 对 Allen/IBL (有 spike-sorted data) 使用 POYO-style，对 TVSD (MUA) 使用 CaPOYO-style。两者共享相同的 PerceiverIO + Backbone，只在 tokenization 前端不同。这实际上已经在 code_research.md 中有类似建议（plan.md 决策 1）。

### 1.2 模型规模设计与实际可用资源的脱节

**提案声明 (第 5.4 节)**:
> "参数配置：12 层，d_model=512，8 heads，FlashAttention 2"

**提案声明 (第 6.1 节)**:
> "预训练阶段：4-8× NVIDIA A100 (80GB)"

**实际情况** (已在 plan.md 中识别):
- 当前 GPU: RTX 4090 D 24GB
- dim=512, depth=12 的模型参数量约 100M+，在 24GB 显存上即使用 AMP 也无法训练合理的 batch size

**深层分析**:

基于 code_research_review 中的参数量计算方法，我对几个配置做精确估算：

| 配置 | dim | depth | heads | dim_head | 估算参数量 | 24GB VRAM 可行? |
|------|-----|-------|-------|----------|----------|---------------|
| POYO 1.3M | 64 | 6 | 8 | 64 | ~1.3M | 完全可行 |
| POYO 11.8M | 128 | 24 | 8 | 64 | ~11.8M | 可行(小batch) |
| proposal 方案 | 512 | 12 | 8 | 64 | ~100M+ | **不可行** |
| plan.md 调整 | 128 | 6 | 4 | 64 | ~5M | 可行 |

proposal 中的"12 层 d_model=512"方案在当前硬件上完全不可行。plan.md 已经做了合理的降级（dim=128, depth=6），但 proposal 本身应该标注这个计算资源限制，或者将 dim=512 标注为"最终目标配置"而非"默认配置"。

### 1.3 Poisson NLL Loss 与 MUA 连续信号的不兼容

**提案声明 (第 5.6 节)**:
> "Loss设计：在 masked grid 位置计算 Poisson NLL loss"
> "λ_ij 是 predicted firing rate, k_ij 是 observed spike count"

**实际问题**:

Poisson NLL 假设目标变量是**非负整数（计数数据）**。这对 spike counts 是合理的（一个 10ms bin 内的 spike 数量是 0, 1, 2, ...），但对 MUA 连续信号**不适用**：
- MUA 是连续实值信号，经过带通滤波和整流
- MUA 值可以是任何非负实数，不是整数
- MUA 的统计分布更接近 Gaussian 或 Gamma 分布，而非 Poisson

torch_brain 代码中已有 `MSELoss` 的实现 (`torch_brain/nn/loss.py:26-53`)，可以直接用于 MUA 重建。

**建议**:
- 对于 Allen/IBL spike-sorted 数据：Poisson NLL（或 Poisson 化的 MSE / Gaussian NLL）
- 对于 TVSD MUA 数据：MSE Loss 或 Gaussian NLL
- 在 proposal 中明确这个区分，并论证为什么需要不同的 loss（或统一使用 Gaussian NLL 以简化）

### 1.4 Target Sequence Decoder 的 Grid 时间分辨率问题

**提案声明 (第 5.6 节)**:
> "time_bin_ms = 10ms"
> "预定义一个 (N_neurons × T_bins) 的 2D query grid"

**计算量分析**:

对于一个典型的 Allen session（1 秒窗口，200 个神经元，10ms bins）：
- Grid 大小: 200 × 100 = 20,000 个 query positions
- 每个 query 通过 cross-attention 从 128 个 latent tokens 读取信息
- Cross-attention 计算: 20,000 × 128 × dim = 巨大

对于 TVSD（1 秒窗口，1024 电极，10ms bins）：
- Grid 大小: 1024 × 100 = **102,400** 个 query positions
- 这甚至比原始 MUA token 数量还要多

**深层分析**:

plan.md 已经将 grid 时间分辨率从 10ms 调整到 50ms（决策 4），这是合理的。但即使 50ms，TVSD 的 grid 仍然是 1024 × 20 = 20,480 个 query。

更根本的问题是：**Target Sequence Decoder 的 grid query 数量应该与 masking 粒度和数据类型相匹配**，而非一刀切。建议：

| 数据类型 | Grid 时间分辨率 | 典型 grid 大小 | 合理性 |
|---------|--------------|--------------|-------|
| Allen spike data (200 units, 1s) | 50ms | 200 × 20 = 4,000 | 可行 |
| TVSD MUA (1024 electrodes, 1s) | 100ms | 1024 × 10 = 10,240 | 勉强可行 |
| TVSD MUA (256 electrodes子集, 1s) | 50ms | 256 × 20 = 5,120 | 可行 |

建议对 TVSD 使用 100ms 的 grid bin，或者先在电极维度上做 subsampling/grouping。

---

## 第二部分：技术方案的深层审视

### 2.1 Pre-PerceiverIO Masking 的"空 chunk"问题需要更强的解决方案

**提案声明 (创新点 1 设计备注)**:
> "应对策略：(1) 设计 masking 窗口不与 chunk 边界对齐；(2) 使用 25-40ms 的 masking 窗口粒度"

**深层分析**:

这个应对策略的假设是：只要 masking 窗口不完全覆盖一个 chunk（125ms），每个 chunk 都至少有部分 spike tokens 供 cross-attention 使用。但这个假设有两个薄弱环节：

1. **低发放率神经元**: 如果某个时间窗口（125ms chunk）内只有 1-2 个 spikes，即使 mask 掉其中的 30ms 就可能移除所有 spikes，造成"空 chunk"。这在真实数据中很常见——V1 神经元的自发发放率可以低至 0.5-2 Hz。

2. **Neuron masking 模式下的风险**: Neuron masking dropout 30-50% 的神经元。如果一个 chunk 内原本就只有少数活跃神经元（如 5 个），dropout 3 个后只剩 2 个，cross-attention 的信息源极度有限。

**更强的解决方案建议**:

```
方案 A: 空 chunk 检测 + 回退
- 在 masking 后检查每个 chunk 的残余 token 数量
- 如果 < min_tokens_per_chunk (如 4)，随机恢复部分被 mask 的 tokens
- 简单有效，但引入了 masking 比例的不确定性

方案 B: Learnable default latent
- 当某个 chunk 没有 input tokens 时，cross-attention 输出退化为初始化值
- 可以为这种情况设计一个 learnable "default latent"
- 本质上相当于告诉模型"这个时间段没有观测数据"

方案 C: 全局注意力补偿
- 在 PerceiverIO encoder 之后增加一层全局 cross-attention
- 让所有 latent tokens（包括空 chunk 的）再次 attend to 全部 non-masked input tokens（不受 chunk 限制）
- 这破坏了 chunk-local 的计算效率，但确保信息可以跨 chunk 流动
```

### 2.2 不使用 Prompt Token 的决策——需要实验验证的关键假设

**提案论证**:
> "Masking pattern 本身是隐式的 'prompt'……Spike-level tokenization 没有歧义"

这个论证逻辑上是成立的，但包含一个**未经验证的假设**：模型能够仅从输入 pattern 的统计特征推断出当前的 masking 模式。

**反面考虑**:

1. **Inter-region masking vs Neuron masking 在统计上可能难以区分**: 如果 mask 掉的 neurons 恰好都在同一个脑区（inter-region），和随机 mask neurons（neuron masking）的 pattern 差异可能很微妙。在 spike-level tokens 中，模型需要知道每个 unit 属于哪个脑区才能区分这两种模式——而这个信息仅通过 unit_embedding 隐式表示，不在显式输入中。

2. **不同 masking 模式可能需要不同的重建策略**:
   - Temporal masking: 需要时间插值能力（从前后时间推断中间）
   - Neuron masking: 需要空间推断能力（从同群体其他神经元推断缺失神经元）
   - Inter-region masking: 需要跨脑区预测能力

   统一策略可能确实更鲁棒（如 proposal 所论证），但也可能更难优化（一个模型要同时学会三种不同的推断模式）。

**建议**: 保持当前设计（不使用 prompt token）作为默认方案，但将 prompt token 消融实验的优先级提高。如果不使用 prompt token 的方案在预训练 loss 上显著落后于使用 prompt token 的方案，说明模型确实需要 mode-specific 的先验信息。

### 2.3 Output Cross-Attention Readout 方案——与 torch_brain 的对接细节

**提案声明 (第 5.7 节设计备注)**:
> "新增 K 个 learnable readout queries，通过 cross-attention 从 backbone latent tokens 中提取表征"
> "与 POYO 的 output cross-attention 接口一致（torch_brain 现有实现）"

**实际代码验证**:

这个声明是准确的。POYO 的 decoder 确实是一个 cross-attention readout：
```python
# poyo.py:248-254
output_queries = self.session_emb(output_session_index)    # 可学习的 queries
output_queries = output_queries + self.dec_atn(             # cross-attention
    output_queries, latents, output_timestamp_emb, latent_timestamp_emb
)
```

NeuroBridge 的 Readout Queries 可以直接复用这个架构，只需：
1. 将 `output_queries` 从 session_emb 改为 K 个 learnable embeddings
2. 将 `output_timestamps` 设为覆盖整个序列的均匀时间点（或可学习的时间点）
3. 输出不经过 `self.readout`（linear projection），而是送入 Neural Projector

**潜在问题**: POYO 的 decoder cross-attention 使用 `rotate_value=False`，这意味着 output 不包含 RoPE 编码的时间信息反旋。对于行为解码（velocity prediction），这是合适的，因为输出时间点是明确的。但对于 NeuroBridge 的 readout queries（目标是提取全局表征，而非特定时间点的值），是否需要 `rotate_value=True` 值得实验。

### 2.4 两阶段对齐策略中的 Anti-forgetting Loss

**提案声明 (Stage 2)**:
> "联合优化：L_align + α · L_pretrain"
> "L_pretrain 是 anti-forgetting loss（继续 masking 重构）"

**深层分析**:

这个设计是合理的，但有一个实现上的微妙之处：在 Phase 3 Stage 2 中，模型同时需要处理两种不同的输入模式：
1. **CLIP 对齐**: 输入是全部 spike tokens（不 mask），经过完整 encoder-processor-decoder 后输出 readout embeddings
2. **Anti-forgetting**: 输入是部分 masked 的 spike tokens，经过 encoder-processor 后进入 Target Seq Decoder 输出 firing rate predictions

这意味着**每个 training step 需要两次前向传播**（一次不 mask，一次 mask），或者更高效的做法是**在同一个 batch 中混合 masked 和 non-masked 样本**。

proposal 没有讨论这个实现细节，但它对训练效率有显著影响（每步计算量翻倍）。

**建议**: 可以采用以下策略降低开销：
- 每 N 步做一次 anti-forgetting（而非每步），N=5-10
- 或者只在 batch 中随机选取 20% 的样本做 masking 重构

---

## 第三部分：研究叙事与定位

### 3.1 "首次" 声明的严谨性

**提案声明 (第九节)**:
> "首次在 spike-level tokenization + PerceiverIO 架构上实现多任务 masking 自监督预训练"

这个声明在技术上是精确的——确实没有在 POYO 的精确架构上做过 MtM 风格的预训练。但有两个值得注意的邻近工作：

1. **NEDS (ICML 2025 Spotlight)**: 虽然不是 spike-level tokenization，但 NEDS 在 multi-modal 神经数据上做了 multi-task masking（neural masking + behavioral masking + cross-modal masking），并且使用了类似 Perceiver 的架构。如果 NEDS 的后续工作扩展到 spike-level tokenization，NeuroBridge 的"首次"声明可能会被挑战。

2. **NDT3**: 使用自回归预训练而非 masking，但规模更大（350M params, 2000 hours data）。NeuroBridge 需要明确论证 masking 预训练相对于自回归预训练的优势。

**建议**: 在论文中，将"首次"声明的范围限定更精确：
> "首次在保留毫秒级时间分辨率的 spike/event-level tokenization + PerceiverIO 压缩架构上，实现了多空间尺度的 masking 自监督预训练，并将其应用于视觉图像重建"

### 3.2 四个创新点的重叠与优先级

proposal 提出了 4 个创新点：
1. 多任务 Masking + PerceiverIO 兼容性
2. 跨 Session 可扩展 Unit Embedding
3. 两阶段视觉对齐
4. Scaling 与泛化性研究

**深层分析**:

- **创新点 2**（跨 Session Unit Embedding）在技术上几乎等同于 POYO+ 已有的方案（InfiniteVocabEmbedding + few-shot adaptation）。proposal 中的"可选增强" IDEncoder 借鉴自 SPINT，但如果不实现，这个"创新点"就只是复用了 POYO+ 的现有方案。建议将其降级为"技术实现"而非"创新点"。

- **创新点 4**（Scaling 研究）在当前资源（单卡 RTX 4090）下很难做出有说服力的结果。从 ~5M 到 ~30M 参数的 scaling curve 和 NDT3 的 10M→350M 相比太小，且数据量也远不及 NDT3 的 2000 小时。建议将其定位为"补充实验"而非"核心创新"。

**建议的核心叙事**：聚焦于创新点 1 + 3（masking 预训练 + 视觉对齐），这两个的组合确实是该领域的空白。将创新点 2 和 4 作为技术实现和补充实验。

### 3.3 与 MonkeySee 的基线对比——需要更公平的比较

**提案声明**: 多处将 MonkeySee 作为主要 baseline。

**问题**: MonkeySee 使用的是 **CNN-based space-time-resolved decoding**，这是一种直接端到端训练的方法。NeuroBridge 使用预训练 encoder + CLIP 对齐 + Stable Diffusion，这个 pipeline 的复杂度远高于 MonkeySee。

如果 NeuroBridge 的重建质量仅略好于 MonkeySee，审稿人可能会质疑：增加的复杂度是否 justify 了有限的性能提升？

**建议**:
1. 增加一个"去预训练"的消融基线：直接在 TVSD 上从随机初始化的 POYO encoder 做 CLIP 对齐，不经过预训练。这个基线与 NeuroBridge 的唯一差异就是预训练，可以直接证明预训练的价值。
2. 增加一个"POYO supervised encoder + CLIP"基线：用 POYO+ 的监督训练（如 stimulus classification）训练 encoder，再做 CLIP 对齐。这证明了自监督预训练相对于监督预训练的优势。

---

## 第四部分：具体技术细节的问题

### 4.1 Diffusion Adapter 的 [B, 77, 768] 格式

**提案声明 (第 5.7 节)**:
> "将单个 CLIP-aligned neural embedding [B, D_clip] 扩展为 [B, 77, 768] 的条件 tensor"
> "通过 token expander (MLP) + refiner (4 层 Transformer encoder)"

**问题**: 77 tokens × 768 维是 CLIP text encoder 的输出格式（对应最大 77 个 text tokens）。从一个 768 维向量扩展到 77×768 = 59,136 维是一个极大的信息膨胀，4 层 Transformer refiner 需要相当多的训练数据才能学会有意义的 token 分布。

MindEye2 使用了更复杂的方案：将 neural embedding 分为两条路径——一条用于 CLIP 检索（single vector），另一条用于 Diffusion（通过 learned queries 从一个更大的中间表征中提取多个 token）。

**建议**: 如果 TVSD 的训练数据不够大（~22K 图像），可以简化 adapter 设计：
1. 将 readout queries 数量 K 直接设为 77，每个 query 的输出维度设为 768，直接作为 SD 的条件——不需要额外的 expander
2. 或者使用 IP-Adapter 的方案：将 neural embedding 注入 SD UNet 的 cross-attention 作为额外的 KV，而不是替换 text encoder 的输出

### 4.2 不同脑区时间窗口的实现

**提案声明 (第 5.7 节)**:
> "V1 响应：stimulus onset 后 30-80ms"
> "V4 响应：stimulus onset 后 60-120ms"
> "IT 响应：stimulus onset 后 80-200ms"
> "不同脑区应使用不同的时间窗口提取 embedding"

**与 Output Cross-Attention Readout 的兼容性分析**:

如果使用 Output Cross-Attention Readout（方案 B），readout queries 通过 cross-attention 从全序列的 latent tokens 中提取信息。每个 readout query 的 timestamp（RoPE）会引导它关注特定时间段的 latent tokens。

这意味着**不需要手动选择时间窗口**——可以为 V1、V4、IT 分别设计 readout queries，让它们的 timestamps 覆盖对应的响应时间范围：
```python
# V1 readout queries: timestamps in [0.03, 0.08]
# V4 readout queries: timestamps in [0.06, 0.12]
# IT readout queries: timestamps in [0.08, 0.20]
```

这比 proposal 中手动选择时间窗口再 average pooling 的方案更优雅，且与 proposal 推荐的方案 B 天然兼容。但这需要将脑区信息编码到 readout queries 中（例如，使用不同的 query embeddings 或添加 region embedding）。

---

## 第五部分：总结

### 核心优势（保持）
1. 研究空白识别精确——确实没有 spike-level tokenization + PerceiverIO + masking 预训练 + 图像重建的统一框架
2. Pre-PerceiverIO masking 的设计考量深入且合理
3. 不使用 prompt token 的论证有说服力（但需要消融验证）
4. Output Cross-Attention Readout 方案与 torch_brain 代码完美对接

### 需要修正的问题（按优先级）

| 优先级 | 问题 | 建议 |
|--------|------|------|
| **关键** | TVSD MUA 与 spike-level tokenization 的矛盾 | 明确 MUA 预处理策略；改称 "event-level tokenization" |
| **关键** | 模型规模与 GPU 资源不匹配 | 将 dim=512/depth=12 标注为"理想目标"，标注实际起步配置 |
| **高** | Poisson NLL 不适用于 MUA | 区分 spike count loss 和 MUA loss |
| **高** | Grid 时间分辨率过细 (10ms) | 增大到 50-100ms，并按数据类型调整 |
| **中** | Anti-forgetting loss 的实现开销 | 说明双前向传播或交替策略 |
| **中** | "首次"声明的范围 | 精确限定，避免被邻近工作挑战 |
| **低** | 创新点 2/4 的创新性 | 降级为技术实现 / 补充实验 |

---

*本审查旨在帮助 proposal 从"好的研究构想"升级为"可落地执行的技术方案"。所有建议均基于对 torch_brain 源代码的直接验证和对领域现有工作的理解。*
