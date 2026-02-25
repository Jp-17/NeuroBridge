# plan.md 深度审查报告

**审查日期**: 2026-02-25
**审查对象**: `cc_core_files/plan.md` — NeuroBridge 项目执行计划
**参照文档**: `cc_core_files/code_research.md`, `cc_core_files/proposal.md`, `cc_core_files/dataset.md`, torch_brain 源代码
**审查者**: Claude Code (Opus 4.6)

---

## 总评

plan.md 是一份务实且具有良好自我审视能力的执行计划。相比 proposal.md 的理想化设计，plan.md 做了大量合理的降级决策（dim=512→128, depth=12→6, 去掉 IBL, 聚焦 Allen+TVSD），问题清单（第 2.1 节）的识别也很精准。

然而，作为一份需要**实际指导执行**的文档，plan.md 在以下方面存在不足：
1. **关键技术细节模糊**——特别是 MUA tokenization 和 Target Sequence Decoder 的具体实现方案
2. **Phase 间依赖关系和时间节奏**——GO/NO-GO 决策点的定义不够明确
3. **与 torch_brain 代码的对接方案缺失**——哪些文件改、哪些文件新建、如何复用
4. **显存和计算量估算缺失**——dim=128/depth=6 的 forward/backward 显存用量未估算
5. **与 dataset.md 的交叉检验不足**——TVSD 数据量、存储等关键约束未充分反映

以下逐节分析。

---

## 第一部分：问题清单（第 2.1 节）的审查

plan.md 第 2.1 节的 6 个问题识别准确，但解决方案的完整度参差不齐。

### 1.1 问题 #1（严重）：MUA vs spike-level tokenization

**plan.md 方案**: "采用 CaPOYO 模式：`input_value_map = nn.Linear(1, dim//2)` + `unit_emb(dim//2)` 拼接"

**审查意见**: 方向正确，但缺少三个关键的后续决策：

**(a) MUA 时间分辨率的选择**

如 code_research_review 中详细分析的，CaPOYO 的 tokenize() 为每个 time × electrode 创建一个 token。对于 TVSD：
- 原始 MUA（30kHz 降采样到 ~1kHz）：1000 timepoints × 1024 electrodes = **1,024,000 tokens/sample**
- 完全不可行

plan.md 没有明确 MUA 要 bin 到什么时间分辨率。**这是一个关键的未决定参数。**

**建议**：在 Phase 1a（TVSD 数据适配）中必须确定的参数：

| MUA 时间 bin | Tokens/Sample (1024 elec) | Tokens/Sample (256 elec 子集) | 可行性 |
|-------------|--------------------------|-------------------------------|--------|
| 5ms | 200 × 1024 = 204,800 | 200 × 256 = 51,200 | 仅 256 子集勉强可行 |
| 10ms | 100 × 1024 = 102,400 | 100 × 256 = 25,600 | 256 子集可行 |
| 25ms | 40 × 1024 = 40,960 | 40 × 256 = 10,240 | 可行 |
| **50ms** | 20 × 1024 = 20,480 | 20 × 256 = 5,120 | **推荐** |

50ms bin 是一个合理的起点：它与 latent_step=0.125 的时间尺度在同一数量级，且 20K tokens 在 PerceiverIO 的处理范围内（cross-attention 的 KV 序列长度）。

**(b) spike-sorted Allen 数据的处理**

如果 Allen 使用 POYO-style（spike tokens），TVSD 使用 CaPOYO-style（value tokens），那么**两种数据不能用同一个 tokenizer**。plan.md 的决策 1 选择了 CaPOYO-style，但没有讨论 Allen spike data 的处理。

**两种方案**：
1. **统一为 CaPOYO-style**：将 Allen spike data 也转为 binned firing rate，然后用 `input_value_map + unit_emb` 的方式。简单统一，但丧失了 spike-level 的时间精度优势。
2. **双 tokenizer**：spike data 用 POYO-style（unit_emb + token_type_emb），MUA data 用 CaPOYO-style（value_map + unit_emb）。共享 PerceiverIO backbone，只在 tokenization 前端不同。这需要在模型中同时保留两套 embedding。

plan.md 应明确选择其一。**建议方案 1**（统一 CaPOYO-style），原因：
- 简化实现
- Allen 数据在本项目中仅用于预训练验证，不是主力
- binned firing rate 已经包含了足够的信息（POYO+ 的解码性能表明 50-100ms bin 内的 spike count 即可支撑行为解码）

**(c) dim//2 的选择与 proposal 的不一致**

CaPOYO 使用 `unit_emb(dim//2)` + `value_map(1→dim//2)` → concat → `dim`。但 proposal 第 5.4 节的描述暗示所有 embedding 都是 full `dim` 维。这不是一个严重问题，但 plan.md 的模型配置（第 4.3 节）应明确标注 `unit_emb.dim = 64`（即 128//2），而非默认的 128。

### 1.2 问题 #2（严重）：模型规模

**plan.md 方案**: "dim=128, depth=6 (~5M params)"

**审查意见**: 这个降级是合理的，但 **"~5M" 的参数量估算未经验证**。让我重新计算：

配置：dim=128, depth=6, dim_head=64, cross_heads=2, self_heads=4, num_latents_per_step=8

- **Encoder CrossAttn**: inner_dim = 2×64 = 128
  - to_q: 128×128 = 16,384
  - to_kv: 128×128×2 = 32,768
  - to_out: 128×128 = 16,384 + 128(bias) = 16,512
  - 2×LayerNorm(128): 512
  - 总计: ~66K

- **Encoder FFN**: LayerNorm(128) + FeedForward(128)
  - LayerNorm: 256
  - Linear(128, 1024): 131,072 + 1024 = 132,096
  - GEGLU → Linear(512, 128): 65,536 + 128 = 65,664
  - 总计: ~198K

- **Self-Attention (每层)**: inner_dim = 4×64 = 256
  - to_qkv: 128×256×3 = 98,304
  - to_out: 256×128 = 32,768 + 128 = 32,896
  - LayerNorm(128): 256
  - 总计: ~131K

- **Self-Attn FFN (每层)**: 同 Encoder FFN = ~198K

- **6 层 processor**: 6 × (131K + 198K) = 6 × 329K = **~1.97M**

- **Decoder CrossAttn + FFN**: ~66K + ~198K = ~264K

- **Embeddings**:
  - latent_emb: 8 × 128 = 1,024
  - token_type_emb: 4 × 128 = 512
  - CaPOYO-style value_map: Linear(1, 64) = 64 + 64 = 128
  - unit_emb: depends on vocab (如 2000 units: 2000 × 64 = 128K)
  - session_emb: depends on vocab (如 50 sessions: 50 × 128 = 6.4K)
  - task_emb: depends on readout count

- **Base 模型 (不含 InfiniteVocab)**:
  66K + 198K + 1,970K + 264K + 2K ≈ **2.5M**

- **含 InfiniteVocab (2000 units, 50 sessions, readout heads)**:
  2.5M + 128K + 6.4K + readout ≈ **~2.7M**

**结论：plan.md 的 "~5M" 估算偏高，实际约 2.5-3M。** 这是好消息——模型比预���更小，可以使用更大的 batch size 或稍增大模型。

**建议**：
- 如果显存允许，考虑增加到 `dim=192, depth=8` 或 `dim=128, depth=12`（约 5-6M params）
- 或者保持 dim=128/depth=6 但增大 batch_size 以稳定训练
- 在 plan.md 中增加精确的参数量计算表

### 1.3 问题 #3（中等）：Loss 选择

**plan.md 方案**: "Spike count → Poisson NLL, MUA → MSE"

**审查意见**: 正确。torch_brain 已有 `MSELoss` 实现 (`nn/loss.py:26-53`)，可直接复用。但有一个细节需要注意：如果统一使用 CaPOYO-style tokenization（问题 1(b) 的建议方案 1），那么 Allen spike data 也被转为 binned firing rate，此时应统一使用 MSE（因为 binned firing rate 是连续值，不再是 Poisson 计数）。

**更新方案**: 统一使用 MSE Loss。这简化了实现，且在 Allen spike data 被 bin 后，MSE 和 Poisson NLL 的差异很小（firing rate > 1 时两者近似）。

### 1.4 问题 #4（中等）：Grid 时间分辨率

**plan.md 方案**: "50ms"

**审查意见**: 合理。但需要补充：
- 50ms grid 对 200 neurons (Allen) = 4,000 queries → 可行
- 50ms grid 对 1024 electrodes (TVSD) = 20,480 queries → 需要验证显存
- 建议对 TVSD 先用 100ms grid (= 10,240 queries) 或 electrode 子集 (256 electrodes × 50ms = 5,120 queries)

### 1.5 问题 #5-6（低）：IBL 和项目范围

**审查意见**: plan.md 正确决定暂不使用 IBL，聚焦 Allen + TVSD。范围收缩合理。

---

## 第二部分：关键架构决策的审查

### 2.1 决策 5（数据优先级）与 dataset.md 的交叉检验

**plan.md**: "TVSD（核心）> Allen（预训练验证）>> IBL（暂不用）"

**dataset.md 提供的关键约束**（plan.md 未充分反映）：

| 约束 | dataset.md 信息 | plan.md 反映情况 |
|------|----------------|-----------------|
| TVSD MUA 存储: 200-500GB | 明确说明 | 未提及存储需求 |
| TVSD 需要 DataLad+GIN | 明确说明 | Phase 0.4 提到但未详细 |
| THINGS 图像需单独下载 | 明确说明 (OSF/things-dataset) | 未提及 |
| Allen 58 sessions = ~146.5GB | 明确说明 | 未提及 |
| Allen 已注册在 registry.py | **code_research_review 新发现** | 未提及 |
| CLIP embedding 预提取 | dataset.md 有代码示例 | Phase 2 提到但未详细 |

**建议**: 在 Phase 0 中增加一个子任务"存储空间确认"，确保 `/root/autodl-tmp/` 有至少 500GB 可用空间。如果空间不够，需要制定 TVSD 数据的分批下载和处理策略。

### 2.2 模型配置（第 4.3 节）的完整性

**当前配置**:
```yaml
dim: 128
depth: 6
dim_head: 64
cross_heads: 2
self_heads: 4
num_latents_per_step: 8
latent_step: 0.125
sequence_length: 1.0
```

**缺失但重要的参数**:

| 缺失参数 | 默认值 (POYO代码) | 建议值 | 理由 |
|---------|------------------|--------|------|
| ffn_dropout | 0.2 | 0.2 | 保持原始 POYO 设置 |
| lin_dropout | 0.4 | 0.3 | 小模型过度 dropout 可能损害学习 |
| atn_dropout | 0.0 | 0.1 | 预训练时加少量 attention dropout |
| emb_init_scale | 0.02 | 0.02 | 标准值 |
| t_min | 1e-4 | 1e-4 | 标准值 |
| t_max | 2.0627 | 2.0627 | 对 1s 窗口足够 |

另外，**num_latents_per_step=8** 相比 POYO 1.3M 的 16 更少。这意味着每个 125ms 的时间步只有 8 个 latent tokens 来压缩所有 input information。如果 input 是 TVSD 的 1024 electrodes × 50ms bins = ~5K tokens per chunk，压缩比为 5000:8 = 625:1，可能过于激进。

**建议**: 将 `num_latents_per_step` 增加到 16 或 32。参数量增加极少（从 8×128=1K 到 32×128=4K），但对信息保留有显著影响。

### 2.3 Masking 策略的实现细节缺失

**plan.md (Phase 1b)**: 提到 "temporal + neuron" masking，但没有给出：
1. Masking 比例范围（如 15-40% random）
2. Masking 粒度（temporal: 以 ms 为单位？以 latent_step 为单位？）
3. Masking 在数据流的哪个点执行（tokenize 时？forward 前？）
4. 多种 masking 的混合策略（每 batch 随机选一种？每 sample 随机选一种？按概率混合？）

**建议的 Masking 实现方案**（基于对代码的理解）：

```
Masking 应在 tokenize() 之后、forward() 之前执行：

1. tokenize() 生成完整的 input_unit_index, input_timestamps, input_values
2. masking module 根据策略选择要 mask 的 tokens
3. 被 mask 的 tokens 的 index 保存为 "target_mask"
4. 被 mask 的 tokens 从 input 中移除（或替换为 [MASK] token）
5. Target Sequence Decoder 使用 "target_mask" 指定的位置作为重建目标

具体策略：
- Temporal masking: 随机选择 1-3 个连续时间段（每段 25-100ms），mask 该时间段内所有 tokens
- Neuron masking: 随机选择 30-50% 的 neurons/electrodes，mask 这些 unit 的所有 tokens
- Combined: 以 0.5/0.5 的概率随机选择 temporal 或 neuron masking
  （初期建议先分别验证，再组合）
```

---

## 第三部分：Phase 间逻辑与依赖关系

### 3.1 Phase 0 → Phase 1a 的瓶颈：TVSD 数据

**plan.md Phase 0.4**: "启动 TVSD datalad clone"
**plan.md Phase 1a**: "TVSD MUA 数据能通过 CaPOYO-style 前向传播"

**问题**: Phase 0.4 只说"启动克隆"，但 TVSD MUA 数据（200-500GB）的实际下载可能需要相当长时间。如果 Phase 1a 开始时 TVSD 还没下载完，整个 Phase 1a 会被阻塞。

**建议**:
1. Phase 0.4 修改为"启动 TVSD 克隆 + 下载第一个 monkey 的 V1 MUA 数据"（几十 GB，可尽快完成）
2. Phase 1a 先用 V1 的子集数据做适配验证，不等全部数据下载完
3. 增加一个并行任务："TVSD 数据持续下载（后台进行）"

### 3.2 Phase 1b 的验收标准过于模糊

**plan.md**:
```
验收标准：
- 重建 loss 持续下降
- Linear probe R² > 随机初始化
```

**问题**:
1. "重建 loss 持续下降" → 几乎任何训练都会满足这个条件，不构成有效验证
2. "Linear probe R² > 随机初始化" → 没有定义在什么任务上的 R²

**建议的具体验收标准**:

```
Phase 1b GO/NO-GO 标准：

必须满足：
1. 重建 loss 在 50 epochs 后仍在下降（未 saturate）
2. 在 Allen Natural Scenes 分类任务上，pretrained encoder 的
   linear probe accuracy > random encoder + 10%
   （具体：random encoder ~8% = 1/119 × 10 倍，pretrained 应 > 18%）
3. 不同 masking 策略的 loss 收敛值有差异
   （说明模型确实在学习不同的重建模式）

可选但重要：
4. 用 t-SNE 可视化 latent tokens，相同图像的不同重复应聚类
5. 不同脑区的 readout embedding 应形成可区分的 cluster
```

### 3.3 Phase 2（CLIP 对齐）的实现路径不够清晰

**plan.md Phase 2 (第 4.4 节)** 列出了新增文件，但没有给出：
1. Readout module 如何从 POYO backbone 获取 latent tokens
2. InfoNCE loss 的 temperature 参数
3. Stage 1（冻结 encoder）的训练 epochs 和学习率
4. Stage 2（解冻 encoder 后 2-3 层）的具体策略——"后 2-3 层"是什么？

**代码层面的实现分析**:

基于对 torch_brain 代码的理解，CLIP 对齐的代码应该这样对接：

```python
# Phase 2 的核心流程（伪代码）
class NeuroBridgeAligner(nn.Module):
    def __init__(self, poyo_encoder, readout_dim=128, clip_dim=768, num_queries=8):
        # poyo_encoder: 预训练的 POYO/CaPOYO backbone（包括 enc + proc + dec）
        self.encoder = poyo_encoder

        # Readout queries: 复用 POYO decoder cross-attention 的架构
        self.readout_queries = nn.Parameter(torch.randn(num_queries, readout_dim))
        self.readout_timestamps = nn.Parameter(torch.linspace(0, 1, num_queries))
        # ↑ 可学习的时间位置，或固定为均匀分布

        self.readout_cross_attn = RotaryCrossAttention(
            dim=readout_dim, heads=2, dim_head=64, rotate_value=False
        )  # 复用 POYO 的 decoder cross-attn 架构

        # Neural Projector: 映射到 CLIP 空间
        self.projector = nn.Sequential(
            nn.Linear(readout_dim, 512),
            nn.GELU(),
            nn.Linear(512, clip_dim),
        )

    def forward(self, batch):
        # 1. POYO encoder: input tokens → latent tokens
        # 需要访问 encoder 中间状态（latent tokens）而非最终输出
        latents = self.encoder.encode_to_latents(**batch["model_inputs"])

        # 2. Readout: learnable queries attend to latent tokens
        readout_emb = self.readout_cross_attn(
            self.readout_queries, latents,
            self.rotary_emb(self.readout_timestamps), latent_timestamp_emb
        )

        # 3. Pool and project
        neural_emb = readout_emb.mean(dim=1)  # [B, readout_dim]
        clip_pred = self.projector(neural_emb)  # [B, clip_dim]
        clip_pred = F.normalize(clip_pred, dim=-1)

        return clip_pred
```

**关键发现**: 当前 POYO/CaPOYO 的 forward() 方法是一个完整的端到端函数，**没有暴露中间的 latent tokens**。要实现 CLIP readout，需要修改 POYO 的 forward() 方法或添加一个新方法 `encode_to_latents()` 来获取 processor 输出后的 latent tokens。

**建议在 plan.md 中增加**：
- Phase 2 的前置工作：为 POYO/CaPOYO 添加 `encode_to_latents()` 方法
- 明确 readout module 的参数量和计算量

### 3.4 "解冻后 2-3 层" 的歧义

POYO 的 processor 由 `depth` 个 self-attention 层组成。"后 2-3 层"可以指：
1. processor 的最后 2-3 个 self-attention 层 → 即 `proc_layers[-3:]`
2. decoder cross-attention + 最后 2 个 processor 层
3. 整个 decoder + 最后 1 个 processor 层

方案 1 最常见且最安全（gradual unfreezing from top down），**建议明确选择方案 1**，并在 plan.md 中写明。

---

## 第四部分：Target Sequence Decoder 的实现方案

plan.md 将 Target Sequence Decoder 列为 Phase 1b 的新增文件 (`neurobridge/decoders/target_seq_decoder.py`)，但没有给出实现细节。这是预训练阶段最核心的新增组件，需要详细设计。

### 4.1 与 POYO 现有 decoder 的关系

POYO 的 decoder (`dec_atn + dec_ffn + readout`) 本身就是一个 Target Sequence Decoder：
- `output_queries = session_emb(output_session_index)` → 从 session embedding 构建 queries
- `output_timestamps` → 指定要查询的时间点
- `dec_atn` (cross-attention) → 从 latent tokens 读取信息
- `readout` (linear) → 映射到输出维度

NeuroBridge 的 Target Sequence Decoder **本质上就是 POYO 的 decoder**，区别在于：
1. output_queries 不是 session_emb，而是 unit_emb（因为要重建每个 neuron 的活动）
2. output_timestamps 不是行为变量的时间点，而是 masking grid 的时间点
3. readout 的输出维度是 1（每个 neuron × time bin 的预测 firing rate/MUA value）

### 4.2 建议的实现方案

```
Target Sequence Decoder 设计:

输入：
- latent_tokens: [B, N_latent, dim]（来自 backbone processor 输出）
- target_unit_index: [B, N_target] —— 要重建的 neuron/electrode indices
- target_timestamps: [B, N_target] —— 要重建的时间点

流程：
1. target_queries = unit_emb(target_unit_index) + time_emb(target_timestamps)
   - 使用与 encoder 相同的 unit_emb（共享参数）
   - 使用与 backbone 相同的 RotaryTimeEmbedding
2. decoded = target_cross_attn(target_queries, latent_tokens, ...)
   - 新的 cross-attention 层（独立于 backbone 的 dec_atn）
3. output = target_ffn(decoded)
4. prediction = target_readout(output)  # Linear(dim, 1)

参数量估算（dim=128）：
- target_cross_attn: ~66K（与 enc_atn 同结构）
- target_ffn: ~198K
- target_readout: Linear(128, 1) = 129
- 总计: ~264K（仅增加约 10% 的参数量）
```

**关键设计决策**:
1. **是否共享 unit_emb？** 建议共享——这强制 encoder 和 decoder 使用一致的 neuron identity 表征。
2. **target_cross_attn 是否独立于 backbone 的 dec_atn？** 建议独立——因为在 Phase 2 (CLIP 对齐) 中，dec_atn 会被替换为 readout_cross_attn，两者不应耦合。
3. **target_readout 的输出维度？** 如果统一使用 MSE loss，输出 1 维（firing rate scalar）。如果同时支持 Poisson NLL，需要输出 2 维（mean + log_var）或通过 softplus 保证非负。

---

## 第五部分：显存与训练效率估算（缺失）

plan.md 完全没有估算 dim=128/depth=6 配置在 RTX 4090 24GB 上的显存用量。这是一个严重遗漏，可能导致训练启动后立即 OOM。

### 5.1 显存估算

**模型参数显存** (float32):
- ~3M params × 4 bytes = **12 MB**（可忽略）

**Optimizer 状态** (AdamW/Lamb):
- 2 × 参数量 = **24 MB**

**激活值显存** (关键瓶颈):

假设 batch_size=32, sequence_length=1.0, num_input_tokens=5000（TVSD 50ms bin × 1024 elec → ~20K，或 256 elec 子集 → ~5K）

- Input embeddings: B × N_input × dim = 32 × 5000 × 128 = 20M values = **80 MB** (float32)
- Encoder cross-attn KV: B × N_input × inner_dim × 2 = 32 × 5000 × 128 × 2 = **40 MB**
- Latent tokens: B × N_latent × dim = 32 × 64 × 128 = 0.26M = **1 MB**
- Processor layers: 每层 self-attn 的 QKV = B × N_latent × inner_dim × 3 = **~0.3 MB/layer**，6 层 = 1.8 MB
- Target decoder queries: B × N_target × dim = 32 × 5120 × 128 = **80 MB**
- Target cross-attn: 类似 encoder = **40 MB**

**总激活值** (forward): ~240 MB
**总激活值** (backward, 约 2x forward): ~480 MB

**AMP (float16 激活 + float32 权重/优化器)**: 约一半 → ~240 MB

**粗略总计** (AMP):
- 模型 + 优化器: ~36 MB
- 激活值 (forward + backward): ~240 MB
- PyTorch 开销: ~500 MB
- **总计: ~800 MB**

**结论**: dim=128/depth=6 在 RTX 4090 24GB 上**极度宽裕**，即使不用 AMP、batch_size=128 也完全可以。

但如果使用全部 1024 electrodes（20K input tokens），激活值会增大 4 倍（~1 GB），加上 attention score 矩阵（cross-attention: 64 latent × 20K input × num_heads × batch = 很大），可能需要更仔细的估算。

**建议**: plan.md 应包含一个显存估算表，覆盖以下场景：
1. Allen (200 units, spike tokens ~2K) → batch=128
2. TVSD (256 electrode 子集, 50ms bin, ~5K tokens) → batch=64
3. TVSD (1024 electrodes, 50ms bin, ~20K tokens) → batch=16

### 5.2 训练 epoch 和收敛预期

plan.md 没有给出训练 epoch 数量的参考。POYO 的默认配置是 1000 epochs。对于预训练（Phase 1b），需要更多的 epochs（因为 masking 重建比直接监督解码更难收敛）。

**建议**: 预训练阶段至少 500 epochs，Phase 1b 验证可以在 200 epochs 时做中间检查。

---

## 第六部分：文件结构和代码组织

### 6.1 新增文件列表的审查

plan.md 列出了以下新增文件：

**Phase 1a**:
- `neurobridge/data/tvsd_loader.py`
- `neurobridge/data/tvsd_dataset.py`

**Phase 1b**:
- `neurobridge/decoders/target_seq_decoder.py`
- `neurobridge/masking/strategies.py`
- `neurobridge/masking/controller.py`
- `neurobridge/losses/reconstruction_loss.py`

**Phase 2**:
- `neurobridge/alignment/readout.py`
- `neurobridge/alignment/projector.py`
- `neurobridge/alignment/infonce.py`
- `neurobridge/alignment/clip_wrapper.py`
- `neurobridge/generation/diffusion_adapter.py`
- `neurobridge/generation/sd_wrapper.py`
- `scripts/evaluate_reconstruction.py`

**缺失的关键文件**:

| 缺失文件 | 用途 | 所属 Phase |
|---------|------|-----------|
| `neurobridge/models/neurobridge.py` | 主模型类（组合 backbone + masking + decoder） | Phase 1b |
| `neurobridge/train_pretrain.py` | 预训练入口（类似 `examples/poyo/train.py`） | Phase 1b |
| `neurobridge/train_align.py` | CLIP 对齐训练入口 | Phase 2 |
| `neurobridge/configs/` | 配置文件目录 | All phases |
| `neurobridge/data/allen_adapter.py` | Allen 数据适配器（将 Allen NWB → torch_brain Data） | Phase 1a |
| `neurobridge/utils/` | 工具函数（如 register_modality 的调用） | Phase 1a |

**建议**: 补充上述文件到计划中，特别是 `neurobridge/models/neurobridge.py`——这是整个项目的核心，应该在 Phase 1b 开始时首先设计。

### 6.2 对 torch_brain 源代码的修改

plan.md 没有明确哪些**已有文件**需要修改。基于分析，至少需要：

| 需要修改的文件 | 修改内容 | 理由 |
|--------------|---------|------|
| `torch_brain/models/poyo.py` 或 `capoyo.py` | 添加 `encode_to_latents()` 方法 | Phase 2 CLIP 对齐需要中间表征 |
| `torch_brain/registry.py` | 注册 MUA 相关的新模态 | TVSD MUA 重建目标 |
| `torch_brain/nn/loss.py` | 可能需要添加 Gaussian NLL loss | 如果 MSE 不够灵活 |

**建议**: plan.md 应明确标注哪些改动是"修改现有代码"（影响范围更大，需要更谨慎），哪些是"新增代码"。

---

## 第七部分：风险登记表的审查

### 7.1 缺失的风险

plan.md 的风险表列了 5 个风险，但遗漏了以下重要风险：

| 新增风险 | 概率 | 影响 | 应对策略 |
|---------|------|------|---------|
| **Attention OOM**: TVSD 1024 electrodes 的 cross-attention 矩阵过大 | 中 | 训练卡住 | 使用 electrode 子集（256/512）；或使用 xformers varlen attention |
| **TVSD 数据格式与预期不符**: dataset.md 的文件路径假设可能不准确（TVSD 仓库结构未实际验证） | 高 | Phase 1a 阻塞 | Phase 0 就下载少量数据验证结构 |
| **CaPOYO-style tokenization 对 spike 数据效果差**: 将 spike 转为 binned rate 丧失时间精度 | 低 | Allen 预训练效果差 | 可回退到双 tokenizer |
| **预训练不收敛**: masking 比例过高或过低导致 loss 不下降 | 中 | Phase 1b 延迟 | 从低 masking 比例（15%）开始，逐步增加 |
| **CLIP 空间维度错配**: ViT-L/14 输出 768 维，但 projector 映射效果差 | 低 | Phase 2 效果差 | 增大 projector 容量；尝试 DINOv2 |

### 7.2 GO/NO-GO 决策点的定义不够明确

plan.md 定义了 3 个 GO/NO-GO 决策点，但只有标题没有具体标准。建议：

**GO/NO-GO #1 (Phase 1a 末尾)**:
- GO: TVSD MUA 数据能通过 CaPOYO-style tokenize() 和 forward()，输出形状正确
- NO-GO: 如果 token 数量 > 50K 导致 OOM → 需要更激进的 binning 或 electrode 子集

**GO/NO-GO #2 (Phase 1b 末尾)**:
- GO: 预训练 loss 在 200 epochs 后仍在下降，且 linear probe accuracy > random + 10%
- NO-GO: loss saturate 或 linear probe 无提升 → 检查 masking 策略、模型规模、数据质量

**GO/NO-GO #3 (Phase 2 Stage 1 末尾)**:
- GO: CLIP Top-5 retrieval > 15% (从 118 张 Allen 图或 TVSD 测试集中检索)
- NO-GO: retrieval 接近随机 → 增大 projector、尝试 DINOv2、检查数据质量

---

## 第八部分：总结与建议

### 8.1 plan.md 的优势（保持）

1. **务实的资源降级**：dim=512→128, 去掉 IBL, 聚焦双数据集
2. **清晰的问题识别**：6 个问题的严重程度评估准确
3. **三个降级方案（A/B/C）**：体现了良好的风险意识
4. **Phase 0 的 baseline 验证**：先复现再创新的策略正确

### 8.2 需要补充的关键内容

| 优先级 | 内容 | 影响 |
|--------|------|------|
| **关键** | MUA 时间分辨率选择（建议 50ms bin） | 决定 token 数量和可行性 |
| **关键** | Allen spike data 统一处理方案（建议统一 CaPOYO-style） | 影响 tokenizer 设计 |
| **关键** | Target Sequence Decoder 的具体实现方案 | Phase 1b 核心组件 |
| **高** | 显存估算表（不同数据集 × 不同 batch size） | 避免 OOM |
| **高** | 精确的参数量计算（当前 ~5M 估算偏高，实际 ~2.5-3M） | 影响模型规模决策 |
| **高** | POYO 源码修改清单（encode_to_latents 方法） | Phase 2 前置依赖 |
| **中** | GO/NO-GO 决策点的具体量化标准 | 避免决策模糊 |
| **中** | 完整的新增文件清单（含 train 脚本、config） | 执行指导 |
| **中** | num_latents_per_step 增大到 16-32 | 信息压缩比太大 |
| **低** | 训练 epoch 数量和学习率 schedule | 可在实验中调整 |
| **低** | Masking 策略的具体参数（比例、粒度） | 可在实验中调整 |

### 8.3 建议的下一步行动

1. **立即执行**: Phase 0.1-0.3（环境安装 + POYO baseline 复现）—— 这部分不依赖任何决策
2. **并行执行**: Phase 0.4-0.5（TVSD 克隆 + Allen 下载）+ MUA 时间分辨率决策
3. **Phase 1a 开始前确定**: 统一 CaPOYO-style tokenization + MUA 50ms bin + 256 electrode 子集起步
4. **Phase 1b 开始前完成**: Target Sequence Decoder 的详细设计文档 + NeuroBridge 主模型类设计

---

*本审查覆盖了 plan.md 与 code_research.md、proposal.md、dataset.md 的全面交叉验证，以及与 torch_brain 源代码的直接对照。所有参数量计算均可通过阅读源代码验证。*
