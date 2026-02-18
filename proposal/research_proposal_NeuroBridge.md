# NeuroBridge：基于多任务Masking预训练的Spiking数据通用神经表征与脑-图像重建

## Universal Neural Representations via Multi-Task Masked Pretraining for Brain-to-Image Reconstruction from Spiking Data

---

**撰写日期**：2026年2月13日
**研究方向**：计算神经科学 × 深度学习 × 脑机接口
**目标会议/期刊**：NeurIPS / ICML / Nature Methods

---

## 一、研究背景 (Background)

### 1.1 问题背景

从大脑神经活动中重建视觉刺激图像（brain-to-image reconstruction）是计算神经科学和脑机接口领域的核心挑战之一。现有的图像重建工作主要集中在两条路线上：

**基于fMRI的路线**：MindEye系列（Scotti et al., 2023/2024）利用fMRI BOLD信号与CLIP视觉-语义空间的对齐，驱动Stable Diffusion生成图像，已取得SOTA成果。MindEye2进一步实现了仅需1小时fMRI数据即可完成跨被试的图像重建。然而，fMRI的时间分辨率极低（秒级），无法捕捉视觉处理的快速动态过程。

**基于spiking数据的路线**：侵入式电极记录的spiking数据具有毫秒级时间分辨率和单神经元空间分辨率，理论上承载着更丰富的视觉信息。MonkeySee（2024）在TVSD数据集上初步验证了从猕猴视觉皮层spiking数据重建图像的可行性，但其使用简单的线性/浅层非线性映射，未涉及大规模预训练或跨session迁移。

### 1.2 spiking数据基础模型的发展

近年来，基于spiking数据的神经基础模型快速发展，已从单session transformer发展到多被试、多脑区架构：

- **NDT系列**（Ye et al., 2021-2025）：从NDT1的单session BERT风格masked autoencoding，到NDT3的350M参数自回归模型，在2000小时数据上训练
- **POYO/POYO+**（Azabou et al., 2023/2025）：开创性地将每个spike作为独立token处理，通过PerceiverIO实现高效序列压缩，保留毫秒级时间分辨率
- **MtM**（Hurwitz et al., NeurIPS 2024）：引入四种互补的masking策略（temporal、neuron、intra-region、inter-region）进行多空间尺度的自监督预训练
- **SPINT**（NeurIPS 2025）：通过IDEncoder实现置换不变的跨session神经元对齐，无需梯度更新即可迁移
- **NEDS**（ICML 2025 Spotlight）：首个统一编码和解码的多模态神经基础模型

### 1.3 研究意义

然而，要从spiking数据实现高质量图像重建，仍需克服三个核心技术瓶颈：

**瓶颈一：高质量神经表征的学习**。现有spiking数据编码器要么是纯监督的（POYO），要么预训练策略与高效压缩架构不兼容（MtM使用binned representation而非spike-level tokenization）。缺乏一种能够在大规模无标签spiking数据上预训练、学习通用神经动力学表征的方法。

**瓶颈二：跨session的表征一致性**。不同记录session的神经元组成完全不同（"词汇表缺失"问题），如果编码器只在特定session上有效，其实际应用价值将大打折扣。需要在表征空间层面实现跨session的一致性。

**瓶颈三：从神经表征到视觉语义空间的有效对齐**。spiking数据的表征空间与CLIP等视觉-语言模型的嵌入空间存在巨大的模态鸿沟（modality gap）。需要高效的跨模态对齐方法，将预训练得到的神经表征映射到CLIP空间，再通过Stable Diffusion生成图像。

### 1.4 研究动机

本项目的核心观察是：**现有方法在上述三个维度上各自解决了部分问题，但没有一个统一的框架能同时解决所有瓶颈**。

| 维度 | POYO | MtM | NDT3 | MonkeySee | MindEye2 | **NeuroBridge (本项目)** |
|------|------|-----|------|-----------|----------|----------------------|
| 数据类型 | Spiking | Spiking | Binned rates | Spiking | fMRI | Spiking |
| Tokenization | Spike-level | Neuron bins | Population vector | Simple | Voxel | Spike-level + PerceiverIO |
| 自监督预训练 | ❌ | ✅ Multi-task masking | ✅ Masked prediction | ❌ | ❌ | ✅ Multi-task masking |
| 跨Session | ⚠️ 需重学习 | ⚠️ Session emb | ⚠️ Session emb | ❌ 单session | N/A | ✅ 可扩展unit embedding |
| 图像重建 | ❌ | ❌ | ❌ | ✅ 浅层映射 | ✅ CLIP+SD | ✅ 预训练表征+CLIP+SD |
| 时间分辨率利用 | ms级 | ms级 | 20ms bins | 粗bins | ~秒级 | ms级（可选时间窗口）|

本项目旨在构建一个**通过多任务masking预训练学习通用神经表征，并通过跨模态对齐实现脑-图像重建**的完整pipeline，命名为**NeuroBridge**。

---

## 二、问题定义 (Problem Statement)

### 2.1 核心问题

**如何从大规模、多session的spiking神经数据中，通过自监督预训练学习到通用的、跨session一致的神经表征，并将其有效对齐到视觉语义空间，实现高质量的脑-图像重建？**

### 2.2 问题的挑战性

1. **spike-level tokenization与自监督预训练的兼容性**：POYO的spike-level tokenization保留了最高时间精度，但其变长序列特性与传统的masking预训练不直接兼容。MtM的多任务masking策略在binned representation上设计，无法直接移植到spike-level tokens上。如何在spike-level tokenization + PerceiverIO压缩架构上实现有效的多任务masking预训练，是一个关键的技术挑战。

2. **跨session神经元对齐**：spiking数据最根本的挑战是跨被试或session完全缺乏标准神经元对应关系。即使在同一被试内，电极漂移也导致神经元在sessions间出现和消失。需要一种不依赖固定神经元身份的表征学习方法。

3. **模态鸿沟**：spiking数据的神经表征空间与CLIP图像嵌入空间之间存在巨大差异——前者编码的是毫秒级神经动力学，后者编码的是高层视觉语义。跨越这一鸿沟需要精心设计的对齐策略。

4. **数据异质性与scaling**：spiking数据集来自不同物种（小鼠、猕猴）、不同脑区（V1、V4、IT等）、不同实验范式，数据异质性极高。Jiang et al.（2025）发现简单的数据积累会因异质性限制scaling收益。

### 2.3 问题的范围

本研究聚焦于：
- **输入模态**：invasive spiking数据（electrophysiology recordings）
- **目标任务**：(1) 多任务masking预测（自监督预训练）；(2) 视觉图像重建（下游任务）
- **目标物种**：小鼠（Allen Brain Observatory）和猕猴（TVSD、实验室数据）
- **目标脑区**：视觉皮层为主（V1、V4、IT），兼顾其他脑区的泛化性

---

## 三、相关工作 (Related Work)

### 3.1 神经数据基础模型

#### 3.1.1 POYO / POYO+（Azabou et al., NeurIPS 2023 / ICLR 2025 Spotlight）

POYO开创性地将每个单独的spike作为离散token处理，每个spike token = learnable unit embedding (per-neuron, D=128) + Rotary Position Embeddings (RoPE) 编码的时间戳。通过PerceiverIO backbone将variable-length spike token序列通过cross-attention压缩到256个固定latent tokens，经过24层self-attention处理后通过output cross-attention解码行为变量。

**优势**：保留毫秒级时间分辨率，无需time-binning；自然适应不同神经元数量和发放率。
**局限**：仅监督学习（无自监督预训练）；固定unit embeddings不能很好处理神经元漂移。

POYO+扩展到钙成像数据，处理多脑区/多细胞类型记录（Allen Brain Observatory，100,000+神经元）。

#### 3.1.2 MtM（Hurwitz et al., NeurIPS 2024）

MtM引入四种互补的masking方案用于跨空间尺度的自监督学习：
1. **Causal masking**：前向预测
2. **Neuron masking**：学习co-firing结构（co-smoothing）
3. **Intra-region masking**：在脑区内预测
4. **Inter-region masking**：跨脑区预测

使用可学习的prompt tokens在推理时切换masking模式。在IBL数据上展示了多空间尺度动力学的捕获能力。

**优势**：多任务masking策略全面覆盖不同空间尺度的神经动力学。
**局限**：绑定到NDT架构（20ms binning），损失时间精度；未使用PerceiverIO压缩；未涉及跨session迁移或图像重建。

#### 3.1.3 SPINT（NeurIPS 2025 / CoSyNe 2026）

SPINT通过架构不变性和IDEncoder解决跨session神经元对齐问题。每个neural unit的temporal活动成为一个spatial token，不使用固定位置编码，而是用IDEncoder网络从测试session的未标记校准trials获取context-dependent identity embeddings。整个架构在数学上**置换不变**——无论神经元排序如何，输出相同。

**优势**：无梯度迁移，仅需在校准数据上通过IDEncoder前向传播。
**局限**：仅监督学习；小规模评估（FALCON benchmark，仅运动任务）；Binned表示损失时间精度。

#### 3.1.4 NDT3（Ye et al., ICLR 2025）

最大规模的神经基础模型，350M参数，在2000小时来自30+被试跨10个实验室的数据上训练。使用自回归transformer，将spike counts离散化为分类变量。

**优势**：大规模预训练；联合多模态建模。
**关键局限**：作者明确指出scaling自回归transformer不太可能解决传感器变异性和输出刻板性。

#### 3.1.5 NEDS（ICML 2025 Spotlight）

首个统一编码和解码的模型，多模态共享transformer backbone，多任务masking策略（neural masking、behavioral masking、within-modality、cross-modal masking）。在IBL数据上超越POYO+和NDT2。

**优势**：统一编码/解码框架；预训练表示展现脑区识别能力。
**局限**：仅在一个任务范式（IBL视觉决策）上评估；仅限于Neuropixels记录。

### 3.2 脑-图像重建

#### 3.2.1 MindEye / MindEye2（Scotti et al., 2023/2024, ICML 2024）

MindEye系列在fMRI→image重建领域取得SOTA成果。核心pipeline：fMRI voxels → 非线性映射 → CLIP image embedding space → Stable Diffusion生成图像。MindEye2通过跨被试预训练（线性映射到共享latent space），实现仅需1小时fMRI数据的图像重建。

**借鉴价值**：CLIP对齐 + Diffusion Adapter的pipeline设计具有重要参考价值。
**不直接适用的原因**：fMRI和spiking数据在表征空间上完全不同——fMRI是大脑区域级BOLD信号，spiking是单神经元级别的离散事件。

#### 3.2.2 MonkeySee（2024）

在TVSD数据集上实现了从猕猴视觉皮层spiking数据到图像的重建。使用CNN-based decoder进行space-time-resolved decoding。

**优势**：首次系统地展示spiking数据图像重建的可行性。
**局限**：简单的线性/浅层非线性映射；未涉及大规模预训练或跨session迁移；未使用CLIP对齐或生成模型。

### 3.3 研究空白

通过对现有工作的系统梳理，我们识别出以下关键研究空白：

1. **没有在spike-level tokenization + PerceiverIO架构上实现多任务masking自监督预训练的工作**
2. **没有将大规模自监督预训练的通用神经表征应用于spiking数据图像重建的工作**
3. **没有同时解决高时间分辨率保留、跨session泛化、和视觉语义对齐的统一框架**

---

## 四、研究创新点 (Innovation Points)

### 创新点 1：多任务Masking预训练与PerceiverIO压缩的兼容性设计

**核心问题**：MtM的多任务masking策略在binned representation上设计，与POYO的spike-level tokenization + PerceiverIO压缩不直接兼容。

**解决方案**：设计在tokenization之前（即spike token层面）进行masking操作的策略。具体地：

- **Neuron masking / Region masking**：在PerceiverIO压缩前，dropout部分神经元的所有spike tokens
- **Temporal masking**：在PerceiverIO压缩前，mask部分时间窗口的所有spike tokens

**重构预测**：设计固定的neuron × time grid作为预测目标——通过cross-attention从latent tokens中解码每个grid位置的firing rate（spike count），仅在masked位置计算Poisson NLL loss。

**为什么这是一个有效的创新**：
- 避免在spike token级别做masking导致的"信息泄露"问题（mask掉的spike位置本身已包含"这里有spike"的信息）
- 预测firing rate比重构变长spike序列更稳定，输出维度固定
- 与PerceiverIO压缩完美兼容——编码端压缩、解码端用grid queries展开

> #### 📐 设计备注：Temporal Masking 的 Pre-PerceiverIO vs Post-PerceiverIO 技术考量
>
> Temporal masking 的操作层级（Pre-PerceiverIO 还是 Post-PerceiverIO）是一个需要审慎权衡的架构设计决策。以下记录两种方案的对比分析和最终选择的理由。
>
> **方案A：Pre-PerceiverIO（当前选择）**
>
> 在 PerceiverIO 压缩之前，直接移除被 mask 时间窗口内的所有 spike tokens。
>
> - ✅ **时间尺度一致性**：masking 和 prediction target（neuron × time grid, 10ms bins）都在原始时间坐标系中定义，天然对齐。不存在"masking 粒度 50ms vs 预测粒度 10ms"的错位问题。
> - ✅ **灵活的 masking 粒度**：masking 时间窗口不需要与 chunk 边界对齐。例如可以 mask 一个 30ms 窗口横跨两个 chunk 交界处，这样每个 chunk 仍有部分 spike tokens 保留，PerceiverIO 不会收到完全空的输入。
> - ✅ **与 Neuron/Region masking 设计统一**：所有 4 种 masking 策略都在同一层级（Pre-PerceiverIO）操作，架构简洁一致。
> - ⚠️ **潜在风险——"空 chunk" 问题**：如果 masking 窗口恰好覆盖整个 chunk（50ms），PerceiverIO 的 cross-attention 在该 chunk 上没有 key/value，latent queries 只能输出接近初始化的默认值。**应对策略**：(1) 设计 masking 窗口不与 chunk 边界对齐，确保每个 chunk 至少保留部分 tokens；(2) 使用 25-40ms 的 masking 窗口粒度（小于 50ms chunk size），避免完全清空单个 chunk。
>
> **方案B：Post-PerceiverIO（备选方案）**
>
> 所有 spike tokens 先正常通过 PerceiverIO 压缩，然后在 latent token 层面 mask 掉某些 chunk 的 latent tokens。
>
> - ✅ **PerceiverIO 获得完整训练信号**：所有数据都参与压缩，PerceiverIO 的梯度更充分。
> - ✅ **语义级别的 masking**：每个 chunk 的 latent tokens 是有意义的压缩表征，mask 整个 chunk 类似 BERT mask 整个 token，预测任务信息量丰富。
> - ❌ **时间尺度错位（关键问题）**：Post-PerceiverIO masking 的粒度被锁定为 chunk size（50ms），但 Target Sequence Decoder 的 query grid 是 10ms bins。decoder 的 cross-attention queries 按 10ms 去读取信息，而被 mask 的信息以 50ms 为单位消失。masking 语义单元和预测语义单元不在同一时间尺度上，造成任务定义的不自然。
> - ❌ **无法进行细粒度 temporal masking**：不能 mask 10-20ms 的短窗口，只能 mask 整个 50ms chunk，限制了预训练任务的多样性。
>
> **结论**：选择 **Pre-PerceiverIO**，核心理由是保持 masking 与 prediction target 在同一时间坐标系内。通过控制 masking 窗口粒度（不与 chunk 边界对齐、窗口大小 < chunk size）来规避"空 chunk"风险。Post-PerceiverIO 方案作为 fallback 保留，在 Pre 方案遇到训练不稳定时可回退尝试。

### 创新点 2：跨Session一致的可扩展Unit Embedding

**核心问题**：不同session记录的神经元完全不同，需要一种不依赖固定神经元身份的表征方法。

**解决方案**：借鉴POYO的可扩展unit embedding方案，每个神经元通过可学习的embedding进行表征。在新session中：
- 冻结模型主体
- 仅通过梯度下降学习新的unit embeddings

**可选增强**：引入类似SPINT的IDEncoder机制，通过feed-forward网络从校准数据中动态生成unit embeddings，实现gradient-free的跨session迁移。

**为什么这是必要的**：
- POYO的方案已证明简单有效，适合大规模预训练
- IDEncoder提供了更优雅的zero-shot迁移路径
- 两种方案可以并行实验，选择效果更好的

### 创新点 3：Spiking数据的两阶段视觉对齐（Contrastive + Diffusion）

**核心问题**：spiking数据的神经表征与CLIP视觉语义空间之间存在巨大模态鸿沟。

**解决方案**：采用两阶段对齐策略：
1. **Stage 1 - 对比学习对齐**：使用InfoNCE loss将预训练encoder的神经表征（通过 output cross-attention readout queries 提取）映射到CLIP image embedding space。冻结encoder和CLIP模型，仅训练 readout queries + projector。
2. **Stage 2 - 联合微调**：解冻encoder最后几层，联合优化InfoNCE对齐loss + anti-forgetting loss（保持masking重构能力），防止遗忘预训练知识。
3. **Diffusion Adapter**：将CLIP-aligned neural embeddings扩展为[B, 77, 768]的条件tensor，注入Stable Diffusion的UNet cross-attention层。

**与MindEye的关键区别**：
- MindEye从fMRI直接对齐到CLIP，本项目先在大规模无标签spiking数据上预训练通用表征，再对齐
- 预训练表征捕获的丰富时间动态信息（ms级分辨率）可以提供比fMRI更精细的视觉信息
- 不同脑区（V1→V4→IT）使用不同时间窗口提取embedding，利用视觉处理的层级时间特性

### 创新点 4：系统性的Scaling与泛化性研究

**核心问题**：Jiang et al.（2025）发现spiking数据的异质性限制了简单scaling的收益。

**解决方案**：系统性地研究以下scaling维度：
- **Parameter scaling**：从~10M到~100M参数的模型规模变化
- **Data scaling**：从单session到多session/多被试的数据规模变化
- **Task scaling**：从单任务到多任务masking的效果变化

同时进行跨session和跨任务的泛化性测试，提供该领域亟需的scaling law分析。

---

## 五、方法设计 (Methodology)

### 5.1 总体框架

NeuroBridge采用**"先学好表征，再做好对齐"**的两阶段策略：

```
===========================================================================
                    Phase 1-2: 自监督Masking预训练
===========================================================================

输入:
  Spike Events ──→ SpikeTokenizer (with unit embedding) ──→ spike tokens
                                                              │
  Multi-Task Masking Controller ──→ 随机选择 masking 模式 ──→ 移除部分 tokens
  (无 prompt token，模型从缺失 pattern 自行推断)                │
                                                              │
  spike tokens (部分 masked) ──→ PerceiverIO ──→ latent tokens [B, N_chunks, N_lat, D]
                                                              │
                              Temporal Transformer Backbone
                                                              │
                              Target Seq Decoder (cross-attend from latents)
                              (通用 query grid，仅 loss mask 随 masking 模式不同)
                                                              │
                         Predicted firing rates on neuron×time grid
                                                              │
                         Loss: Poisson NLL on masked grid positions

===========================================================================
                    Phase 3: 视觉对齐与图像重建
===========================================================================

预训练 Encoder (冻结/微调):
  Spike Events (全部，不mask) ──→ PerceiverIO ──→ Backbone ──→ latent tokens
                                                                  │
  Output Cross-Attention (K个 learnable readout queries) ──→ neural embedding [B, K, D]
                                                                  │
  Neural Projector (MLP) ──→ CLIP-aligned embedding [B, D_clip]
                                │    ↑
                         InfoNCE Loss ← CLIP Image Encoder (冻结)
                                │
  Diffusion Adapter ──→ SD condition [B, 77, 768]
                          │
  Stable Diffusion (冻结) ──→ 重建图像 [B, 3, 512, 512]
```

### 5.2 Tokenization：Spike-Level + RoPE

借鉴POYO的设计，每个spike event成为一个独立的token：

- **Unit Embedding**：每个神经元拥有可学习的embedding向量（D=128），支持任意数量神经元
- **时间编码**：通过Rotary Position Embeddings (RoPE) 注入精确的时间戳信息，保留毫秒级时间分辨率
- **Variable-length序列**：不同trial/时间窗口的spike数量不同，token序列长度自然变化

```python
# Tokenization伪代码
for each spike event (unit_id, spike_time):
    token = unit_embedding[unit_id]  # [D]
    token = apply_rope(token, spike_time)  # 注入时间信息
    tokens.append(token)
```

### 5.3 空间压缩：PerceiverIO Cross-Attention

直接复用torch_brain的PerceiverIO实现，将variable-length spike token序列压缩到固定数量的latent tokens：

- 每个时间窗口（如50ms chunk）内的spike tokens通过cross-attention压缩到N_lat个latent tokens（如N_lat=4）
- Latent queries是可学习的参数
- 压缩比通常为10x-100x，大幅减少后续transformer的计算开销

### 5.4 Temporal Backbone：双向Transformer

使用双向Transformer处理PerceiverIO压缩后的latent token序列：

- **不使用因果attention mask**（每个位置看到全部上下文）
- 适合masking预训练（BERT-style双向）和离线图像重建
- 每2层self-attention后插入cross-attention（为Phase 3多模态条件预留）
- 参数配置：12层，d_model=512，8 heads，FlashAttention 2

### 5.5 Multi-Task Masking Controller

从MtM移植四种masking策略，但适配到PerceiverIO压缩架构：

| Masking模式 | 操作层级 | 描述 | Masking比例 |
|------------|--------|------|-----------|
| Temporal | Pre-PerceiverIO | Mask 50-75%时间窗口的所有spike tokens | 50-75% |
| Neuron | Pre-PerceiverIO | Dropout 30-50%神经元的所有spike tokens | 30-50% |
| Intra-region | Pre-PerceiverIO | 在脑区内mask部分神经元 | 50-75% |
| Inter-region | Pre-PerceiverIO | Mask整个脑区的所有神经元 | 50-100% |

**关键适配**：所有 4 种 masking 策略均在 tokenization 后、PerceiverIO 压缩前进行（Pre-PerceiverIO）。被 mask 的 spike tokens 直接从输入序列中移除（不是替换为 [MASK] 符号），PerceiverIO 只处理保留的 tokens。训练时按概率随机交替使用四种模式。

> #### 📐 设计备注：Prompt Token 与 Query Grid 的设计考量
>
> **关于区分 masking 模式的 prompt token：建议不使用**
>
> MtM 原始设计中为每种 masking 模式配备一个 learnable prompt token，让模型识别当前模式。但在 NeuroBridge 的 spike-level tokenization 架构下，我们认为 prompt token **不是必需的**，去掉可能更好。理由如下：
>
> 1. **Masking pattern 本身是隐式的 "prompt"**。在 spike-level tokenization 中，被 mask 的 tokens 是直接缺席的（不是被替换为某个占位符）。Temporal masking 导致某些时间窗口的 spike tokens 消失（部分 chunk token 数骤减）；Neuron masking 导致某些神经元的 tokens 在所有 chunk 中均匀消失。这两种 pattern 的统计特征截然不同，模型可以从输入中自行推断缺失模式。这与 MtM 的情况不同——MtM 用 binned representation，masking 是将 bin 值置零，但零值既可表示"被 mask"也可表示"真的没有 spike"，存在歧义，因此需要 prompt token 消歧义。Spike-level tokenization 没有这个歧义。
>
> 2. **不区分模式可能学到更鲁棒的表征**。如果模型被告知当前 masking 模式，它可能发展 mode-specific 的重构策略，导致不同模式下的表征不在同一空间。去掉 prompt token 迫使模型发展统一的缺失数据推断策略，产生更通用的表征。这也更符合真实场景——实际记录中电极漂移、死通道等导致的数据缺失不会告知"当前缺失模式"。
>
> 3. **消除下游阶段的 prompt 处理问题**。如果预训练使用 prompt tokens，下游（不做 masking）时面临"用哪个 prompt"的难题。去掉 prompt token 后，预训练和下游的模型输入格式完全一致，架构更简洁。
>
> 如需消融验证，可对比：(a) 无 prompt token、(b) 4 个 mode-specific prompt tokens、(c) 下游时用 4 个 prompt 分别推理再 ensemble。
>
> **关于 Target Sequence Decoder 的 Query Grid：所有 masking 模式共享通用 grid**
>
> Target Sequence Decoder 使用的 neuron × time query grid **在所有 masking 模式下结构完全相同**。Grid 始终覆盖完整的 (N_neurons × T_bins) 空间，每个 grid 位置的 query embedding = neuron_emb + time_emb，不随 masking 模式变化。
>
> 不同 masking 模式之间变化的**仅是 loss mask**（在哪些 grid 位置上计算 Poisson NLL）：
>
> | Masking 模式 | Loss mask pattern | 说明 |
> |-------------|------------------|------|
> | Temporal | 若干"整列"（被 mask 时间窗 × 所有神经元） | 预测被遮挡时间段内所有神经元的活动 |
> | Neuron | 若干"整行"（被 mask 神经元 × 所有时间 bins） | 预测被遮挡神经元在全时间段的活动 |
> | Intra-region | 某脑区内的若干"整行" | 预测脑区内被遮挡神经元的活动 |
> | Inter-region | 某脑区的"所有行" | 预测整个脑区所有神经元的活动 |
>
> Decoder 的所有参数（query embeddings、cross-attention weights、output projection）在 4 种模式间**完全共享**，不存在 mode-specific 的解码组件。这一设计保证了：(1) 不同 masking 模式学到的表征在同一空间内；(2) 架构简洁，无需按模式切换 decoder；(3) 模型被迫学会从任意缺失 pattern 中重构，而非依赖模式先验。

### 5.6 Target Sequence Decoder（核心创新模块）

**设计思路**：不直接重构变长的spike events，而是在固定的neuron × time grid上预测firing rates。

**核心设计**：

1. 预定义一个 (N_neurons × T_bins) 的2D query grid，time_bin_ms = 10ms
2. 每个grid位置有一个query embedding = neuron_emb + time_emb
3. Queries通过cross-attention从latent tokens中"读取"信息
4. 输出每个grid位置的predicted firing rate
5. Loss只在masked positions上计算（类比BERT只在[MASK]上计算loss）

**为什么不直接mask spike tokens再用其位置做query**：
- 如果mask掉某个spike token并用其位置做query，decoder已经知道"这个位置有spike"（因为只有发放了spike的位置才有token），预训练信号太弱
- 在grid上预测firing rate，模型需要真正推断"某个神经元在某个时间段发放了多少个spikes"，这是信息量丰富的预训练目标

**Loss设计**：在masked grid位置计算Poisson NLL loss：
$$L = \sum_{(i,j) \in \text{masked}} [\lambda_{ij} - k_{ij} \cdot \log(\lambda_{ij})]$$

其中 $\lambda_{ij}$ 是predicted firing rate，$k_{ij}$ 是observed spike count。

### 5.7 Phase 3：Neural-to-CLIP对齐与图像重建

> #### 📐 设计备注：从预训练模型提取神经表征的方案选择
>
> 预训练阶段（Phase 1-2）的目标是 masking 重构，模型从未被显式训练过"输出一个好的 summary 向量"。进入下游任务时，需要一个明确的表征提取策略。这个过渡在其他模态的基础模型中有成熟的先例可供参考。
>
> **跨模态参考：masked 预训练 → 表征提取的范式**
>
> | 模态 | 代表工作 | 预训练任务 | 下游表征提取方式 | 与 NeuroBridge 的关系 |
> |------|---------|----------|----------------|---------------------|
> | NLP | BERT | Masked Language Modeling (mask 15% tokens) | 丢弃 MLM head，[CLS] token 或 avg pooling | 最经典的 "masked pretrain → extract repr" 范式 |
> | Vision | MAE (He et al., 2022) | Masked Autoencoding (mask 75% patches) | 丢弃 decoder，encoder 处理全部 patches → [CLS] / avg pool | **最直接的类比**：MAE encoder 只看可见 patches（类比 PerceiverIO 只看未 mask spike tokens），decoder 重构像素（类比 Target Seq Decoder 重构 firing rates），下游丢弃 decoder |
> | Audio | wav2vec 2.0 / HuBERT | Masked frame prediction / 对比学习 | 丢弃预测头，encoder 各层输出的加权和 | 时间序列数据，变长输入，与 spiking 数据最相似 |
> | Multimodal | data2vec | 预测 masked 位置的 latent representations | 丢弃预测头，encoder 输出 | 跨模态统一的 masked prediction 范式 |
>
> **所有模态的共同模式**：预训练的重构/预测模块（decoder/prediction head）在下游被丢弃，只保留学到通用表征的 encoder。NeuroBridge 同理——Target Sequence Decoder 在 Phase 3 丢弃，只保留 SpikeTokenizer + PerceiverIO + Backbone。
>
> **MAE 的关键经验**：MAE 的 encoder 在预训练时只看 25% 的 patches，下游要处理 100% 的 patches，存在分布偏移。MAE 通过下游 fine-tuning 弥合这个偏移。NeuroBridge 面临同样的偏移（预训练时 PerceiverIO 收到部分 spike tokens，下游收到全部），但 PerceiverIO 的 cross-attention 天然适应变长输入，偏移影响应小于 ViT。Phase 3 Stage 2 的 encoder 微调可以进一步弥合。
>
> **NeuroBridge 的表征提取方案对比**：
>
> | 方案 | 做法 | 优点 | 缺点 | 推荐度 |
> |------|------|------|------|--------|
> | **A. Temporal Average Pooling** | Backbone 输出 [B, N_chunks, N_lat, D] → 选定时间窗口 → avg pool → [B, D] | 最简单，零额外参数 | pooling 丢失时间结构信息；pooling 操作未被预训练优化；需手动选择时间窗口 | ⭐⭐ 作为 baseline |
> | **B. Output Cross-Attention Readout（推荐）** | 新增 K 个 learnable readout queries，通过 cross-attention 从 backbone latent tokens 中提取表征 → [B, K, D] → flatten/pool → projector | 可学习的 pooling，自动关注最有信息量的时间段；与 POYO 的 output cross-attention 接口一致（torch_brain 现有实现）；readout queries 在 Phase 3 与 projector 一起训练 | 增加少量参数（K × D） | ⭐⭐⭐⭐ **推荐方案** |
> | **C. [CLS] Readout Token** | 预训练时在 latent 序列前加入 learnable [CLS] token，参与 backbone 所有 self-attention | 预训练阶段就开始学习全局表征 | [CLS] 在预训练时没有显式 loss 约束，可能学不到有用信息；需修改预训练 pipeline | ⭐⭐⭐ 作为消融对比 |
>
> **推荐方案 B 的完整数据流**：
>
> ```
> Phase 3 下游使用：
> spike tokens (全部，不mask)
>     → PerceiverIO (冻结) → latent tokens [B, N_chunks, N_lat, D]
>     → Backbone (冻结/最后4层微调) → contextualized latents [B, N_chunks, N_lat, D]
>     → Output Cross-Attention (K个 learnable readout queries)
>     → neural embedding [B, K, D]
>     → Flatten/Pool → [B, K*D 或 D]
>     → Neural Projector (MLP) → CLIP-aligned embedding [B, D_clip]
> ```
>
> 方案 B 的优势在于 readout queries 通过 cross-attention 可以**动态关注不同时间位置**，天然解决"不同脑区响应时间不同"的问题（V1: 30-80ms, V4: 60-120ms, IT: 80-200ms），不需要手动选择时间窗口。

#### Stage 1：Contrastive Alignment

- 冻结预训练encoder + 冻结CLIP
- 训练Neural Projector（3层MLP + LayerNorm + Dropout）+ Output Readout Queries（K个 learnable queries）
- 输入：readout queries 通过 cross-attention 从 backbone latent tokens 中提取的 neural embedding [B, K, D]
- 目标：与CLIP image embedding对齐（InfoNCE loss）
- 选择CLIP（而非DINOv2）因为SD的cross-attention层原本接收CLIP text encoder输出，用CLIP image embedding替代最为自然

#### Stage 2：Joint Fine-tuning

- 解冻encoder最后4层
- 联合优化：L_align + α · L_pretrain
- L_pretrain是anti-forgetting loss（继续masking重构），防止遗忘预训练知识
- 低学习率微调

#### Diffusion Adapter

- 将单个CLIP-aligned neural embedding [B, D_clip] 扩展为 [B, 77, 768] 的条件tensor
- 通过token expander（MLP）+ refiner（4层Transformer encoder）
- 输入Stable Diffusion UNet的cross-attention层
- 冻结SD，只训练adapter

#### 时间窗口选择

不同脑区的视觉响应时间不同：
- V1响应：stimulus onset后30-80ms
- V4响应：stimulus onset后60-120ms
- IT响应：stimulus onset后80-200ms
- 不同脑区应使用不同的时间窗口提取embedding

### 5.8 完整Loss设计总结

| 阶段 | Loss | 公式 | 说明 |
|------|------|------|------|
| 预训练 | Poisson NLL (masked grid) | L = Σ[λ - k·log(λ)] on masked positions | 主loss，在masked的neuron×time grid位置计算 |
| 对齐 Stage 1 | InfoNCE | L = -log(exp(sim(n,v+)/τ) / Σexp(sim(n,v)/τ)) | 冻结encoder，只训练projector |
| 对齐 Stage 2 | InfoNCE + Anti-forgetting | L_align + α · L_pretrain | 微调encoder最后4层 |
| Diffusion Adapter | DDPM Denoising | L = ‖ε - ε_θ(x_t, t, c)‖² | 冻结SD，只训练adapter |

---

## 六、实验设置 (Experimental Setup)

### 6.1 计算资源

> 💡 **待确认**: 以下计算资源配置请根据实际可用资源调整

- **预训练阶段**：4-8× NVIDIA A100 (80GB) 或等效GPU
- **对齐与重建阶段**：2-4× A100
- **总训练时间估计**：预训练~200 GPU-hours；对齐~50 GPU-hours；Diffusion Adapter~100 GPU-hours
- **存储**：~2TB用于数据集存储和checkpoint

### 6.2 软件环境

- **编程语言**：Python 3.10+
- **深度学习框架**：PyTorch 2.x
- **关键库**：
  - `torch_brain`（POYO/POYO+官方框架）
  - `transformers`（HuggingFace，用于CLIP模型）
  - `diffusers`（HuggingFace，用于Stable Diffusion）
  - `flash-attn`（FlashAttention 2）
  - `wandb`（实验tracking）
- **预训练模型**：
  - CLIP: `openai/clip-vit-large-patch14`
  - Stable Diffusion: `stabilityai/stable-diffusion-2-1` 或 SDXL

### 6.3 关键超参数

| 参数 | 预训练阶段 | 对齐阶段 |
|------|---------|--------|
| d_model | 512 | 512 (frozen) |
| d_tok (token dim) | 128 | - |
| n_layers | 12 | 最后4层微调 |
| n_heads | 8 | 8 |
| n_latent_per_chunk | 4 | - |
| chunk_size_ms | 50 | - |
| learning_rate | 3e-4 (cosine decay) | 1e-5 (Stage 2) |
| batch_size | 64-128 | 32-64 |
| optimizer | AdamW (weight_decay=0.01) | AdamW |
| d_clip | 768 | 768 |
| n_tokens (SD condition) | 77 | 77 |

### 6.4 评估指标

#### 预训练阶段

- **Masked reconstruction loss**：Poisson NLL on held-out data
- **Linear probe R²**：冻结encoder，训练线性层解码行为变量（速度等）
- **k-NN accuracy**：在表征空间上的最近邻分类准确率

#### 图像重建阶段

- **低层指标**：PixCorr（像素相关性）、SSIM（结构相似性）
- **高层指标**：
  - Retrieval accuracy（Top-1/Top-5 image retrieval）
  - CLIP similarity（重建图像与原始图像的CLIP embedding余弦相似度）
  - Inception Score
  - FID（Fréchet Inception Distance）
- **语义指标**：
  - Caption similarity（可选，使用BLIP-2生成caption后比较）

---

## 七、数据集 (Datasets)

### 7.1 数据集列表

| 数据集 | 物种 | 脑区 | 规模 | 包含视觉刺激 | 用途 |
|-------|------|------|------|----------|------|
| Allen Brain Observatory - Visual Coding | 小鼠 | V1, LM, AL, PM, AM, RL + 皮层下 | ~100,000神经元，多session | ✅ 自然图像/视频 | 预训练 + 图像重建 |
| TVSD | 猕猴 | V1, V4, IT | 2只猴，1024微电极，25,000+图像 | ✅ THINGS自然图像 | 图像重建（主要） |
| IBL Brain-Wide Map | 小鼠 | 多脑区（全脑） | 83只小鼠，10个实验室 | ⚠️ 简单视觉决策 | 预训练 |
| NLB MC_Maze | 猕猴 | 运动皮层 | 单session | ❌ | Baseline验证 |
| JiaLab实验室数据 | 猕猴 | 视觉皮层 | 多session，含image/video | ✅ | 预训练 + 图像重建 |

### 7.2 数据集详细描述

#### Allen Brain Observatory - Visual Coding（Neuropixels）

- **来源**：Allen Institute for Brain Science
- **记录方式**：Neuropixels探针（384通道硅探针，同时插入多个脑区）
- **物种**：小鼠（C57BL/6J野生型 + 多种转基因品系）
- **神经元数量**：近100,000个经过质量筛选的单元（across ~60个sessions）
- **脑区覆盖**：6个视觉皮层区域（V1、LM、AL、PM、AM、RL）+ 皮层下结构（LGN丘脑外侧膝状体、海马CA1/CA3、SC上丘等）
- **视觉刺激**：高度标准化的多种刺激范式：
  - **Natural Scenes（118张）**：来自Berkeley Segmentation Dataset和ImageNet子集的静态自然图像，每张呈现250ms，重复50次——这是图像重建的核心刺激
  - **Natural Movies（3段）**：30fps的自然场景视频片段（各30-120秒），重复10次
  - **Drifting Gratings**：8方向 × 5空间频率 × 5时间频率的参数化光栅刺激
  - **Static Gratings**：6方向 × 5空间频率 × 4相位
  - **Gabor patches**：局部化感受野映射刺激
  - **Full-field flashes**：全场闪烁用于响应特性表征
- **数据格式**：NWB（Neurodata Without Borders）标准格式，包含spike times、spike waveforms、LFP、running speed、pupil tracking、stimulus metadata
- **特点**：
  - 每个session同时记录多个脑区，天然支持inter-region masking预训练
  - 标准化的刺激呈现范式，所有session使用相同刺激，便于跨session对比
  - 提供详细的单元质量指标（SNR、ISI violation rate、presence ratio等）
  - Allen SDK提供便捷的数据访问API
  - POYO+已成功在此数据集上进行多脑区解码，证明了与torch_brain框架的兼容性
- **数据量**：每个session约2-6小时的连续记录，spike events约10^7-10^8量级
- **链接**：https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels
- **访问方式**：通过Allen SDK (`allensdk.brain_observatory.ecephys`) 或直接下载NWB文件

#### TVSD（The Ventral Stream Dataset）

- **来源**：Neuron, February 2025（Xu et al.）
- **记录方式**：1024微电极通道，分布在31个Utah 8×8阵列上，采集MUA（Multi-Unit Activity）信号，采样率30kHz
- **物种**：2只成年猕猴（Macaca mulatta）
- **脑区**：完整腹侧视觉通路——V1（初级视觉皮层，~300通道）、V4（~400通道）、IT（下颞叶皮层，~300通道）
- **视觉刺激**：
  - **THINGS数据库自然图像**：>25,000张覆盖1,854个物体概念类别的自然图像
  - 每张图像呈现时间：100-300ms（具体范式参见论文）
  - 每个图像至少重复2-5次，部分核心图像重复更多次
  - 图像覆盖广泛的语义类别（动物、工具、食物、场景等），语义丰富度极高
- **数据格式**：GIN repository存储，包含spike times/MUA signals + stimulus metadata + 电极位置信息
- **特点**：
  - **最大规模的猕猴视觉皮层spiking-image配对数据集**——25,000+张图像远超其他同类数据集
  - 同时记录V1→V4→IT完整腹侧通路，可研究视觉信息的层级处理
  - MonkeySee（2024）已在此数据集上验证了spiking数据图像重建的可行性，提供了直接的baseline对比
  - 图像来自THINGS数据库，有丰富的语义标注（概念层级、语义相似度矩阵等）
  - MUA信号比单单元记录更稳定，但空间分辨率稍低
- **数据量**：每只猴约12,000-15,000个trials，总数据量约50-100GB
- **链接**：https://doi.gin.g-node.org/10.12751/g-node.hc7zlv/
- **访问方式**：GIN CLI或Web下载

#### IBL Brain-Wide Map

- **来源**：International Brain Laboratory（国际脑实验室），多篇Nature系列论文
- **记录方式**：Neuropixels 1.0探针（384通道），每次插入1-2根探针
- **物种**：小鼠（C57BL/6J）
- **规模**：83只小鼠，来自10个实验室（跨4个国家），~550个recording sessions
- **脑区**：全脑范围（>200个脑区），包括视觉皮层、前额叶、纹状体、丘脑、中脑、小脑等
- **任务**：标准化的视觉决策任务（Biased Visual Decision Task）——小鼠通过转动方向盘将屏幕上出现的Gabor patch移动到中心
- **视觉刺激**：
  - Gabor patches：出现在左侧或右侧屏幕，对比度从0%到100%变化
  - 注意：**刺激语义信息极其有限**（仅简单Gabor patch），不适合图像重建任务
  - 但其丰富的行为变量（wheel velocity、lick times、choice、reward）适合表征质量评估
- **数据格式**：标准化NWB格式，通过ONE API统一访问
- **特点**：
  - **最大规模的标准化spiking数据集**——统一的实验范式、数据采集和处理流程
  - 全脑覆盖提供了跨脑区动力学学习的理想数据
  - MtM和NEDS均在此数据集上进行了预训练和评估
  - POYO+已验证在IBL数据上的解码性能
  - 标准化程度极高，非常适合作为大规模预训练数据（尽管刺激简单）
- **数据量**：每个session约1-2小时，总计约1000+小时的neural recording
- **链接**：https://int-brain-lab.github.io/
- **访问方式**：IBL ONE API (`ONE().load_object()`)

#### NLB MC_Maze（Neural Latents Benchmark）

- **来源**：Neural Latents Benchmark Challenge（Pei et al., NeurIPS 2021 Datasets Track）
- **记录方式**：Utah阵列（96通道）
- **物种**：猕猴（Macaca mulatta）
- **脑区**：初级运动皮层（M1）和背侧前运动皮层（PMd）
- **任务**：delayed reaching task（延迟抵达任务）——猕猴执行不同方向的手臂运动
- **视觉刺激**：❌ 无视觉刺激（运动任务）
- **数据格式**：标准化HDF5格式，提供binned spike counts和spike times
- **特点**：
  - POYO的原始benchmark数据集，velocity decoding R²≈0.87是标准baseline
  - 数据量小（单session，~30分钟），适合快速验证和调试
  - 作为torch_brain框架的标准测试用例，代码示例完善
  - **仅用于验证代码pipeline和baseline复现，不用于NeuroBridge的核心实验**
- **链接**：https://neurallatents.github.io/

#### JiaLab实验室数据

- **来源**：实验室内部数据（待确认具体参数）
- **记录方式**：多电极阵列（具体配置待确认）
- **物种**：猕猴
- **脑区**：视觉皮层（V1/V4/IT，具体覆盖待确认）
- **视觉刺激**：自然图像和/或视频
- **特点**：
  - 实验室内部数据，获取便利，可按需设计实验范式
  - 数据格式和预处理流程需要适配torch_brain格式
  - 多session记录，可用于跨session泛化性验证
  - **具体数据参数需与实验室确认后补充**

### 7.3 NeuroBridge项目数据集使用策略与推荐

> 💡 **核心原则**：优先保证**基础模型预训练**获得通用神经表征的数据需求，其次满足**图像重建**下游任务的数据需求。

#### 7.3.1 优先级一：基础模型预训练——学习通用神经表征

基础模型预训练的目标是从大规模、多样化的spiking数据中学习通用的神经动力学表征。理想的预训练数据应具备：**(a) 大规模**（神经元数量多、session数量多）、**(b) 多脑区覆盖**（支持inter-region masking）、**(c) 多样化的神经活动模式**。

**推荐的预训练数据集组合**：

| 优先级 | 数据集 | 角色 | 理由 |
|-------|-------|------|------|
| ⭐ 核心 | Allen Brain Observatory | 主要预训练数据 | 规模最大（~100K神经元）、多脑区同时记录（天然支持4种masking策略）、已验证与torch_brain兼容（POYO+） |
| ⭐ 核心 | IBL Brain-Wide Map | 大规模预训练补充 | 标准化程度最高、全脑覆盖、session数量最多（~550）、MtM/NEDS已验证其预训练价值 |
| 🔵 重要 | TVSD | 视觉皮层专项预训练 | V1/V4/IT完整腹侧通路、猕猴数据（物种多样性）、大量视觉响应数据 |
| 🔵 重要 | JiaLab数据 | 补充预训练数据 | 实验室数据可按需扩展 |

**预训练数据混合策略**：
- **Stage 1（单数据集热身）**：先在Allen Brain Observatory单独预训练，验证multi-task masking pipeline的有效性
- **Stage 2（跨数据集扩展）**：逐步加入IBL数据，验证跨物种/跨任务范式的泛化性
- **Stage 3（全量预训练）**：混合Allen + IBL + TVSD + JiaLab数据进行大规模预训练
- **数据采样策略**：参考Jiang et al.（2025）关于数据异质性的发现，按数据集规模的平方根比例采样（√N-sampling），避免大数据集主导训练

#### 7.3.2 优先级二：图像重建——从神经表征到视觉刺激

图像重建下游任务需要**spiking数据 + 对应视觉刺激图像**的配对数据。数据集的图像数量、图像多样性、以及记录脑区决定了重建质量的上限。

**推荐的图像重建数据集**：

| 优先级 | 数据集 | 角色 | 理由 |
|-------|-------|------|------|
| ⭐ 主要 | TVSD | 主要重建数据集 | 25,000+张自然图像（远超其他数据集）、V1/V4/IT完整通路、MonkeySee已提供baseline、THINGS语义标注丰富 |
| 🔵 辅助 | Allen Brain Observatory | 辅助重建数据集 | 118张Natural Scenes（重复50次，信噪比高）、多脑区信号可探究不同脑区贡献 |
| 🟡 探索 | JiaLab数据 | 扩展验证 | 如包含自然图像刺激，可作为独立验证集 |

**不适合图像重建的数据集**：
- **IBL Brain-Wide Map**：视觉刺激仅为简单Gabor patch，语义信息极为有限，不适合图像重建。但其预训练得到的通用表征可能有助于提升在其他数据集上的重建质量（迁移学习假设）
- **NLB MC_Maze**：无视觉刺激，仅用于baseline代码验证

#### 7.3.3 数据集与项目阶段的对应关系

| 项目阶段 | 使用数据集 | 目的 |
|---------|----------|------|
| Phase 0: 环境搭建 | NLB MC_Maze | 复现POYO baseline，验证pipeline |
| Phase 1a: 单任务masking | NLB MC_Maze → Allen (小规模子集) | 验证masking预训练基本功能 |
| Phase 1b: 多任务masking | Allen Brain Observatory | 多脑区数据支持所有4种masking策略 |
| Phase 1c: 大规模预训练 | Allen + IBL + TVSD | 混合数据大规模预训练 |
| Phase 2: 跨session泛化 | Allen（多session）+ IBL | 验证跨session迁移 |
| Phase 3a: CLIP对齐 | TVSD（主要）+ Allen Natural Scenes | 视觉刺激-神经活动配对数据 |
| Phase 3b: 图像重建 | TVSD（主要评估）+ Allen（辅助评估） | 重建质量评估 |
| Phase 4: 完整实验 | 全部数据集 | Scaling实验、消融实验 |

### 7.4 数据预处理

1. **Spike sorting**：使用Kilosort/MountainSort处理原始电生理数据（如有需要）
2. **Quality filtering**：去除低质量单元（firing rate < 0.5 Hz，ISI violation > 1%）
3. **Trial alignment**：对齐到视觉刺激onset
4. **Time windowing**：截取stimulus onset后的固定时间窗口（如0-500ms）
5. **Spike train格式化**：转换为 (unit_id, spike_time) 事件列表
6. **数据增强**：
   - Dynamic channel dropout（训练时随机dropout部分神经元）
   - 时间jittering（±1ms）

---

## 八、预期结果与实验 (Expected Results & Experiments)

### 8.1 实验计划

#### 实验1：预训练效果验证

**目标**：验证多任务masking预训练在spike-level tokenization + PerceiverIO架构上的有效性。

**设置**：
- 数据集：IBL Brain-Wide Map + Allen Visual Coding
- 评估：linear probe R²（解码行为变量）、masking reconstruction loss
- 对比条件：(a) 随机初始化 vs (b) 预训练encoder

**预期结果**：预训练encoder的linear probe R² > 随机初始化encoder至少10-15个百分点。

#### 实验2：多任务Masking消融实验

**目标**：验证不同masking模式的互补性。

**设置**：
- 单独使用每种masking模式训练
- 两两组合
- 全部四种模式联合训练
- 评估下游任务性能

**预期结果**：联合训练 > 任何单一或部分组合。

#### 实验3：跨Session泛化性测试

**目标**：评估模型在新session上的迁移能力。

**设置**：
- 在多session数据上预训练
- 在held-out session上评估（zero-shot / few-shot adaptation）
- 对比：(a) 从头训练 vs (b) 预训练+微调 vs (c) 预训练+冻结

**预期结果**：预训练模型通过少量adaptation即可达到从头训练的性能。

#### 实验4：图像重建质量评估

**目标**：评估从spiking数据重建图像的质量。

**设置**：
- 数据集：TVSD（主要）、Allen Visual Coding、JiaLab数据
- 对比基线：
  - MonkeySee（浅层映射）
  - Linear mapping to CLIP
  - 随机初始化encoder + CLIP对齐
  - 预训练encoder + CLIP对齐（本方法）
- 评估指标：PixCorr, SSIM, CLIP similarity, Retrieval accuracy, FID

**预期结果**：预训练encoder + CLIP对齐 > 随机初始化encoder + CLIP对齐 > MonkeySee。

#### 实验5：不同脑区对图像重建的贡献分析

**目标**：研究V1、V4、IT等不同脑区的神经活动对重建质量的影响。

**设置**：
- 单独使用各脑区数据进行重建
- 逐步添加脑区（V1 → V1+V4 → V1+V4+IT）
- 使用不同时间窗口优化各脑区的贡献

**预期结果**：
- IT区域提供最多的高层语义信息（类别、物体身份）
- V1区域提供低层细节（边缘、纹理）
- 多脑区联合 > 任何单一脑区

#### 实验6：Scaling特性研究

**目标**：研究数据规模、模型规模、任务规模对性能的影响。

**设置**：
- **Parameter scaling**：10M / 30M / 100M参数
- **Data scaling**：1/4 / 1/2 / 全部训练数据
- **Session scaling**：1 / 5 / 10 / all sessions
- 绘制scaling curves

**预期结果**：性能随规模增加呈log-linear增长，但可能存在异质性导致的diminishing returns。

#### 实验7：多模态Ablation实验

**目标**：验证pipeline各组件的贡献。

**设置**：
- (a) 无预训练，直接对齐
- (b) 预训练但不微调encoder
- (c) 预训练 + 微调（完整方法）
- (d) 去除anti-forgetting loss
- (e) 使用DINOv2替代CLIP
- (f) 不同时间窗口的影响

### 8.2 预期对比基线

| 基线方法 | 描述 |
|--------|------|
| MonkeySee | 浅层CNN映射 |
| Linear Probe → CLIP → SD | 线性映射到CLIP空间 |
| POYO encoder → CLIP → SD | POYO监督预训练的encoder |
| Random Init encoder → CLIP → SD | 随机初始化encoder |
| MindEye2 (fMRI) | 作为upper bound参考（不同模态） |

---

## 九、创新性与局限性讨论 (Innovation & Limitations)

### 9.1 创新性总结

1. **首次在spike-level tokenization + PerceiverIO架构上实现多任务masking自监督预训练**，解决了高时间分辨率保留与自监督预训练的兼容性问题
2. **首次将大规模自监督预训练的通用神经表征应用于spiking数据的图像重建**，填补了spiking数据领域在脑-图像重建方向的空白
3. **构建了从spike-level tokenization到CLIP对齐到Stable Diffusion生成的完整端到端pipeline**，为spiking数据的视觉解码提供了新范式
4. **系统性的scaling和泛化性研究**，为spiking数据基础模型的设计提供了重要的经验性指导

### 9.2 理论意义

- 验证了自监督预训练表征对于下游视觉解码任务的价值
- 提供了spiking数据与视觉语义空间之间的桥梁
- 揭示了不同脑区在视觉信息编码中的互补作用
- 为神经动力学的时间结构如何编码视觉信息提供了计算模型层面的证据

### 9.3 实际应用价值

- **脑机接口**：为视觉假体（visual prosthesis）提供解码框架
- **神经科学**：作为分析视觉皮层信息处理的工具
- **临床应用**：辅助理解视觉障碍患者的皮层活动

### 9.4 局限性

1. **数据获取门槛高**：侵入式spiking数据需要手术植入电极，数据获取成本远高于fMRI
2. **物种差异**：模型主要在小鼠和猕猴数据上训练，向人类数据的迁移面临额外挑战
3. **视觉刺激类型限制**：目前主要在静态图像上评估，视频/动态刺激的重建是更大的挑战
4. **Stable Diffusion的偏差**：生成模型本身的先验知识可能影响重建的忠实度
5. **评估指标的局限性**：现有图像质量指标可能无法完全反映神经信息的忠实编码

### 9.5 未来工作

1. **扩展到视频重建**：利用spiking数据的高时间分辨率优势，实现时间连续的视频重建
2. **双向pipeline**：不仅从神经活动重建图像，还从图像预测神经活动
3. **实时解码**：结合POSSM等循环架构，实现低延迟的在线图像重建
4. **人类数据**：将方法迁移到人类皮层记录数据（如BrainGate临床试验数据）
5. **多模态融合**：结合EEG/fMRI等非侵入式信号，构建多尺度脑-图像重建系统

---

## 十、参考文献 (References)

### 学术论文

1. Ye, J., et al. (2021). "Representation learning for neural population activity with Neural Data Transformers." *Neurons, Behavior, Data Analysis, and Theory*. arXiv:2108.01210

2. Ye, J., et al. (2025). "Neural Data Transformer 3: A Foundation Model for Intracortical Brain-Computer Interfaces." *ICLR 2025*. bioRxiv:10.1101/2025.02.02.634313

3. Trungle, et al. (2022). "Spatiotemporal Neural Data Transformer (STNDT)." *NeurIPS 2022*. arXiv:2206.04727

4. Azabou, M., et al. (2023). "A Unified, Scalable Framework for Neural Population Decoding (POYO)." *NeurIPS 2023*. arXiv:2310.16046

5. Azabou, M., et al. (2025). "Multi-session, multi-task neural decoding from the visual cortex (POYO+)." *ICLR 2025 (Spotlight)*. https://poyo-plus.github.io/

6. Hurwitz, C., et al. (2024). "Multi-task Masking for Neural Translators (MtM)." *NeurIPS 2024*. arXiv:2407.14668

7. Zhang, Y., et al. (2025). "Neural Encoding and Decoding at Scale (NEDS)." *ICML 2025 (Spotlight)*. arXiv:2504.08201

8. SPINT. (2025). "Permutation-Invariant Neural Transformer." *NeurIPS 2025*. arXiv:2507.08402

9. Scotti, P., et al. (2024). "MindEye2: Shared-Subject Models Enable fMRI-To-Image With 1 Hour of Data." *ICML 2024*. arXiv:2403.11207

10. MonkeySee. (2024). "Space-time-resolved reconstructions of natural images from macaque multi-unit activity." OpenReview.

11. TVSD Dataset. (2025). "An extensive dataset of spiking activity to reveal the syntax of the ventral stream." *Neuron*.

12. Jiang, P., et al. (2025). "Scaling Limitations of Neural Data Transformers." *bioRxiv*. 10.1101/2025.05.12.653551v1

13. Wang, et al. (2025). "A foundation model for the visual cortex." *Nature*, 640, 470-477.

14. CEBRA. (2023). "Learnable latent embeddings for joint behavioural and neural analysis." *Nature*.

15. LDNS. (2024). "Latent Diffusion for Neural Spiking Data." *NeurIPS 2024 (Spotlight)*. arXiv:2407.08751

16. POSSM. (2025). "Real-time neural decoding with hybrid state space models." *NeurIPS 2025*. arXiv:2506.05320

### 开源项目

- **torch_brain** (POYO/POYO+框架): https://github.com/neuro-galaxy/poyo
- **NEDS**: https://github.com/yzhang511/NEDS
- **MtM**: https://ibl-mtm.github.io/
- **NDT3**: https://github.com/joel99/ndt3
- **MindEye2**: https://medarc-ai.github.io/mindeye2/
- **Stable Diffusion**: https://github.com/Stability-AI/stablediffusion
- **CLIP**: https://github.com/openai/CLIP

### 数据集

- **Allen Brain Observatory**: https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels
- **TVSD**: https://doi.gin.g-node.org/10.12751/g-node.hc7zlv/
- **IBL Brain-Wide Map**: https://int-brain-lab.github.io/
- **NLB**: https://neurallatents.github.io/

---

## 十一、执行计划 (Execution Plan)

> 💡 **核心原则**：本执行计划以 **POYO/torch_brain 代码库**为基础，采用**增量式代码改进和实验验证**的策略。每个Phase在前一阶段的代码基础上进行最小化修改，确保每一步都有可验证的中间结果，降低技术风险。

### 开发策略总览

```
torch_brain (POYO原始代码)
    │
    ├── Phase 0: 原封不动运行 → 复现baseline
    │
    ├── Phase 1a: +Target Seq Decoder模块 → 最小masking预训练
    │       (新增文件: models/target_decoder.py, losses/poisson_nll.py)
    │       (修改文件: models/perceiver.py 增加decoder输出接口)
    │
    ├── Phase 1b: +Multi-Task Masking Controller → 多任务预训练
    │       (新增文件: masking/controller.py, masking/strategies.py)
    │       (修改文件: training/train_loop.py 增加masking模式切换)
    │
    ├── Phase 1c: +数据加载器适配 → 大规模多数据集预训练
    │       (新增文件: data/allen_loader.py, data/ibl_loader.py, data/tvsd_loader.py)
    │       (修改文件: data/dataset.py 增加multi-dataset sampling)
    │
    ├── Phase 2: +跨session适配模块 → 泛化性验证
    │       (修改文件: models/unit_embedding.py 增加adaptation逻辑)
    │
    ├── Phase 3a: +Neural Projector + CLIP对齐 → 表征对齐
    │       (新增文件: alignment/projector.py, alignment/infonce.py)
    │
    ├── Phase 3b: +Diffusion Adapter → 图像重建
    │       (新增文件: generation/diffusion_adapter.py)
    │
    └── Phase 4: 完整实验 + 论文
```

### Phase 0：环境搭建与POYO Baseline复现（2周）

**目标**：完整复现POYO原始代码的训练和评估流程，建立代码理解和开发基础。

**代码工作（基于torch_brain原始代码，零修改）**：
1. **环境配置**：
   - Clone `neuro-galaxy/poyo` 仓库，安装 `torch_brain` 及所有依赖
   - 配置 CUDA/FlashAttention 2 环境
   - 设置 wandb 实验tracking
2. **数据准备**：
   - 下载NLB MC_Maze数据集（HDF5格式，~100MB）
   - 运行torch_brain的标准数据预处理pipeline
   - 理解torch_brain的数据格式：`(unit_id, spike_time)` 事件列表 → spike token序列
3. **Baseline训练与评估**：
   - 运行POYO的MC_Maze监督训练脚本（原始配置）
   - 评估velocity decoding R²
   - 可视化PerceiverIO的latent representations（t-SNE/UMAP）
4. **代码深度阅读**（关键模块）：
   - `torch_brain/models/perceiver.py`：PerceiverIO的cross-attention压缩实现
   - `torch_brain/models/units.py`：unit embedding的实现方式
   - `torch_brain/data/`：数据加载和tokenization pipeline
   - `torch_brain/training/`：训练循环和loss计算

**数据集**：NLB MC_Maze（小规模，快速迭代）

**验收标准**：
- MC_Maze velocity decoding R² ≈ 0.87（复现POYO论文结果）
- 训练pipeline无内存泄漏，GPU利用率正常
- 完成代码结构文档（记录关键类/函数的接口和数据流）

**实验记录**（wandb）：
- `poyo-baseline/mc_maze_reproduce`：baseline复现实验

---

### Phase 1a：Target Sequence Decoder + 单任务Masking（3周）

**目标**：在POYO架构上增加masking预训练能力，验证最基本的自监督预训练信号。

**核心思路**：保持POYO的SpikeTokenizer + PerceiverIO完全不变，在PerceiverIO输出端新增一个Target Sequence Decoder用于重构预测。

**代码增量修改**：

1. **新增 `neurobridge/decoders/target_seq_decoder.py`**：
   - 实现neuron × time grid的2D query生成
   - Cross-attention: grid queries attend to PerceiverIO latent tokens
   - 输出: predicted firing rates [B, N_neurons, T_bins]
   - 关键参数: `time_bin_ms=10`, `n_query_layers=2`

2. **新增 `neurobridge/losses/poisson_nll.py`**：
   - Poisson NLL loss计算（仅在masked positions上）
   - 支持mask tensor输入
   - 数值稳定性处理（clamp λ > 0）

3. **新增 `neurobridge/masking/temporal_masking.py`**：
   - 最简单的random temporal masking（Phase 0的增量）
   - 在spike token层面，按时间窗口mask掉一定比例的spike tokens（Pre-PerceiverIO）
   - Masking比例: 50%（固定）

4. **修改 `torch_brain/models/perceiver.py`**（最小化修改）：
   - 在PerceiverIO输出端增加hook，暴露latent tokens给外部decoder
   - 不改变原始前向传播逻辑

5. **新增训练脚本 `scripts/pretrain_single_mask.py`**：
   - 基于POYO原始训练脚本修改
   - 训练流程: spike tokens → (temporal masking) → PerceiverIO → latent tokens → Target Seq Decoder → firing rate predictions → Poisson NLL loss

**实验计划**：

| 实验 | 数据集 | 目的 | 预期结果 |
|------|-------|------|---------|
| Exp 1.1 | MC_Maze | 验证decoder能否重构firing rate | Poisson NLL持续下降 |
| Exp 1.2 | MC_Maze | 预训练→linear probe vs 随机初始化→linear probe | 预训练R² > 随机初始化 |
| Exp 1.3 | Allen (5 sessions子集) | 在视觉数据上验证masking预训练 | Poisson NLL下降，表征有结构 |

**数据集**：NLB MC_Maze（验证）→ Allen Brain Observatory 小规模子集（5 sessions，过渡到视觉数据）

**验收标准**：
- MC_Maze上 masking预训练 → linear probe R² > 0.75
- Allen子集上 Poisson NLL持续下降（训练稳定）
- 预训练表征的t-SNE可视化展现时间/脑区结构

---

### Phase 1b：Multi-Task Masking Controller（3周）

**目标**：从单一temporal masking扩展到完整的4种masking策略，验证多任务masking的互补性。

**代码增量修改**：

1. **新增 `neurobridge/masking/controller.py`**：
   - Multi-Task Masking Controller主控制器
   - 每个training step随机选择一种masking模式（或按概率混合）
   - 接口: `controller.apply_mask(spike_tokens, mode=None)` → masked_tokens, mask_indices

2. **新增 `neurobridge/masking/strategies.py`**：
   - `TemporalMasking`: 按时间窗口mask spike tokens（已有，重构为标准接口）
   - `NeuronMasking`: dropout 30-50%神经元的所有spike tokens
   - `IntraRegionMasking`: 在每个脑区内随机mask部分神经元（需要脑区metadata）
   - `InterRegionMasking`: mask整个脑区的所有神经元

3. **（可选，消融对比用）新增 `neurobridge/masking/prompt_tokens.py`**：
   - 4个learnable prompt tokens（每种masking模式一个）
   - 默认不使用——模型从缺失 pattern 自行推断 masking 模式（参见 5.5 节设计备注）
   - 仅作为消融实验的对比条件

4. **修改训练脚本**：
   - 支持masking模式的随机切换
   - 支持消融实验配置（单模式/组合模式/全部模式、有/无 prompt token）
   - wandb logging各模式的loss分别记录

**实验计划**：

| 实验 | 数据集 | 配置 | 目的 |
|------|-------|------|------|
| Exp 2.1 | Allen (5 sessions) | 仅temporal masking | 单模式baseline |
| Exp 2.2 | Allen (5 sessions) | 仅neuron masking | 单模式baseline |
| Exp 2.3 | Allen (5 sessions) | 仅intra-region masking | 单模式baseline |
| Exp 2.4 | Allen (5 sessions) | 仅inter-region masking | 单模式baseline |
| Exp 2.5 | Allen (5 sessions) | 4种模式联合（无 prompt token） | 验证互补性（默认方案） |
| Exp 2.6 | Allen (5 sessions) | 两两组合（6种） | 组合效果分析 |
| Exp 2.7 | Allen (5 sessions) | 4种模式联合 + prompt tokens | 消融：prompt token 有无的影响 |

**数据集**：Allen Brain Observatory（5-10 sessions子集，多脑区数据天然支持region masking）

**验收标准**：
- 联合训练的linear probe R² > 任何单一masking模式
- 各masking模式的loss曲线合理（无某种模式主导/退化）

---

### Phase 1c：大规模多数据集预训练（4周）

**目标**：在多数据集上进行大规模预训练，获得通用的神经表征。

**代码增量修改**：

1. **新增数据加载器**：
   - `neurobridge/data/allen_loader.py`: Allen Brain Observatory NWB数据 → torch_brain格式
     - 使用 `allensdk.brain_observatory.ecephys` API
     - 提取spike times + unit metadata（脑区、quality metrics）
     - 标准化为 `(unit_id, spike_time, brain_region)` 格式
   - `neurobridge/data/ibl_loader.py`: IBL ONE数据 → torch_brain格式
     - 使用 IBL ONE API
     - 处理多探针、多脑区数据
   - `neurobridge/data/tvsd_loader.py`: TVSD数据 → torch_brain格式
     - 处理MUA信号，提取spike events
     - 同时加载对应的视觉刺激图像路径（为Phase 3准备）

2. **修改 `neurobridge/data/multi_dataset.py`**：
   - 多数据集混合采样器（√N-sampling策略）
   - 数据集级别的标识token（dataset embedding）
   - 动态batch组装：同一batch内可混合不同数据集

3. **Curriculum Learning调度器**：
   - Week 1: 仅Allen数据（~20 sessions）
   - Week 2: Allen + IBL（逐步增加IBL比例）
   - Week 3-4: Allen + IBL + TVSD（全量混合训练）

**实验计划**：

| 实验 | 数据规模 | 模型规模 | 目的 |
|------|---------|---------|------|
| Exp 3.1 | Allen 全量 (~60 sessions) | ~30M参数 | 单数据集大规模预训练 |
| Exp 3.2 | Allen + IBL | ~30M参数 | 跨物种/跨任务预训练 |
| Exp 3.3 | Allen + IBL + TVSD | ~30M参数 | 全量混合预训练 |
| Exp 3.4 | Allen + IBL + TVSD | ~100M参数 | Parameter scaling |
| Exp 3.5 | 数据量消融 (1/4, 1/2, 全量) | ~30M参数 | Data scaling curve |

**数据集**：Allen Brain Observatory（全量） + IBL Brain-Wide Map + TVSD

**验收标准**：
- 混合预训练的linear probe R² > 单数据集预训练
- 表征空间展现有意义的跨数据集结构（不同数据集/脑区可区分但共享子空间）
- Scaling curve呈现正向趋势

---

### Phase 2：跨Session泛化性验证（3周）

**目标**：验证预训练模型在未见过的session上的迁移能力。

**代码增量修改**：

1. **修改 `torch_brain/models/units.py`**：
   - 增加unit embedding的few-shot adaptation模式
   - 冻结模型主体，仅通过梯度下降更新新session的unit embeddings
   - 实现adaptation训练循环（~100-500步即可收敛）

2. **（可选）新增 `neurobridge/models/id_encoder.py`**：
   - 借鉴SPINT的IDEncoder设计
   - 从校准数据（calibration trials）通过前向传播生成unit embeddings
   - Gradient-free迁移：无需额外训练步骤

**实验计划**：

| 实验 | 数据集 | 配置 | 目的 |
|------|-------|------|------|
| Exp 4.1 | Allen held-out sessions | Zero-shot (随机unit emb) | 迁移下界 |
| Exp 4.2 | Allen held-out sessions | Few-shot adaptation (100步) | 快速适配 |
| Exp 4.3 | Allen held-out sessions | Few-shot adaptation (500步) | 充分适配 |
| Exp 4.4 | Allen held-out sessions | 从头训练 (full training) | 迁移上界 |
| Exp 4.5 | TVSD held-out session | Few-shot adaptation | 跨物种迁移 |

**数据集**：Allen Brain Observatory（留出5个session作为held-out） + TVSD（留出1个session）

**验收标准**：
- Few-shot adaptation (500步) 达到从头训练性能的90%+
- Zero-shot迁移优于随机初始化

---

### Phase 3a：Neural-to-CLIP对齐（3周）

**目标**：将预训练的神经表征映射到CLIP视觉语义空间。

**代码增量修改**：

1. **新增 `neurobridge/alignment/projector.py`**：
   - Neural Projector: 3层MLP + LayerNorm + Dropout
   - 输入: readout queries 经 cross-attention 从 backbone 提取的 [B, K, D] → flatten → [B, D_proj] → [B, D_clip=768]
   - readout queries 自动学习关注最有信息量的时间段和 latent 维度

2. **新增 `neurobridge/alignment/infonce.py`**：
   - InfoNCE contrastive loss实现
   - 支持多GPU的all_gather（扩大negative样本池）
   - Temperature parameter τ可学习

3. **新增 `neurobridge/alignment/clip_wrapper.py`**：
   - CLIP模型封装（冻结参数）
   - 图像预处理pipeline
   - 支持提取intermediate layer features（备用方案）

4. **Joint Fine-tuning模块**：
   - 解冻encoder最后4层
   - Anti-forgetting loss: 继续在masking重构任务上计算Poisson NLL
   - 总loss: L_total = L_infonce + α · L_poisson_nll (α=0.1-0.5)

**实验计划**：

| 实验 | 数据集 | 配置 | 目的 |
|------|-------|------|------|
| Exp 5.1 | TVSD | Stage 1: 冻结encoder + projector | Contrastive对齐baseline |
| Exp 5.2 | TVSD | Stage 2: 解冻encoder后4层 | Joint fine-tuning |
| Exp 5.3 | TVSD | 无预训练encoder + projector | 验证预训练价值 |
| Exp 5.4 | Allen Natural Scenes | Stage 1 + Stage 2 | 跨数据集对齐验证 |
| Exp 5.5 | TVSD (各脑区分别) | Stage 1 | V1 vs V4 vs IT对齐效果对比 |

**数据集**：TVSD（主要，图像最丰富） + Allen Brain Observatory Natural Scenes（辅助）

**验收标准**：
- CLIP retrieval Top-5 accuracy > 30%（Stage 1）/ > 50%（Stage 2）
- 预训练encoder对齐效果显著优于随机初始化encoder

---

### Phase 3b：Diffusion Adapter与图像重建（3周）

**目标**：通过Stable Diffusion从CLIP-aligned neural embeddings生成重建图像。

**代码增量修改**：

1. **新增 `neurobridge/generation/diffusion_adapter.py`**：
   - Token Expander: MLP将 [B, D_clip] → [B, 77, 768]
   - Refiner: 4层Transformer encoder精炼token序列
   - 接口与Stable Diffusion UNet的cross-attention层对接

2. **新增 `neurobridge/generation/sd_wrapper.py`**：
   - Stable Diffusion封装（冻结所有参数）
   - 替换text encoder输出为neural adapter输出
   - 支持DDIM采样加速（50步）

3. **新增评估脚本 `scripts/evaluate_reconstruction.py`**：
   - 批量生成重建图像
   - 计算全部评估指标：PixCorr, SSIM, CLIP similarity, Retrieval accuracy, FID

**实验计划**：

| 实验 | 数据集 | 配置 | 目的 |
|------|-------|------|------|
| Exp 6.1 | TVSD | 预训练encoder + CLIP + SD | 完整pipeline评估 |
| Exp 6.2 | TVSD | 随机encoder + CLIP + SD | 验证预训练价值 |
| Exp 6.3 | TVSD | Linear mapping + CLIP + SD | MonkeySee增强版baseline |
| Exp 6.4 | Allen Natural Scenes | 完整pipeline | 跨数据集泛化 |
| Exp 6.5 | TVSD (V1 only / V4 only / IT only) | 完整pipeline | 脑区贡献分析 |
| Exp 6.6 | TVSD (V1+V4 / V1+IT / V4+IT / V1+V4+IT) | 完整pipeline | 多脑区联合效果 |

**数据集**：TVSD（主要评估） + Allen Natural Scenes（辅助评估）

**验收标准**：
- TVSD上重建图像在视觉上可辨识对应的原始图像
- 预训练encoder + CLIP + SD > 随机encoder + CLIP + SD > Linear mapping
- 多脑区联合 > 任何单一脑区

---

### Phase 4：完整实验与论文撰写（5周）

**Week 1-2：补充实验**
- Scaling实验（Exp 3.4, 3.5的完整版本）
- 多模态Ablation实验（去除各组件的效果）
- 时间窗口优化实验
- 与MindEye2（fMRI baseline）的横向对比分析
- 结果可视化（重建图像gallery、attention可视化、表征空间可视化）

**Week 3-5：论文撰写**
- 撰写初稿（按NeurIPS/ICML格式）
- 制作figures和tables
- 内部审阅和修改
- 最终定稿

### 里程碑与检查点

| 里程碑 | 时间节点 | 验收标准 | 关键数据集 |
|-------|--------|---------|----------|
| M0: POYO Baseline复现 | Week 2 | MC_Maze R² ≈ 0.87 | NLB MC_Maze |
| M1: 单任务masking预训练 | Week 5 | Allen子集上Poisson NLL下降，linear probe R² > 0.75 | MC_Maze → Allen子集 |
| M2: 多任务masking完成 | Week 8 | 联合训练R² > 单任务 | Allen (5-10 sessions) |
| M3: 大规模预训练完成 | Week 12 | 混合预训练R² > 单数据集 | Allen + IBL + TVSD |
| M4: 跨session验证 | Week 15 | Few-shot达到从头训练90%+ | Allen held-out + TVSD |
| M5: CLIP对齐完成 | Week 18 | CLIP retrieval Top-5 > 50% | TVSD + Allen |
| M6: 图像重建初步结果 | Week 21 | 重建图像视觉上可辨识 | TVSD |
| M7: 完整实验结果 | Week 23 | 所有实验完成 | 全部 |
| M8: 论文定稿 | Week 26 | 可提交版本 | - |

### 代码仓库结构规划

```
neurobridge/
├── configs/                    # 实验配置文件（yaml）
│   ├── pretrain_single_mask.yaml
│   ├── pretrain_multi_mask.yaml
│   ├── pretrain_large_scale.yaml
│   ├── alignment_stage1.yaml
│   ├── alignment_stage2.yaml
│   └── diffusion_adapter.yaml
├── data/                       # 数据加载器（增量新增）
│   ├── allen_loader.py         # Allen Brain Observatory数据适配
│   ├── ibl_loader.py           # IBL数据适配
│   ├── tvsd_loader.py          # TVSD数据适配（含图像路径）
│   └── multi_dataset.py        # 多数据集混合采样
├── masking/                    # Masking策略（全新模块）
│   ├── controller.py           # Multi-Task Masking Controller
│   ├── strategies.py           # 4种masking策略实现
│   └── prompt_tokens.py        # Learnable prompt tokens（可选，消融对比用）
├── decoders/                   # 解码器（全新模块）
│   └── target_seq_decoder.py   # Target Sequence Decoder
├── alignment/                  # CLIP对齐（全新模块）
│   ├── projector.py            # Neural Projector (MLP)
│   ├── infonce.py              # InfoNCE loss
│   └── clip_wrapper.py         # CLIP模型封装
├── generation/                 # 图像生成（全新模块）
│   ├── diffusion_adapter.py    # Diffusion Adapter
│   └── sd_wrapper.py           # Stable Diffusion封装
├── losses/                     # Loss函数
│   └── poisson_nll.py          # Poisson NLL loss
├── models/                     # 模型修改（基于torch_brain增量修改）
│   └── id_encoder.py           # （可选）IDEncoder
├── scripts/                    # 训练/评估脚本
│   ├── pretrain_single_mask.py
│   ├── pretrain_multi_mask.py
│   ├── pretrain_large_scale.py
│   ├── train_alignment.py
│   ├── train_diffusion_adapter.py
│   └── evaluate_reconstruction.py
└── utils/                      # 工具函数
    ├── visualization.py        # t-SNE/UMAP/重建图像可视化
    └── metrics.py              # 评估指标计算
```

### 风险评估与应对方案

1. **技术风险**：spike-level masking预训练可能收敛困难
   - **对策**：Phase 1a从最简单的temporal masking开始，逐步增加复杂度；准备binned representation作为fallback；MC_Maze上先验证小规模可行性再扩展
   - **降级方案**：如果spike-level masking效果不佳，回退到NDT风格的binned representation + masking

2. **数据风险**：TVSD数据获取或格式适配困难
   - **对策**：Phase 0-1使用Allen和IBL数据（已有torch_brain适配先例）进行预训练，TVSD适配与预训练并行推进
   - **降级方案**：以Allen Brain Observatory Natural Scenes作为主要图像重建数据集（118张图像，重复50次，信噪比高）

3. **数据加载器开发风险**：不同数据集格式差异大
   - **对策**：Allen/IBL均已有POYO+的处理先例，可参考其代码；TVSD优先处理核心字段（spike times + stimulus ID）；每个loader独立开发和测试
   - **降级方案**：优先保证Allen loader完成（POYO+已验证），IBL和TVSD延后

4. **对齐风险**：neural representation与CLIP空间对齐效果不佳
   - **对策**：先用Allen Natural Scenes（少量但高信噪比数据）快速验证对齐可行性；尝试DINOv2作为替代；增加projector网络容量
   - **降级方案**：使用intermediate CLIP layers而非最终embedding；增加可训练参数

5. **Scaling风险**：数据异质性限制scaling收益
   - **对策**：采用Curriculum Learning + √N-sampling；参考Jiang et al.的发现；每加入新数据集都单独评估对性能的影响
   - **降级方案**：如果混合效果不佳，回退到单数据集（Allen）预训练，仅在图像重建时使用TVSD

6. **计算资源风险**：大规模预训练所需GPU时间超出预算
   - **对策**：Phase 1c的大规模预训练从小模型（~10M）开始，确认趋势后再扩展；使用mixed precision (fp16/bf16) + gradient accumulation减少显存需求
   - **降级方案**：缩小模型规模（~30M而非100M），缩短预训练epoch数

---

## 附录

### A. 项目命名

**NeuroBridge** — 寓意为连接神经活动（Neural activity）与视觉世界（Visual world）的桥梁。

### B. 与先前方案的差异说明

本研究计划基于2026年2月8日版本的研究方案进行了以下关键更新（以最新构想为准）：

1. **Masking策略**：从Grid-Query Decoder方案更新为在tokenization前进行masking的方案
2. **跨Session方案**：从强制使用IDEncoder更新为以POYO可扩展unit embedding为主、IDEncoder为可选增强
3. **Target预测**：通过cross-attention在固定time bins上预测spike counts
4. **下游任务**：更明确地定义了图像重建pipeline的三个阶段

### C. 参考代码库

- torch_brain官方示例：https://github.com/neuro-galaxy/poyo
- MindEye2官方实现：https://github.com/MedARC-AI/MindEye2
- Stable Diffusion Hugging Face：https://huggingface.co/stabilityai/stable-diffusion-2-1

---

*本研究计划为初版，将随着研究进展和实验结果持续更新。*
