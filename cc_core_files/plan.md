# NeuroBridge 项目分析与执行计划

**分析日期**：2026-02-21
**分析者**：Claude Code
**项目仓库**：`/root/autodl-tmp/NeuroBridge`

---

## 1. 项目概述

NeuroBridge 旨在构建一个从 spiking/MUA 神经数据到图像重建的完整 pipeline：
1. **自监督预训练**：在 POYO 的 spike-level tokenization + PerceiverIO 架构上实现 MtM 风格的多任务 masking 预训练
2. **视觉对齐**：将预训练神经表征通过对比学习映射到 CLIP 视觉语义空间
3. **图像重建**：通过 Diffusion Adapter 注入 Stable Diffusion 生成重建图像

**目标发表**：NeurIPS / ICML / Nature Methods

---

## 2. Proposal 合理性审查

### 2.1 问题清单

| # | 严重程度 | 问题描述 | 解决方案 | 状态 |
|---|---------|---------|---------|------|
| 1 | 🔴 严重 | TVSD 提供 MUA（连续信号），不是 spike-sorted 数据，与 POYO 的 spike-level tokenization 不兼容 | 采用 CaPOYO 模式：`input_value_map = nn.Linear(1, dim//2)` 映射连续值 + unit_emb 拼接 | 方案已确定 |
| 2 | 🔴 严重 | Proposal 提出 d_model=512, depth=12（~100M params），当前 GPU 为 RTX 4090 24GB，无法训练 | 从 dim=128, depth=6 (~5M params) 起步，验证后逐步扩大 | 方案已确定 |
| 3 | 🟡 中等 | Poisson NLL 假设离散计数，不适用于 MUA 连续信号 | 对 spike count 用 Poisson NLL，对 MUA 用 MSE | 方案已确定 |
| 4 | 🟡 中等 | Allen 仅 118 张图不足以做图像重建主力 | TVSD 是核心（22K 张），Allen 仅作辅助验证 | proposal 已识别 |
| 5 | 🟢 低 | IBL 预训练对视觉重建的迁移价值未验证 | 在单卡资源下暂不整合 IBL，专注 Allen + TVSD | 方案已确定 |
| 6 | 🟢 低 | 项目范围过大（4 创新点 + 7 实验 + 3 数据源） | 削减到最小可发表单元 | 方案已确定 |

### 2.2 Proposal 优点

1. **研究空白识别准确**：确实没有在 spike-level tokenization + PerceiverIO 上做 masking 预训练 + 图像重建的工作
2. **技术方案设计周到**：Pre-PerceiverIO masking、不使用 prompt token、neuron×time grid 重建等设计决策有充分理由
3. **Target Sequence Decoder 设计合理**：避免了 spike token 级别 masking 的信息泄露问题
4. **两阶段视觉对齐策略成熟**：借鉴 MindEye2 的 Contrastive + Diffusion 路线

### 2.3 需要修正的声明

- "spike-level tokenization" → 应改为 "event-level tokenization with continuous value support"（因为 TVSD 用的是 MUA，不是 spike）
- "d_model=512, 12 layers" → 实际起步 dim=128, depth=6，论文中体现 scaling 实验
- Proposal 中的 "4-8× A100" → 实际用单卡 RTX 4090 24GB，需要调整所有 batch size 和 gradient accumulation

---

## 3. 关键架构决策

### 决策 1：MUA 数据处理方式

**选择**：CaPOYO-style 连续值 tokenization

```python
# 参考 capoyo.py:68
self.input_value_map = nn.Linear(1, dim // 2)  # MUA 值 → 半维度嵌入
self.unit_emb = InfiniteVocabEmbedding(dim // 2)  # 电极 ID → 半维度嵌入
# 拼接为完整 token: [value_emb || unit_emb] → dim 维

# 对于 spike-sorted 数据 (Allen/IBL): value = 1.0
# 对于 MUA 数据 (TVSD): value = MUA amplitude
```

**理由**：CaPOYO 已验证此方案可处理钙成像连续信号，直接复用代码。

### 决策 2：模型规模

**选择**：dim=128, depth=6 (~5M params)

**理由**：RTX 4090 24GB 在 AMP 下可训练此规模。POYO 已在 dim=64 和 dim=128 上验证。

### 决策 3：重建 Loss

**选择**：
- Spike count targets (Allen): Poisson NLL
- MUA targets (TVSD): MSE Loss

**理由**：匹配数据统计特性。`torch_brain/nn/loss.py` 已有 MSE 实现。

### 决策 4：Grid 时间分辨率

**选择**：50ms（= latent_step 的一半或等于）

**理由**：10ms grid 对 100 个神经元产生 10,000 个 query positions，计算量过大。50ms 与 backbone 压缩粒度接近，更合理。

### 决策 5：数据优先级

**选择**：TVSD（核心）> Allen（预训练验证）>> IBL（暂不用）

**理由**：单卡资源有限，TVSD 是图像重建的唯一可靠数据源。

---

## 4. 执行计划

### 4.0 当前资源约束

| 资源 | 实际情况 | 影响 |
|------|---------|------|
| GPU | RTX 4090 D 24GB | 模型 ≤ 30M params，batch_size ≤ 64，需 AMP |
| 存储 | /root/autodl-tmp/ | 需确认可用空间（TVSD MUA ~200-500GB） |
| TVSD | 未下载 | 关键路径，需立即启动 |
| JiaLab 数据 | 不可用 | 从计划中移除 |

### 4.1 Phase 0: 环境搭建 + 数据启动

**目标**：复现 POYO baseline + 启动数据下载

| 子任务 | 描述 | 验收标准 |
|--------|------|---------|
| 0.1 | 安装 torch_brain 及依赖 | `import torch_brain` 成功 |
| 0.2 | 下载 NLB MC_Maze 数据 | 数据文件存在且可加载 |
| 0.3 | 运行 POYO baseline 训练 | velocity R² ≈ 0.87 |
| 0.4 | 启动 TVSD datalad clone | 仓库克隆成功 |
| 0.5 | 下载 Allen 前 5 sessions | NWB 文件可加载 |

### 4.2 Phase 1a: TVSD 数据适配 + 统一 Tokenizer

**目标**：TVSD MUA 数据能通过 CaPOYO-style 前向传播

**新增文件**：
- `neurobridge/data/tvsd_loader.py`
- `neurobridge/data/tvsd_dataset.py`

**GO/NO-GO 决策点 1**：CaPOYO tokenization 是否有效处理 TVSD MUA？

### 4.3 Phase 1b: Masking 预训练

**目标**：在 Allen 上验证 masking 预训练产生有意义的表征

**新增文件**：
- `neurobridge/decoders/target_seq_decoder.py`
- `neurobridge/masking/strategies.py` (temporal + neuron)
- `neurobridge/masking/controller.py`
- `neurobridge/losses/reconstruction_loss.py`

**模型配置**：
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

**验收标准**：
- 重建 loss 持续下��
- Linear probe R² > 随机初始化

**GO/NO-GO 决策点 2**：预训练是否产生有意义的表征？

### 4.4 Phase 2: CLIP 对齐 + 图像重建

**目标**：从 TVSD MUA 数据重建可辨识的图像

**新增文件**：
- `neurobridge/alignment/readout.py`（K=8 learnable queries）
- `neurobridge/alignment/projector.py`（3 层 MLP → 768 维 CLIP 空间）
- `neurobridge/alignment/infonce.py`
- `neurobridge/alignment/clip_wrapper.py`
- `neurobridge/generation/diffusion_adapter.py`
- `neurobridge/generation/sd_wrapper.py`
- `scripts/evaluate_reconstruction.py`

**两阶段对齐**：
1. Stage 1: 冻结 encoder + 冻结 CLIP，训练 readout + projector (InfoNCE)
2. Stage 2: 解冻 encoder 最后 2-3 层，联合 InfoNCE + anti-forgetting loss

**验收标准**：
- CLIP retrieval Top-5 > 15% (Stage 1) / > 30% (Stage 2)
- 重建图像视觉上可辨识

**GO/NO-GO 决策点 3**：重建图像是否有意义？

### 4.5 Phase 3: 消融实验

**优先级**（从高到低）：
1. ✅ 必做：预训练 vs 随机初始化 encoder
2. ✅ 必做：V1 vs V4 vs IT vs V1+V4+IT 脑区贡献
3. ✅ 必做：temporal masking vs neuron masking vs combined
4. 🔵 可做：不同时间窗口对各脑区的效果
5. 🔵 可做：dim=128 vs dim=256 scaling
6. ⬜ 延后：跨 session 泛化
7. ⬜ 延后：IBL 数据混合

### 4.6 Phase 4: 论文撰写

- 结果可视化
- NeurIPS/ICML 格式初稿
- 审阅修改

---

## 5. 风险登记表

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|---------|
| TVSD 下载耗时/失败 | 高 | 阻塞图像重建 | 立即启动；用 Allen 118 张图做原型 |
| 24GB VRAM 不够 | 中 | 限制模型/batch 大小 | AMP + gradient checkpoint + accumulation |
| MUA tokenization 效果差 | 低-中 | TVSD 不可用 | 回退到 binned representation |
| 预训练对重建无帮助 | 中 | 核心 claim 不成立 | 改发 "direct alignment" 论文 |
| CLIP 对齐失败 | 低-中 | 无重建结果 | 尝试 DINOv2；增加 projector 容量 |

---

## 6. 最小可发表论文方案

### 方案 A（最优）：完整 NeuroBridge
- 预训练（Allen + TVSD） + CLIP 对齐（TVSD） + 图像重建（TVSD）
- 消融实验证明预训练价值 + 脑区贡献分析
- 目标：NeurIPS / ICML

### 方案 B（降级）：Masked Pretraining Only
- 如果图像重建效果不佳，仅发表预训练组件
- 在 Allen/IBL 上验证 masking 预训练的表征质量
- 目标：Workshop / ICLR

### 方案 C（最小）：Direct Neural-to-CLIP Alignment
- 跳过预训练，直接在 TVSD 上做 MUA → CLIP 对齐
- 与 MonkeySee baseline 对比
- 目标：计算神经科学 venue

---

## 7. 进度记录

| 日期 | 阶段 | 完成内容 | 遇到的问题 | 解决方案 |
|------|------|---------|-----------|---------|
| 2026-02-21 | 项目启动 | ✅ POYO 代码分析完成 | PyTorch 环境未安装 | 待安装 |
| 2026-02-21 | 项目启动 | ✅ NeuroBridge proposal 审查完成 | GPU 为 4090 24GB 而非 A100 80GB | 调整模型配置为 dim=128 |
| 2026-02-21 | 项目启动 | ✅ 执行计划制定完成 | TVSD 未下载 | 需立即启动 |
| 2026-02-21 | 项目启动 | ✅ 分析文档保存到 cc_todo | - | - |
| | | | | |

---

*本文档将随项目进展持续更新。*
