# NeuroBridge 项目进展总结

**日期**：2026-02-21
**项目路径**：`/root/autodl-tmp/NeuroBridge`

---

## 一、项目目标

NeuroBridge 旨在构建从猕猴视觉皮层 MUA（多单元活动）神经信号到图像重建的完整 Pipeline：

1. **神经编码器**：基于 POYO/CaPOYO 的 PerceiverIO 架构，将 1024 通道 MUA 信号编码为紧凑表征
2. **自监督预训练**：通过电极 masking 重建任务学习通用神经表征
3. **CLIP 对齐**：将神经表征映射到 CLIP 视觉语义空间（768 维）
4. **图像重建**：通过 Stable Diffusion 从神经嵌入生成重建图像

**目标发表**：NeurIPS / ICML / Nature Methods

---

## 二、已完成工作

### Phase 0: 环境搭建 + 基线验证 ✅

| 任务 | 状态 | 结果 |
|------|------|------|
| POYO 环境搭建 | ✅ 完成 | PyTorch 2.10.0+cu128, conda env: poyo |
| POYO Baseline 训练 | ✅ 完成 | MC_Maze velocity R²=0.836 (1000 epochs, 目标≈0.87) |
| TVSD 数据克隆 | ✅ 完成 | datalad + git-annex, normMUA.mat 下载完成 |
| TVSD 数据探索 | ✅ 完成 | 22248 train + 100 test, 1024 electrodes/monkey |
| THINGS 图像下载 | ✅ 完成 | 4.68GB, 1854 类, OSF (password: things4all) |

**环境信息**：
- GPU: RTX 4090 D 24GB
- PYTHONPATH: `/root/autodl-tmp/NeuroBridge`
- HF 镜像: `HF_ENDPOINT=https://hf-mirror.com`

### Phase 1a: TVSD 数据适配 + CaPOYO 编码器 ✅

| 任务 | 状态 | 结果 |
|------|------|------|
| TVSDNormMUADataset 实现 | ✅ 完成 | 支持 raw/capoyo 模式，脑区过滤，SNR 过滤 |
| NeuroBridgeEncoder 实现 | ✅ 完成 | CaPOYO 架构，dim=128, depth=6, 2.3M params |
| 前向传播验证 | ✅ 完成 | 5/5 测试通过，143.5MB GPU (batch=32, AMP) |

**GO/NO-GO 决策点 1**: ✅ **GO** — CaPOYO tokenization 成功处理 TVSD MUA

**TVSD 数据格式**（normMUA，时间平均后的 2D 数据）：
```
train_MUA: [22248, 1024]  — 22248 张图对应的神经响应
test_MUA:  [100, 1024]    — 100 张测试图
```

**电极-脑区映射** (monkeyF)：
| 脑区 | 通道范围 | 电极数 | 时间窗口 (ms) |
|------|---------|--------|--------------|
| V1 | 1-512 | 512 | 25-125 |
| IT | 513-832 | 320 | 75-175 |
| V4 | 833-1024 | 192 | 50-150 |

### Phase 1b: 自监督 Masking 预训练 ✅

| 任务 | 状态 | 结果 |
|------|------|------|
| MaskingStrategy 实现 | ✅ 完成 | 电极随机 masking，支持脑区平衡 |
| MaskingDecoder 实现 | ✅ 完成 | Cross-attention 解码器，330K params |
| MaskingPretrainingModel | ✅ 完成 | Encoder+Decoder 完整预训练模型，2.6M params |
| 预训练训练 | ✅ 完成 | 200 epochs, MSE 0.212→0.092 (56% 下降) |

**预训练架构**：
```
Input MUA (1024 electrodes)
  → Mask 25% electrodes (随机)
  → NeuroBridgeEncoder (CaPOYO: cross-attn → self-attn × 6)
  → Latent tokens (8 × 128)
  → MaskingDecoder (cross-attn from query tokens to latents)
  → Predicted MUA values
  → MSE loss on masked positions only
```

**训练结果**：
```
Checkpoint: checkpoints/masking_pretrain_v1/
Epochs: 200 (完整训练，未 early stop)
Train masked MSE: 0.212 → 0.094
Val masked MSE:   0.175 → 0.092
Best epoch: 196
```

### Phase 2a: CLIP 对齐模块实现 ✅

| 任务 | 状态 | 结果 |
|------|------|------|
| CLIPWrapper | ✅ 完成 | open_clip ViT-L-14 封装 |
| NeuralReadout | ✅ 完成 | 8 learnable queries, cross-attention |
| NeuralProjector | ✅ 完成 | 3 层 MLP → 768 维 CLIP 空间 |
| InfoNCE Loss | ✅ 完成 | 对称 InfoNCE + temperature |
| Pipeline 验证 | ✅ 完成 | 随机嵌入下 chance level，3.3M params |

**对齐 Pipeline**：
```
TVSD normMUA [B, 1024]
  → NeuroBridgeEncoder → 8 latents × 128d
  → NeuralReadout (8 queries, cross-attention) → 8 tokens × 128d
  → NeuralProjector (mean pool → 512 hidden → 768) → L2 normalize
  → InfoNCE loss ← CLIP image embedding (768-dim)
```

### Phase 2b: CLIP 对齐训练 ✅

#### V1 训练（无增强）

| 指标 | 值 |
|------|-----|
| 总参数 | 3,299,648 |
| Best epoch | 40 (val top5=56.0%) |
| Test N→I Top-1 | **53.0%** |
| Test N→I Top-5 | **82.0%** |
| Test N→I Top-10 | 94.0% |
| Test I→N Top-1 | 54.0% |
| Test I→N Top-5 | 81.0% |
| Positive similarity | 0.164 ± 0.046 |

**过拟合分析**：从 epoch 35-40 开始过拟合。

#### V2 训练（带增强，推荐版本）

**改进**：电极 dropout=0.15, 高斯噪声 std=0.1, weight_decay=5e-4, early_stopping=30

| 指标 | V1 | V2 | 变化 |
|------|----|----|------|
| Best val Top-5 | 56.0% | **62.8%** | +6.8% |
| Test N→I Top-1 | **53.0%** | 49.0% | -4% |
| Test N→I Top-5 | 82.0% | **85.0%** | **+3%** |
| Test I→N Top-5 | 81.0% | **89.0%** | **+8%** |
| Positive sim | 0.164 | 0.157 | -0.007 |

**GO/NO-GO 决策点 3**: ✅ **GO** — Top-5 retrieval 82-85% 远超 15% 阈值

**Checkpoint**: `checkpoints/clip_alignment_v2/best_model.pt`

### Phase 2c: Stable Diffusion 图像重建 ✅

#### DiffusionAdapter

| 任务 | 状态 | 结果 |
|------|------|------|
| DiffusionAdapter 实现 | ✅ 完成 | 77 query tokens + cross-attn + self-attn refiner |
| SD 模型选择 | ✅ 完成 | SD 1.5 (SD 2.1 需认证，不可用) |
| Adapter 训练 | ✅ 完成 | 17.2M params, best epoch 33, val cos_sim=0.851 |
| 100 张图像重建 | ✅ 完成 | PixCorr=0.036, CLIP-sim=0.606 |

**DiffusionAdapter 架构**：
```
Neural CLIP embedding (768-dim)
  → DiffusionAdapter (17.2M params)
    - 77 learnable query tokens
    - Cross-attention to input embedding
    - Self-attention refiner (2 layers)
  → 77 tokens × 768 dim (SD prompt conditioning)
  → StableDiffusion 1.5 (DDIM 50 steps)
  → Reconstructed image (512×512)
```

#### 重建评估结果 (100 张测试图)

| 方法 | PixCorr | CLIP Similarity | CLIP ID Top-1 | CLIP ID Top-5 |
|------|---------|----------------|---------------|---------------|
| **SD 重建** (trained adapter) | 0.036 ± 0.108 | 0.606 ± 0.056 | 1.0% | 5.0% |
| **检索 Baseline** (nearest neighbor) | 0.125 ± 0.168 | 0.677 ± 0.102 | 28.0% | — |

**Retrieval baseline 指标**：
| 指标 | 值 |
|------|-----|
| Neural → Image Top-1 | 49.0% |
| Neural → Image Top-5 | 85.0% |
| Retrieval PixCorr | 0.125 |
| Retrieval CLIP-sim | 0.677 |
| Retrieval CLIP ID Top-1 | 28.0% |

**分析**：
- SD 重建产生了连贯的自然图像，但语义匹配度不高
- 检索 baseline 在所有指标上优于 SD 重建（这在脑解码领域是常见现象）
- 核心瓶颈：neural CLIP embeddings 与真实 CLIP embeddings 的对齐精度有限 (positive sim=0.157)

**结果文件**: `results/reconstruction_v3_trained/`

### Phase 3: 消融实验 ✅

#### 3a. 预训练 vs 随机初始化 Encoder

| 指标 | V2 (随机初始化) | V3 (预训练, freeze 10 ep) |
|------|----------------|--------------------------|
| Best Val Top-5 | **62.8%** | 55.5% |
| Test N→I Top-1 | 49% | **50%** |
| Test N→I Top-5 | **85%** | 76% |
| Test N→I Top-10 | **94%** | 88% |
| Test I→N Top-5 | **89%** | 83% |
| Positive sim | 0.157 | **0.172** |

**结论**：随机初始化在 Top-k retrieval 指标上**优于**预训练。可能原因：
1. 预训练特征（MUA 重建优化）与 CLIP 对齐目标不完全兼容
2. 数据集较小 (22K)，端到端学习比迁移更高效
3. Freeze-then-unfreeze 策略可能不是最优的

#### 3b. 脑区贡献分析

| 区域 | 电极数 | Best Val Top-5 | Test N→I Top-1 | Test N→I Top-5 | Test N→I Top-10 |
|------|--------|---------------|----------------|----------------|-----------------|
| **V1+V4+IT (全部)** | 1024 | **62.8%** | 49% | **85%** | **94%** |
| **IT only** | 320 | 53.9% | **50%** | 77% | 89% |
| **V4 only** | 192 | 34.0% | 25% | 58% | 74% |
| **V1 only** | 512 | 30.0% | 19% | 45% | 58% |

**关键发现**：

1. **IT 区域贡献最大**：IT-only (320 electrodes) 达到接近全区域的性能 (Top-5: 77% vs 85%)
   - IT 是腹侧视觉通路的高级区域，负责物体识别，与 CLIP 的语义空间最对齐
2. **V4 > V1**：V4 (192 electrodes) > V1 (512 electrodes) 尽管 V4 电极数更少
   - V4 处理形状和颜色等中级视觉特征，比 V1 的低级边缘/纹理特征更有语义信息
3. **全区域最优**：结合所有区域获得最佳性能，但 IT 的边际贡献最大
4. **电极数不决定性能**：V1 有 512 个电极但性能最低，IT 只有 320 个但性能最高

**神经科学意义**：这些结果与腹侧视觉通路的层级处理理论一致 —— 高级视觉区域 (IT) 包含更丰富的物体语义表征。

---

## 三、完整代码结构

```
neurobridge/
  __init__.py
  data/
    __init__.py
    tvsd_dataset.py                # TVSD normMUA 数据集适配器
  models/
    __init__.py
    neurobridge_encoder.py         # CaPOYO-based 编码器 (2.3M params)
  alignment/
    __init__.py
    clip_wrapper.py                # CLIP 封装 (open_clip ViT-L-14)
    readout.py                     # 可学习 readout 查询 (cross-attention)
    projector.py                   # MLP 投影器 → CLIP 空间
    infonce.py                     # 对称 InfoNCE 对比损失
  generation/
    __init__.py
    diffusion_adapter.py           # DiffusionAdapter + SDWrapper
  pretraining/
    __init__.py
    masking_strategy.py            # 电极 masking 策略
    masking_decoder.py             # Cross-attention 重建解码器
    masking_pretraining.py         # 完整预训练模型
  tests/
    __init__.py
    test_tvsd_forward.py           # 前向传播测试

scripts/
  train_clip_alignment.py          # CLIP 对齐训练 (支持预训练/冻结/增强)
  train_masking_pretraining.py     # Masking 预训练
  train_diffusion_adapter.py       # DiffusionAdapter 训练
  extract_clip_embeddings.py       # CLIP 嵌入预提取
  evaluate_alignment.py            # 检索评估 (支持脑区过滤)
  evaluate_reconstruction.py       # 重建评估 (PixCorr/CLIP-sim/检索)
  reconstruct_images.py            # 端到端图像重建
```

### Checkpoint 与数据文件

```
checkpoints/
  clip_alignment_v1/               # 无增强, epoch 40, val top5=56.0%
  clip_alignment_v2/               # 带增强(推荐), epoch 42, val top5=62.8%
  clip_alignment_v3_pretrained/    # 预训练 encoder, epoch 23, val top5=55.5%
  clip_alignment_V1only/           # V1 区域, val top5=30.0%
  clip_alignment_V4only/           # V4 区域, val top5=34.0%
  clip_alignment_ITonly/           # IT 区域, val top5=53.9%
  diffusion_adapter_v1/            # Adapter, epoch 33, cos_sim=0.851
  masking_pretrain_v1/             # 预训练, epoch 196, MSE=0.092

data/clip_embeddings/
  clip_train_monkeyF.npy           # (22248, 768) float32
  clip_test_monkeyF.npy            # (100, 768) float32

results/
  reconstruction_v3_trained/       # 100 张重建图 + 评估指标
    reconstructed/                 # recon_0000.png - recon_0099.png
    comparison/                    # 原图 vs 重建 对比
    retrieval_comparison/          # 原图 vs 检索 vs 重建 三方对比
    evaluation_metrics.json        # 完整评估指标
    neural_embeddings.npy          # 测试集神经嵌入
```

---

## 四、Git 提交记录

```
b77774d feat: complete Phase 3 ablation experiments + comprehensive results
755fd63 feat: add pretrained encoder support + freeze warmup for CLIP alignment
4a1d42c feat: comprehensive reconstruction evaluation + retrieval baseline
aec765b feat: implement Phase 1b — masking pretraining modules
003a120 feat: add DiffusionAdapter training script + first reconstruction results
e2f0acf docs: V2 training results + daily summary
1a62620 feat: add augmentation, switch to SD 1.5, improve training pipeline
1005358 feat: CLIP alignment training complete — Test Top-1=53%, Top-5=82%
cdbb231 docs: update progress log with Phase 2b
33780c7 docs: update progress log — Phase 2a-2c modules complete
02236d5 feat: implement Phase 2c — Diffusion Adapter and SD wrapper
c089711 feat: add CLIP alignment evaluation script
439b0de feat: add CLIP embedding extraction script
25776f9 feat: implement Phase 2a — CLIP alignment modules and training pipeline
748a83d feat: implement Phase 1a — TVSD data adapter and NeuroBridge encoder
fd297fd docs: update progress log with POYO baseline results
67f6697 添加 POYO 代码分析和 NeuroBridge 项目执行计划文档
```

---

## 五、关键结论与分析

### 5.1 核心成果

1. **CLIP 对齐表现优秀**：Test Top-1=49-53%, Top-5=82-89%，远超 chance level (1%/5%)
2. **脑区层级关系得到验证**：IT >> V4 > V1，与腹侧视觉通路理论一致
3. **SD 重建产生连贯图像**：但语义匹配度有限，检索方法优于生成方法
4. **自监督预训练对 CLIP 对齐无帮助**：随机初始化反而表现更好

### 5.2 主要发现

- **最佳 CLIP 对齐性能**：V2 模型（带增强），Test N→I Top-5=85%, I→N Top-5=89%
- **IT 区域是关键**：仅用 IT 区域 320 个电极即可达到全区域 ~90% 的性能
- **数据增强有效**：电极 dropout + 高斯噪声将 val top-5 从 56.0% 提升到 62.8%
- **SD 重建瓶颈**：neural-to-CLIP 对齐精度 (positive sim ≈ 0.16) ���制了重建质量

### 5.3 遇到的问题与解决

| 问题 | 解决方案 |
|------|---------|
| HuggingFace 直连不可用 | `HF_ENDPOINT=https://hf-mirror.com` 镜像 |
| SD 2.1 需认证 (401) | 改用 SD 1.5 (非受限, text_hidden_dim=768 兼容) |
| MAT v7.3 文件 | 用 h5py 读取 (scipy.io.loadmat 不支持) |
| 模型过拟合 (epoch 35-40) | 增加电极 dropout/噪声/weight_decay |
| 脑区消融评估 vocabulary 不匹配 | 为 evaluate_alignment.py 添加 --regions 参数 |

---

## 六、待做工作

### 高优先级（直接影响论文质量）

1. **改进 SD 图像重建质量**
   - 端到端 Adapter 训练（使用 neural embeddings 而非仅 CLIP embeddings）
   - 尝试检索+SD 混合方法（检索最近邻图像，用 SD 基于 neural embedding 精修）
   - 探索 SoftCLIP、label smoothing 等技术提升 CLIP 对齐精度

2. **Masking 预训练策略优化**
   - 当前预训练对 CLIP 对齐无帮助，需要：
     - 尝试不同 freeze/unfreeze 策略（如 gradual unfreezing）
     - 增大预训练数据量（下载完整时间序列 THINGS_MUA_trials.mat ~58GB）
     - 尝试多任务预训练（reconstruction + classification）
   - 或直接发表 "direct alignment" 方案（方案 C）

3. **补充消融实验**
   - temporal masking vs neuron masking vs combined（当前仅实现了 neuron masking）
   - 不同 mask ratio (15%, 25%, 50%) 对比
   - 不同时间窗口对各脑区的效果

### 中优先级

4. **模型扩展实验**
   - dim=256, depth=8 (~15-20M params)，评估 scaling 效果
   - 需要 gradient checkpointing 以适应 24GB GPU

5. **跨猴泛化测试**
   - 在 monkeyN 数据上训练和评估
   - 测试 monkeyF → monkeyN 的跨个体迁移

6. **更多评估指标**
   - SSIM (结构相似度)
   - FID / IS (生成质量)
   - 按语义类别分析性能

### 低优先级（可延后）

7. **IBL 数据混合训练**
8. **Allen 数据验证**（仅 118 张图，验证价值有限）
9. **论文撰写**（需上述实验结果支撑）

---

## 七、论文发表方案

### 方案 A（最优）：完整 NeuroBridge
- 预训练 + CLIP 对齐 + 图像重建
- 消融实验证明预训练价值 + 脑区贡献分析
- **当前状态**：预训练对 CLIP 对齐无帮助，需要进一步优化或调整叙事
- 目标：NeurIPS / ICML

### 方案 B（降级）：Masked Pretraining + 表征分析
- 仅发表预训练组件，强调 MUA 表征学习
- 需要更多下游任务验证
- 目标：Workshop / ICLR

### 方案 C（当前数据最支持的方案）：Direct Neural-to-CLIP Alignment
- 核心 claim：CaPOYO 架构直接对齐 MUA → CLIP，无需预训练
- 强调脑区贡献分析（IT >> V4 > V1）
- 对比 MonkeySee 等 baseline
- Top-5 retrieval 85% 是一个有竞争力的结果
- 目标：计算神经科学 venue

---

## 八、资源与环境

| 资源 | 详情 |
|------|------|
| GPU | RTX 4090 D 24GB |
| Python 环境 | conda env: poyo, PyTorch 2.10.0+cu128 |
| 关键依赖 | open_clip 3.2.0, transformers 5.2.0, diffusers 0.36.0 |
| TVSD 数据 | `/root/autodl-tmp/TVSD_dataset/` |
| THINGS 图像 | `~/autodl-tmp/THINGS_images/object_images/` |
| HF 镜像 | `HF_ENDPOINT=https://hf-mirror.com` |
| SD 模型 | `runwayml/stable-diffusion-v1-5` (已缓存) |

---

*文档生成时间：2026-02-21*
