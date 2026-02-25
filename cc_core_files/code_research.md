# POYO / torch_brain 代码库深度分析

**分析日期**：2026-02-21
**分析者**：Claude Code
**代码仓库**：`/root/autodl-tmp/NeuroBridge`（fork 自 `neuro-galaxy/poyo`）
**远程仓库**：`git@github.com:Jp-17/NeuroBridge.git`

---

## 1. 项目概览

POYO (Population-level POYO) 是一个基于 Transformer 的神经群体解码框架，由 Azabou et al. 2023 (NeurIPS 2023) 提出。核心创新是将每个 spike 作为独立 token 处理（spike-level tokenization），通过 PerceiverIO 架构高效压缩变长 spike 序列，保留毫秒级时间分辨率。

### 核心论文
- **POYO**: Azabou et al., "A Unified, Scalable Framework for Neural Population Decoding", NeurIPS 2023, arXiv:2310.16046
- **POYO+**: Azabou et al., "Multi-session, multi-task neural decoding from the visual cortex", ICLR 2025 (Spotlight)

---

## 2. 目录结构

```
NeuroBridge/
├── torch_brain/                # 核心包
│   ├── data/                   # 数据加载、采样、collation
│   │   ├── sampler.py          # RandomFixedWindowSampler, DistributedStitchingFixedWindowSampler
│   │   ├── collate.py          # pad/pad8/track_mask/track_mask8 collation 函数
│   │   └── dataset.py          # [已废弃] 旧 Dataset 类
│   ├── dataset/                # 新 Dataset 接口
│   │   ├── dataset.py          # HDF5 lazy-loading Dataset 类
│   │   ├── nested.py           # NestedDataset / NestedSpikingDataset
│   │   └── mixins.py           # 数据增强 mixins
│   ├── models/                 # 模型实现
│   │   ├── poyo.py             # POYO 单任务模型 (核心)
│   │   ├── poyo_plus.py        # POYO+ 多任务模型
│   │   └── capoyo.py           # CaPOYO 钙成像变体
│   ├── nn/                     # 神经网络组件
│   │   ├── rotary_attention.py # RoPE Cross/Self Attention
│   │   ├── position_embeddings.py # RotaryTimeEmbedding
│   │   ├── embedding.py        # Embedding 基础类
│   │   ├── infinite_vocab_embedding.py # InfiniteVocabEmbedding
│   │   ├── feedforward.py      # GEGLU FeedForward
│   │   ├── loss.py             # MSELoss, CrossEntropyLoss, MallowDistanceLoss
│   │   └── multitask_readout.py # MultitaskReadout (POYO+/CaPOYO用)
│   ├── transforms/             # 数据增强
│   │   └── unit_dropout.py     # UnitDropout transform
│   ├── utils/                  # 工具
│   │   ├── tokenizers.py       # create_start_end_unit_tokens, create_linspace_latent_tokens
│   │   ├── stitcher.py         # DecodingStitchEvaluator (验证时窗口拼接)
│   │   └── callbacks.py        # Lightning callbacks
│   ├── registry.py             # 模态注册 (ModalitySpec)
│   └── optim.py                # SparseLamb 优化器
├── examples/
│   ├── poyo/                   # POYO 训练示例
│   │   ├── train.py            # 主训练脚本 (Lightning)
│   │   ├── configs/            # Hydra 配置
│   │   │   ├── defaults.yaml   # 默认超参数
│   │   │   ├── model/          # 模型配置 (poyo_1.3M.yaml, poyo_11.8M.yaml)
│   │   │   └── train_mc_maze_small.yaml  # MC_Maze 训练配置
│   │   └── datasets/           # 数据集加载器
│   │       ├── nlb.py          # NLB MC_Maze 数据集
│   │       ├── poyo_mp.py      # Perich-Miller Population 数据集
│   │       └── poyo_1.py       # 多数据集组合
│   └── poyo_plus/              # POYO+ 训练示例
├── tests/                      # 测试 (21 个测试文件)
├── proposal/                   # NeuroBridge 研究计划
├── cc_todo/                    # 项目记录
├── docs/                       # Sphinx 文档
├── scripts/                    # 辅助脚本
├── pyproject.toml              # 项目配置
└── environment.yaml            # Conda 环境
```

---

## 3. 核心架构分析

### 3.1 POYO 模型 (`torch_brain/models/poyo.py`)

**类**: `POYO(nn.Module)`

**处理流程**:
```
INPUT:
  spike_unit_index [B, N_spikes]    # 每个 spike 所属的神经元 ID
  spike_timestamps [B, N_spikes]    # 每个 spike 的精确时间戳
  spike_token_type [B, N_spikes]    # token 类型 (spike/start-of-unit/end-of-unit)
  spike_mask [B, N_spikes]          # 有效性 mask (padding 区分)

ENCODING:
  1. input_tokens = unit_emb(unit_index) + token_type_emb(token_type)
     # dim = dim (如 64)
  2. latent_tokens = latent_emb + session_emb
     # 固定数量: num_latents_per_step × (sequence_length / latent_step)
     # 例: 16 × (1.0 / 0.125) = 128 个 latent tokens
  3. Cross-Attention: latent_tokens attend to input_tokens (with RoPE)
  4. FFN on latent_tokens

PROCESSING:
  5. depth 层 Self-Attention (with RoPE) + FFN
     # 每层: RotarySelfAttention → FeedForward

DECODING:
  6. output_queries = session_emb (broadcast to output timestamps)
  7. Cross-Attention: output_queries attend to latent_tokens (with RoPE)
  8. FFN on output_queries
  9. Linear projection: dim → output_dim (如 2 for velocity)
```

**关键参数 (1.3M 模型)**:
```yaml
dim: 64
depth: 6
dim_head: 64
cross_heads: 2
self_heads: 8
num_latents_per_step: 16
latent_step: 0.125  # 125ms
sequence_length: 1.0  # 1 秒
```

**参数量估算 (1.3M)**:
- Encoder cross-attn: ~16K (dim=64, 2 heads)
- Encoder FFN: ~33K (dim=64, mult=4, GEGLU)
- 6× Self-attn: 6 × ~16K = ~96K
- 6× FFN: 6 × ~33K = ~198K
- Decoder cross-attn + FFN: ~49K
- Embeddings: ~800K (InfiniteVocab + session + token_type + latent)
- Output linear: ~128
- Total: ~1.3M

### 3.2 CaPOYO 模型 (`torch_brain/models/capoyo.py`)

**与 POYO 的关键差异**:

```python
# CaPOYO 额外组件:
self.input_value_map = nn.Linear(1, dim // 2)  # 连续值 → 半维度嵌入
self.unit_emb = InfiniteVocabEmbedding(dim // 2)  # 另一半维度

# 输入 token 构造:
inputs = cat(
    (self.input_value_map(input_values),  # [B, N, dim//2] 连续值映射
     self.unit_emb(input_unit_index)),    # [B, N, dim//2] unit 嵌入
    dim=-1                                 # → [B, N, dim]
)
```

CaPOYO 是处理 TVSD MUA 数据的关键参考，因为它已经解决了"连续值信号如何 tokenize"的问题。

### 3.3 POYO+ 模型 (`torch_brain/models/poyo_plus.py`)

**与 POYO 的关键差异**:
- `MultitaskReadout`: 多任务解码头，每个任务有独立的 linear projection
- `task_emb`: 可学习的任务嵌入
- `output_decoder_index`: 指定每个输出时间点属于哪个任务
- 支持同时解码多种行为变量（velocity, cursor position, etc.）

---

## 4. 关键组件详解

### 4.1 Rotary Attention (`torch_brain/nn/rotary_attention.py`)

两个核心类：

**RotaryCrossAttention** (lines 17-100):
- Queries (latent tokens) attend to Keys/Values (input tokens)
- RoPE 应用于 Q 和 K 的时间戳
- 支持 xformers varlen path (memory efficient)
- `rotate_value=True`: 额外将 RoPE 应用于 V（CaPOYO 使用）

**RotarySelfAttention** (lines ~100+):
- 标准 self-attention with RoPE
- 用于 processor 阶段的 latent token 间交互

### 4.2 Position Embeddings (`torch_brain/nn/position_embeddings.py`)

**RotaryTimeEmbedding**:
- 将连续时间戳编码为旋转位置编码
- 频率范围: t_min=1e-4 到 t_max=2.0627
- rotate_dim=32 (默认 dim_head//2)
- 保留毫秒级时间分辨率

### 4.3 Infinite Vocab Embedding (`torch_brain/nn/infinite_vocab_embedding.py`)

- 处理任意数量的神经元 ID
- Lazy 初始化: 第一次 forward pass 时自动发现并注册新 ID
- 支持 `SparseLamb` 优化器的稀疏更新
- 跨 session 时，新 session 的新 unit IDs 会自动分配新的 embedding 向量

### 4.4 Tokenizers (`torch_brain/utils/tokenizers.py`)

```python
create_start_end_unit_tokens(unit_ids, domain):
    # 为每个 unit 创建 start/end boundary tokens
    # token_type: 0=spike, 1=start, 2=end, 3=calcium_event

create_linspace_latent_tokens(sequence_length, latent_step, num_latents_per_step):
    # 创建固定时间网格上的 latent tokens
    # timestamps: linspace from start to end
    # indices: [0, 1, ..., num_latents_per_step-1] repeated
```

### 4.5 Registry (`torch_brain/registry.py`)

```python
@dataclass
class ModalitySpec:
    dim: int                    # 输出维度 (如 2 for velocity)
    type: DataType              # CONTINUOUS / DISCRETE / ORDINAL
    loss_fn: Loss               # Loss 函数实例
    timestamp_key: str          # 时间戳数据键
    value_key: str              # 值数据键
    metric: torchmetrics.Metric # 评估指标

# 已注册的模态:
# "cursor_velocity_2d": dim=2, MSELoss, R2Score
# 等其他模态...
```

### 4.6 Data Pipeline

**Dataset** (`torch_brain/dataset/dataset.py`):
- HDF5 lazy-loading
- `DatasetIndex(recording_id, start, end)` 定义时间窗口
- `temporaldata.Data` 对象作为数据容器

**Sampler** (`torch_brain/data/sampler.py`):
- `RandomFixedWindowSampler`: 训练时随机时间窗口 + jitter
- `DistributedStitchingFixedWindowSampler`: 验证时确定性窗口（支持重叠拼接）

**Collate** (`torch_brain/data/collate.py`):
- `pad8()`: padding 到 8 的倍数（GPU 优化）
- `track_mask8()`: 生成有效性 mask

### 4.7 Training Pipeline (`examples/poyo/train.py`)

基于 PyTorch Lightning:

```python
class TrainWrapper(L.LightningModule):
    configure_optimizers():
        # SparseLamb + OneCycleLR
        # base_lr = 3.125e-5, lr = base_lr × batch_size
        # weight_decay = 1e-4

    training_step():
        # Forward → MSE loss on velocity prediction

    validation_step():
        # Model eval with DecodingStitchEvaluator

class DataModule(L.LightningDataModule):
    setup():     # 初始化 dataset
    link_model(): # 绑定 unit/session embeddings
    train_dataloader(): # RandomFixedWindowSampler
    val_dataloader():   # DistributedStitchingFixedWindowSampler
```

**Hydra 配置系统**:
- `defaults.yaml`: 全局默认参数
- `model/poyo_1.3M.yaml`: 模型架构参数
- `train_mc_maze_small.yaml`: 数据集 + 训练特定参数

---

## 5. Fork 修改历史

本仓库 fork 自 `neuro-galaxy/poyo`，已有以下修改：

| Commit | 内容 | 影响 |
|--------|------|------|
| `2ee8cc6` | 重命名 pytorch_brain → torch_brain | 包名变更 |
| `1105508` / `461bab4` | 新 Dataset 类 (HDF5 lazy-loading) | 新 `torch_brain.dataset` 模块 |
| `0e008bc` (implied) | 添加可运行的 examples | 新 `examples/poyo/datasets/` |
| `cc5c814` (implied) | 添加 CaPOYO | 新 `torch_brain/models/capoyo.py` |
| `0a7748b` | 添加 NeuroBridge proposal | 新 `proposal/` 目录 |
| `0f4dd39` | 添加数据集下载说明 | 新文档 |

---

## 6. 对 NeuroBridge 项目的关键启示

### 6.1 可直接复用的组件

| 组件 | 文件路径 | 复用方式 |
|------|---------|---------|
| RotaryCrossAttention | `torch_brain/nn/rotary_attention.py` | Target Seq Decoder 的 cross-attention |
| RotarySelfAttention | 同上 | Backbone 的 self-attention |
| RotaryTimeEmbedding | `torch_brain/nn/position_embeddings.py` | 所有时间编码 |
| InfiniteVocabEmbedding | `torch_brain/nn/infinite_vocab_embedding.py` | Unit embedding |
| CaPOYO.input_value_map | `torch_brain/models/capoyo.py:68` | MUA 连续值 tokenization |
| FeedForward (GEGLU) | `torch_brain/nn/feedforward.py` | 所有 FFN 层 |
| MSELoss | `torch_brain/nn/loss.py` | MUA 重建 loss |
| ModalitySpec / Registry | `torch_brain/registry.py` | 注册新的输出模态 |
| RandomFixedWindowSampler | `torch_brain/data/sampler.py` | 训练数据采样 |
| pad8 / track_mask8 | `torch_brain/data/collate.py` | Batch collation |
| Dataset (HDF5) | `torch_brain/dataset/dataset.py` | 数据加载 |
| SparseLamb | `torch_brain/optim.py` | Embedding 稀疏优化 |

### 6.2 需要新增的组件

| 组件 | 用途 | 参考 |
|------|------|------|
| Target Sequence Decoder | Masking 预训练的重建解码器 | 基于 RotaryCrossAttention |
| Multi-Task Masking Controller | 4 种 masking 策略切换 | 全新 |
| Poisson NLL Loss | Spike count 重建 loss | 全新 |
| Neural Projector | 神经表征 → CLIP 空间 | MLP |
| InfoNCE Loss | 对比学习对齐 | 标准实现 |
| Diffusion Adapter | CLIP embedding → SD 条件 | 基于 MindEye2 |
| TVSD Data Loader | TVSD MUA → torch_brain 格式 | 全新 |

### 6.3 RTX 4090 (24GB) 下的模型配置建议

```yaml
# 推荐起步配置 (适配 24GB VRAM)
dim: 128          # 比 POYO 1.3M 的 64 翻倍
depth: 6          # 与 POYO 1.3M 相同
dim_head: 64      # 与 POYO 相同
cross_heads: 2
self_heads: 4     # 128/64 = 2 heads minimum, 用 4
num_latents_per_step: 8  # 减少 latent 数量以适应显存
latent_step: 0.125
sequence_length: 1.0
batch_size: 32-64  # 需要实测
precision: 16      # Mixed precision 必须开启
```

预估参数量: ~5M params，单卡 24GB 在 AMP 下可以训练。

---

## 7. 版本记录

| 日期 | 版本 | 内容 |
|------|------|------|
| 2026-02-21 | v1.0 | 初始代码分析完成 |
