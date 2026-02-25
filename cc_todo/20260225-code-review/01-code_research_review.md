# code_research.md 深度审查报告

**审查日期**: 2026-02-25
**审查对象**: `cc_core_files/code_research.md` (v1.0, 2026-02-21)
**审查方法**: 逐条对照仓库源代码 (`torch_brain/` 及 `examples/`) 进行验证
**审查者**: Claude Code (Opus 4.6)

---

## 总评

`code_research.md` 是一份整体质量较高的代码分析文档，在项目概览、目录结构、核心架构流程方面的描述基本准确。但在**参数量估算**、**模型细节精度**、**CaPOYO tokenization 的含义**等方面存在需要修正的问题。此外有若干对 NeuroBridge 项目具有重要影响的代码细节被遗漏。

以下按"需要修正的错误" → "需要补充的遗漏" → "正确但可深化的部分"三个维度逐一分析。

---

## 第一部分：需要修正的错误

### 1.1 参数量估算（第 3.1 节）——单组件估值有显著偏差

**原文声明**:
```
Encoder cross-attn: ~16K (dim=64, 2 heads)
Encoder FFN: ~33K (dim=64, mult=4, GEGLU)
6x Self-attn: 6 x ~16K = ~96K
6x FFN: 6 x ~33K = ~198K
```

**实际代码验证** (基于 `poyo_1.3M.yaml`: dim=64, dim_head=64, cross_heads=2, self_heads=8):

| 组件 | 原文估算 | 实际参数量 | 偏差原因 |
|------|---------|-----------|---------|
| Encoder CrossAttn | ~16K | ~33K | to_q(64→128) + to_kv(64→256) + to_out(128→64) + 2×LayerNorm(64) ≈ 33K。原文可能只算了 to_q 或漏算了 to_kv |
| Encoder FFN | ~33K | ~50K | `FeedForward` 的第一层是 `nn.Linear(dim, dim*mult*2)` = Linear(64,512)（因为 GEGLU 需要 2 倍宽度），第二层是 Linear(256,64)。实际 = 64×512 + 512 + 256×64 + 64 + LayerNorm ≈ 50K |
| **Self-Attn (每层)** | **~16K** | **~131K** | **这是最大的偏差**。self_heads=8, dim_head=64 → inner_dim = 8×64 = **512**（不是64）。to_qkv: Linear(64, 1536) = 98K, to_out: Linear(512, 64) = 33K。原文似乎按 inner_dim=64 或 heads=1 估算 |
| Self-Attn FFN (每层) | ~33K | ~50K | 同 Encoder FFN |
| 6× Self-Attn 总计 | ~96K | ~787K | 131K × 6 |
| 6× FFN 总计 | ~198K | ~300K | 50K × 6 |
| Decoder CrossAttn + FFN | ~49K | ~83K | 同 Encoder 结构 |

**关键发现**: 虽然单组件估值偏差显著（Self-Attention 层差了约 8 倍），但总参数量 ~1.25M（不含 InfiniteVocab embedding）加上典型的 embedding 开销后确实约 1.3M，因此**总数恰好是大致正确的**。这是一个巧合——Self-Attention 被严重低估，但 embedding 开销被高估（原文估算 ~800K embedding，实际 InfiniteVocab 规模取决于数据集，MC_Maze 约 100-200 个 units = 6-13K）。

**对 NeuroBridge 的影响**: 如果基于这些错误的单组件估算来设计 NeuroBridge 的模型规模（dim=128），Self-Attention 层的参数量会被严重低估。dim=128, self_heads=4, dim_head=64 时：inner_dim = 256, 每层 Self-Attn ≈ 131K（恰好和 dim=64 的 POYO 一样，因为 inner_dim 都是由 heads × dim_head 决定的，与 dim 无直接关系……等等，不对）。

让我重新算：dim=128, self_heads=4, dim_head=64:
- inner_dim = 4 × 64 = 256
- to_qkv: Linear(128, 768) = 128×768 = 98.3K
- to_out: Linear(256, 128) = 256×128 + 128 = 32.9K
- norm: 256
- 每层总计: ~131.5K

所以 NeuroBridge 的 dim=128 模型与 POYO 1.3M 的每层 Self-Attn 参数量几乎相同（因为 inner_dim 从 512 降到 256，但 dim 从 64 升到 128，两者恰好抵消）。**建议在 plan.md 中提供精确的参数量计算表。**

### 1.2 模型前向接口参数命名不一致

**原文 (第 3.1 节)**:
```
INPUT:
  spike_unit_index [B, N_spikes]
  spike_timestamps [B, N_spikes]
  spike_token_type [B, N_spikes]
  spike_mask [B, N_spikes]
```

**实际代码** (`torch_brain/models/poyo.py:159-167`):
```python
def forward(self, *,
    input_unit_index,    # 不是 spike_unit_index
    input_timestamps,    # 不是 spike_timestamps
    input_token_type,    # 不是 spike_token_type
    input_mask,          # 不是 spike_mask
    ...
```

**影响程度**: 低。语义正确，只是命名不同。但在编写 NeuroBridge 代码时需要使用正确的接口名称。`tokenize()` 方法中使用 `spike_xxx` 命名的是中间变量，最终打包到 `data_dict["model_inputs"]` 时转换为 `input_xxx`。

### 1.3 TokenType 枚举值描述不精确

**原文**: "token_type: 0=spike, 1=start, 2=end, 3=calcium_event"

**实际代码** (`torch_brain/utils/tokenizers.py:6-9`):
```python
class TokenType(Enum):
    DEFAULT = 0
    START_OF_SEQUENCE = 1
    END_OF_SEQUENCE = 2
```

枚举只定义了 3 个值 (0, 1, 2)，没有 `3=calcium_event`。虽然 `token_type_emb = Embedding(4, dim)` 支持 4 个 token 类型，但第 4 个（index 3）在当前代码中并未被命名或使用。CaPOYO 模型甚至完全不使用 `token_type_emb`——它的 tokenize() 方法不创建 start/end tokens。

**对 NeuroBridge 的影响**: 如果要同时处理 spike 数据和 MUA 数据，可以利用 index=3 定义一个新的 token type（如 MUA_SAMPLE），但需要注意 CaPOYO 的实际做法是完全不使用 token_type。

### 1.4 `rotate_value` 参数的描述不够精确

**原文 (第 4.1 节)**: "`rotate_value=True`: 额外将 RoPE 应用于 V（CaPOYO 使用）"

**实际代码**: `rotate_value=True` 不仅是 CaPOYO 使用，**POYO 和 POYO+ 的 encoder cross-attention 和所有 self-attention 层都使用 `rotate_value=True`**。唯一使用 `rotate_value=False` 的是**所有模型的 decoder cross-attention** (`dec_atn`)。

```python
# poyo.py
self.enc_atn = RotaryCrossAttention(..., rotate_value=True)    # Encoder
self.proc_layers: RotarySelfAttention(..., rotate_value=True)   # Processor
self.dec_atn = RotaryCrossAttention(..., rotate_value=False)    # Decoder ← 唯一的 False
```

**对 NeuroBridge 的影响**: NeuroBridge 的 Target Sequence Decoder 应该延续这个模式——decoder cross-attention 使用 `rotate_value=False`。这个设计选择的理由可能是：decoder 的 output queries 代表"要查询的时间点"，其时间信息通过 RoPE 作用于 Q 和 K 就足够了，对 V 进行旋转反而可能引入不必要的时间偏移。

---

## 第二部分：需要补充的遗漏

### 2.1 CaPOYO tokenize() 的 T×N 问题——对 TVSD 适配至关重要

**这是最关键的遗漏。**

code_research.md 正确描述了 CaPOYO 的 `input_value_map` 机制，但完全没有讨论 CaPOYO 的 tokenize() 是如何构造输入序列的：

```python
# capoyo.py tokenize() 的核心逻辑:
T, N = calcium_traces.df_over_f.shape  # T=时间点数, N=神经元数
input_timestamps = repeat(timestamps, "T -> (T N)", T=T, N=N)
input_values = rearrange(df_over_f, "T N -> (T N) 1", T=T, N=N)
input_unit_index = repeat(unit_index, "N -> (T N)", T=T, N=N)
# 结果：T*N 个 tokens
```

**关键含义**: CaPOYO 为**每个时间点 × 每个神经元**创建一个 token。如果 T=100, N=1000, 那就是 100,000 个 tokens。

而 POYO 的 spike-level tokenization 只为**每个实际发放的 spike**创建一个 token，token 数量 = spike 数量，通常远小于 T×N。

**对 TVSD 适配的严重影响**:
- TVSD MUA 数据：1024 电极，如果以 1kHz 采样 1 秒 = 1,024,000 tokens per sample
- 即使降采样到 100Hz × 1s = 100 × 1024 = 102,400 tokens
- 这远超 POYO/CaPOYO 设计的处理能力

**建议**: plan.md 中必须明确 TVSD MUA 的时间分辨率选择策略。可能的方案：
1. **MUA binning**: 将 MUA 在时间维度上 bin 到 50-100ms bins，每个 bin 内取平均，得到 [10-20 时间点 × 1024 电极] = 10K-20K tokens
2. **阈值化为 spike events**: 对 MUA 设定阈值提取离散事件，转化为 POYO-style spike tokens
3. **���层 tokenization**: 先在电极阵列 (8×8) 内做空间聚合，再在时间上 token 化

### 2.2 POYO.load_pretrained() 方法

code_research.md 完全没有提到 `POYO.load_pretrained()` 这个类方法 (`poyo.py:364-417`)。这个方法：
- 从 Lightning checkpoint 加载模型参数
- 支持 `skip_readout=True` 跳过输出投影层
- 自动从 checkpoint 的 `hyper_parameters` 恢复模型架构

**对 NeuroBridge 的影响**: 这个方法是实现"预训练 → 下游迁移"的现成工具。NeuroBridge Phase 3（CLIP 对齐）需要加载预训练模型并替换 readout 层，可以直接使用 `skip_readout=True`。

### 2.3 Registry 中已注册的丰富模态

code_research.md 只提到 `cursor_velocity_2d` 一个注册模态。实际上 `registry.py` 注册了 **18 个模态**，包括：

| 模态 | 维度 | 类型 | 与 NeuroBridge 的关系 |
|------|------|------|---------------------|
| `natural_scenes` | dim=119 | MULTINOMIAL | **直接可用**：Allen Natural Scenes 的 119 类图像分类 |
| `drifting_gratings_orientation` | dim=8 | MULTINOMIAL | Allen 视觉刺激 |
| `natural_movie_one/two/three_frame` | dim=900/900/3600 | MULTINOMIAL | Allen 自然视频帧分类 |
| `running_speed` | dim=1 | CONTINUOUS | 行为变量解码 |
| `pupil_size_2d` / `gaze_pos_2d` | dim=2 | CONTINUOUS | 行为变量解码 |

**对 NeuroBridge 的影响**:
1. `natural_scenes` 模态已经注册，说明 POYO+/CaPOYO 框架已经为 Allen Natural Scenes 数据做了适配。NeuroBridge 可以直接利用这个模态作为图像分类的 baseline（先用 POYO+ 做 119 类分类，验证视觉编码质量，再进行 CLIP 对齐）。
2. 大量已注册的 Allen 模态意味着 Allen 数据加载器很可能已经存在或接近完成，不需要从头写。

### 2.4 varlen attention 路径

`RotaryCrossAttention` 和 `RotarySelfAttention` 都提供 `forward_varlen()` 方法 (`rotary_attention.py:127-194`, `288-341`)，使用 xformers 的 `BlockDiagonalMask` 实现变长序列的高效 attention。

这条路径避免了 padding 带来的计算浪费，对 spike-level tokenization 这种天然变长序列尤为重要。但它**要求安装 xformers**，且**仅在 GPU 上可用**。

**对 NeuroBridge 的影响**: 训练效率优化的关键路径。当 batch 内不同样本的 spike 数量差异很大时（常见于多 session 混合训练），varlen attention 比 padding-based attention 节省大量显存和计算。plan.md 应将 xformers 列为必要依赖。

### 2.5 `poyo_mp` 函数中的参数差异

code_research.md 准确描述了 1.3M 配置文件的参数，但遗漏了 `poyo.py` 底部定义的 `poyo_mp()` 函数（`poyo.py:420-440`），其中 **`t_max=4.0`**（而非默认的 2.0627）。

```python
def poyo_mp(readout_spec, ckpt_path=None):
    return POYO(
        ...
        t_max=4.0,  # 注意：默认值是 2.0627，这里是 4.0
        ...
    )
```

`t_max` 控制 RoPE 能编码的最大时间周期。4.0 意味着这个配置能处理更长的时间依赖关系（up to 4 秒的周期性模式），适合 multi-session population 数据。

**对 NeuroBridge 的影响**: 对于 TVSD 数据（单图像呈现 100-300ms），t_max=2.0627 可能就足够了。但如果进行长序列预训练（如 2-5 秒窗口），可能需要增大 t_max。

### 2.6 POYO 默认构造参数与配置文件参数的差异

代码中 `POYO.__init__` 的默认值是 `dim=512, depth=2, num_latents_per_step=64`，而 `poyo_1.3M.yaml` 的配置是 `dim=64, depth=6, num_latents_per_step=16`。这两者差距巨大。

code_research.md 只讨论了配置文件的参数，没有提到代码默认值。虽然实践中总是用配置文件覆盖，但如果有人直接实例化 `POYO()` 而不传参数，会得到一个完全不同规模的模型。

### 2.7 Dropout 策略的层级差异

POYO 使用了两种不同的 dropout 机制，code_research.md 未区分：

1. **模型级 dropout** (`self.dropout = nn.Dropout(p=lin_dropout)`)：只应用在 **processor 层**的 self-attention 和 FFN 输出上
2. **层内 dropout** (`ffn_dropout`, `atn_dropout`)：分别在 FeedForward 内部和 attention 计算中

关键观察：**encoder 和 decoder 的输出没有经过模型级 dropout**（只有 processor 层有）。这意味着 NeuroBridge 的 Target Sequence Decoder 如果直接从 latent tokens 解码，应该遵循相同的模式。

---

## 第三部分：正确但可深化的分析

### 3.1 "PerceiverIO" 的准确性

code_research.md 多次使用"PerceiverIO backbone"来描述 POYO 架构。从 Perceiver IO 论文 (Jaegle et al., 2021) 的角度来看，这个描述是**基本准确的**：
- Cross-attention encoder 将变长输入压缩到固定数量的 latent tokens ✓
- Self-attention processor 在 latent space 中处理信息 ✓
- Cross-attention decoder 从 latent tokens 解码到输出 ✓

但需要注意 POYO 的实现比完整 PerceiverIO 更简化：
- 只有 **1 层 encoder cross-attention**（原始 PerceiverIO 可以有多层迭代）
- 只有 **1 层 decoder cross-attention**（原始 PerceiverIO 同样可以有多层）
- 没有 iterative cross-attention（即 latent tokens 不会多次回看 input tokens）

**对 NeuroBridge 的影响**: 如果预训练中发现单层 encoder cross-attention 不足以充分压缩 spike 信息（特别是当 masking 导致可见 tokens 减少时），可以考虑增加 encoder cross-attention 的层数。这在代码上只需要简单复制 `enc_atn + enc_ffn` 模块。

### 3.2 latent token 的时间网格设计

code_research.md 正确描述了 `create_linspace_latent_tokens` 的功能，但可以更深入地分析其含义：

```python
# tokenizers.py:48
latent_timestamps = np.arange(0, sequence_len, step) + step / 2 + start
```

关键细节：latent tokens 的时间戳是每个 step 的**中心点**（`+ step/2`），而非起始点。

对于 `sequence_length=1.0, latent_step=0.125`:
- 8 个时间步 × 16 个 latents/step = 128 个 latent tokens
- 时间戳: [0.0625, 0.1875, 0.3125, ..., 0.9375]（每个 125ms 窗口的中心）
- 同一时间步的 16 个 latent tokens 共享相同时间戳，但有不同的 latent_index (0-15)

**对 NeuroBridge 的影响**: masking 策略需要考虑这个网格结构。Temporal masking 的粒度最好是 latent_step 的整数倍或半整数倍。如果 mask 一个 125ms 的时间窗口，恰好对应 16 个 latent tokens 完全没有输入可看，这就是 proposal 中提到的"空 chunk 问题"。

### 3.3 可复用组件列表的完整性

code_research.md 第 6.1 节的可复用组件列表整体完整，但建议增加：

| 组件 | 文件路径 | NeuroBridge 用途 |
|------|---------|----------------|
| `POYO.load_pretrained()` | `models/poyo.py:364` | 加载预训练权重 |
| `MultitaskReadout` | `nn/multitask_readout.py` | Target Seq Decoder 的多输出头 |
| `prepare_for_multitask_readout` | `nn/multitask_readout.py:158` | 输出数据准备 |
| `Compose` | `torch_brain/transforms/` | transform 链组合 |
| `register_modality` | `registry.py:49` | 注册新的输出模态（如 MUA firing rate） |

---

## 第四部分：总结与建议

### 4.1 修正优先级

| 优先级 | 问题 | 影响 |
|--------|------|------|
| **高** | CaPOYO T×N tokenization 问题未讨论 | 直接影响 TVSD 数据适配方案选择 |
| **高** | Self-Attention 参数量低估 8 倍 | 影响 NeuroBridge 模型规模设计 |
| **中** | load_pretrained() 未提及 | 影响预训练→下游迁移的实现路径 |
| **中** | Registry 已注册模态未列举 | 影响 Allen 数据利用策略 |
| **低** | 接口参数命名差异 | 编码时注意即可 |
| **低** | TokenType 枚举描述不精确 | 了解即可 |

### 4.2 建议更新

1. **补充 CaPOYO tokenization 的 T×N 分析**，以及对 TVSD MUA 数据的具体影响估算
2. **重新计算各组件参数量**，特别是标注 inner_dim = heads × dim_head 的关系
3. **增加 registry.py 已注册模态的完整列表**，标注哪些与 NeuroBridge 直接相关
4. **补充 load_pretrained() 方法的文档**
5. **补充 varlen attention 路径的说明**
6. **增加"代码默认值 vs 配置文件"的对比表**

---

*本审查基于对 torch_brain 完整源代码的逐文件阅读，所有参数量计算均可通过代码验证。*
