# NeuroBridge 数据集下载与使用指南

> 本文档针对 `research_proposal_NeuroBridge.md` 中涉及的核心数据集，重点介绍 **TVSD**（THINGS Ventral-stream Spiking Dataset，主力数据集）以及 **Allen Brain Observatory** 和 **IBL** 在本项目中的适用范围、下载方式、所需数据部分及使用方法，并附各数据集与 NeuroBridge 项目目标的匹配度深度分析。

---

## 目录

1. [数据集总览与角色定位](#1-数据集总览与角色定位)
2. [TVSD 数据集（图像重建核心）](#2-tvsd-数据集)
3. [Allen Brain Observatory（预训练 + 辅助重建）](#3-allen-brain-observatory-数据集)
4. [IBL Brain-wide Map（大规模预训练补充）](#4-ibl-数据集)
5. [数据集匹配度深度分析](#5-数据集匹配度深度分析)
6. [NeuroBridge 推荐执行工作流](#6-neurobridge-推荐执行工作流)

---

## 1. 数据集总览与角色定位

NeuroBridge 的两大支柱是**自监督预训练**（学习通用神经表征）和**脑-图像重建**（下游任务）。这两个目标对数据集的要求截然不同，因此各数据集的角色高度专门化：

| 数据集 | 角色 | 核心优势 | 关键局限 | 磁盘空间 |
|--------|------|---------|---------|---------|
| **TVSD** | 图像重建主力 + 视觉预训练 | 22k张自然图像，V1/V4/IT完整通路，猕猴 | 3.2 TiB总量，MUA非单神经元 | ~3.2 TiB（全量）或按需部分下载 |
| **Allen Neuropixels** | 多脑区预训练 + 辅助重建 | 多区域同时记录，118张Natural Scenes×50次重复 | 仅118张图像，小鼠V1为主 | ~146.5 GB（NWB文件）|
| **IBL Brain-wide Map** | 大规模无视觉刺激预训练 | 459个session，全脑覆盖，最大规模 | 无语义图像，Gabor patch刺激 | 按需下载（每session数百MB）|
| **NLB MC_Maze** | Pipeline验证（仅Phase 0） | torch_brain标准测试用例 | 单session，无视觉刺激 | ~几GB |

**⚠️ 最重要的认识**：NeuroBridge 图像重建任务的成功，**关键取决于 TVSD** 而非 Allen 或 IBL。Allen 仅有 118 张图像，IBL 完全没有语义图像，这两者都不能替代 TVSD 在图像重建中的核心地位。

---

## 2. TVSD 数据集

### 2.1 数据集概况

**TVSD（THINGS Ventral-stream Spiking Dataset）** 由荷兰神经科学研究所（NIN）发布（Papale & Roelfsema, 2024），是目前最大规模的猕猴腹侧视觉通路 spiking 配对图像数据集：

- **发表时间**：2024年11月15日（数据集），配套论文发表于 *Neuron* 2025年1月
- **记录方式**：1,024个微电极，分布在31个 Utah 8×8 阵列上，采样率30kHz
- **物种**：2只成年雄性猕猴（Macaca mulatta）
- **脑区覆盖**：
  - **V1**（初级视觉皮层）：15个阵列，960个电极
  - **V4**（视觉联络皮层）：7个阵列，448个电极
  - **IT**（下颞叶皮层）：9个阵列，576个电极
- **视觉刺激**：来自 [THINGS 数据库](https://things-initiative.org/) 的 **~22,000张自然图像**，覆盖1,854个物体类别（动物、工具、食物、场景等）
- **刺激时长**：每张图像约呈现100-300ms，部分核心图像重复多次
- **数据格式**：原始30kHz电生理信号 + 预处理后的MUA（Multi-Unit Activity，多单元活动）
- **许可证**：Creative Commons Attribution 4.0 (CC BY 4.0)
- **总数据量**：~3.2 TiB（完整数据集）
- **DOI**：https://doi.gin.g-node.org/10.12751/g-node.hc7zlv/

> **MonkeySee（NeurIPS 2024）** 已在 TVSD 上验证了从猕猴 V1/V4/IT spiking 数据重建自然图像的可行性，提供了直接的 baseline 参考。

### 2.2 安装依赖（DataLad + GIN）

TVSD 托管在 G-Node GIN（基于 DataLad/git-annex），支持**按需选择性下载**，无需一次性下载全部 3.2 TiB：

```bash
# 安装 DataLad（推荐通过 conda 或 pip）
conda install -c conda-forge datalad

# 或者 pip 安装
pip install datalad

# 验证安装
datalad --version
```

### 2.3 克隆仓库（仅下载元数据，不下载文件内容）

```bash
# 克隆 TVSD 仓库（只获取文件列表和元数据，几乎不占空间）
datalad clone https://gin.g-node.org/NIN/TVSD ./TVSD_dataset

# 进入目录浏览文件结构
cd TVSD_dataset
ls -la

# ⚠️ 重要：克隆后文件是"符号链接占位符"，实际内容未下载
# 使用 datalad get <文件名> 来按需下载具体文件
```

> **注意**：URL 必须使用 HTTPS 格式（不带 `.git`），否则 git-annex 无法拉取实际文件内容。

### 2.4 需要下载的数据部分（NeuroBridge 视角）

TVSD 的完整数据包含原始 30kHz 信号（体积极大）和预处理后的 MUA。**对于 NeuroBridge，强烈建议只下载预处理后的 MUA 数据**，跳过原始信号：

#### 必须下载（预处理MUA + 刺激信息）

```bash
# 下载预处理后的 MUA 数据（核心神经数据）
datalad get mua/

# 下载刺激信息（图像ID、呈现时间、重复次数等）
datalad get stimuli/

# 下载电极元数据（阵列位置、脑区归属）
datalad get electrodes/

# 下载 THINGS 图像元数据（概念标注、语义相似度矩阵）
datalad get images_metadata/
```

#### 可选下载（按需）

```bash
# 原始 30kHz 原始信号（体积极大，一般不需要）
# datalad get raw/  # ⚠️ 不推荐，体积巨大

# 预处理的 spike-sorted 数据（如有）
datalad get sorted/

# 质量控制报告
datalad get qc/
```

#### THINGS 图像文件本身

TVSD 中的图像来自 THINGS 数据库，需要单独获取：

```bash
# THINGS 图像可以从 OSF 下载（CC BY 4.0）
# https://osf.io/jum2f/  (THINGS 官方图像集)

# 或通过 THINGS Python 包
pip install things-dataset
python -c "from things import THINGS; t = THINGS(); t.download()"

# 建议使用 THINGS 的标准化图像版本（resize 到统一分辨率）
# 用于与 DINOv2/CLIP 预处理对齐
```

### 2.5 数据结构理解

TVSD 的数据组织方式（基于 NIN 发布的说明）：

```
TVSD_dataset/
├── mua/                    # 预处理后的 MUA 数据（核心）
│   ├── monkey1/
│   │   ├── V1/            # V1 脑区 MUA
│   │   ├── V4/            # V4 脑区 MUA
│   │   └── IT/            # IT 脑区 MUA
│   └── monkey2/
│       ├── V1/
│       ├── V4/
│       └── IT/
├── stimuli/                # 刺激信息
│   ├── image_ids.npy      # 每个trial的图像ID
│   ├── onset_times.npy    # 图像呈现时刻
│   └── trial_info.csv     # trial元数据
├── electrodes/             # 电极信息
│   ├── array_positions.csv
│   └── brain_region_labels.csv
├── images_metadata/        # THINGS图像元数据
│   ├── concept_labels.csv
│   └── semantic_similarity.npy
└── raw/                    # 原始30kHz信号（体积极大，按需下载）
```

### 2.6 加载和使用 MUA 数据

```python
import numpy as np
import pandas as pd
from pathlib import Path

data_root = Path('./TVSD_dataset')

# ① 加载 V1 的 MUA 数据（以 monkey1 为例）
# MUA 数据通常是 [n_trials, n_electrodes, n_timepoints] 格式
mua_v1 = np.load(data_root / 'mua/monkey1/V1/mua_data.npy')
print(f"V1 MUA shape: {mua_v1.shape}")
# 例如：(22000, 960, 500) = 22000张图像 × 960个电极 × 500个时间点

# ② 加载刺激信息
image_ids = np.load(data_root / 'stimuli/image_ids.npy')
onset_times = np.load(data_root / 'stimuli/onset_times.npy')
trial_info = pd.read_csv(data_root / 'stimuli/trial_info.csv')
print(f"总 trials: {len(image_ids)}")
print(f"唯一图像数: {len(np.unique(image_ids))}")

# ③ 加载电极元数据（脑区归属）
electrode_info = pd.read_csv(data_root / 'electrodes/brain_region_labels.csv')
v1_electrodes = electrode_info[electrode_info['region'] == 'V1'].index
v4_electrodes = electrode_info[electrode_info['region'] == 'V4'].index
it_electrodes = electrode_info[electrode_info['region'] == 'IT'].index
print(f"V1: {len(v1_electrodes)} 电极, V4: {len(v4_electrodes)} 电极, IT: {len(it_electrodes)} 电极")
```

### 2.7 时间窗口提取（NeuroBridge 关键设计）

NeuroBridge 根据不同脑区的视觉响应时间选择不同时间窗口（提案第 5.7 节）：

```python
def extract_neural_features(mua_data, onset_times, sampling_rate=1000,
                             region='V1'):
    """
    从 MUA 数据中提取指定脑区的视觉响应特征。

    不同脑区的视觉响应峰值时间：
      V1:  stimulus onset 后 30-80ms
      V4:  stimulus onset 后 60-120ms
      IT:  stimulus onset 后 80-200ms
    """
    windows = {
        'V1': (0.030, 0.080),   # 30-80ms post-onset
        'V4': (0.060, 0.120),   # 60-120ms post-onset
        'IT': (0.080, 0.200),   # 80-200ms post-onset
    }
    t_start, t_end = windows[region]

    # 转换为时间点索引（假设数据从 stimulus onset 前 50ms 开始）
    pre_onset_ms = 50
    idx_start = int((pre_onset_ms/1000 + t_start) * sampling_rate)
    idx_end = int((pre_onset_ms/1000 + t_end) * sampling_rate)

    # 提取时间窗口内的平均 firing rate
    # mua_data: [n_trials, n_electrodes, n_timepoints]
    neural_response = mua_data[:, :, idx_start:idx_end].mean(axis=-1)
    # shape: [n_trials, n_electrodes]

    return neural_response


# 使用示例
mua_v1 = np.load('./TVSD_dataset/mua/monkey1/V1/mua_data.npy')
v1_features = extract_neural_features(mua_v1, onset_times, region='V1')
print(f"V1 neural features shape: {v1_features.shape}")
# 例如：(22000, 960) = 22000 trials × 960 electrodes
```

### 2.8 与 CLIP 对齐的预处理流程（Phase 3 准备）

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ① 预加载 CLIP 模型（冻结）
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()

# ② 提取所有 THINGS 图像的 CLIP embedding（离线处理，只做一次）
def extract_clip_embeddings(image_paths, batch_size=64):
    """批量提取 THINGS 图像的 CLIP image embeddings"""
    all_embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        inputs = clip_processor(images=images, return_tensors='pt', padding=True)
        with torch.no_grad():
            embeddings = clip_model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # 归一化
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)  # [n_images, 768]

# 预提取并保存（~22000 张图像约需 10-30 分钟，存储<1GB）
# clip_embeddings = extract_clip_embeddings(all_image_paths)
# torch.save(clip_embeddings, './TVSD_dataset/clip_embeddings_vitl14.pt')

# ③ 构建训练配对：(neural_features, clip_embedding)
# 通过 image_ids 将神经活动与 CLIP embedding 对应
def build_neural_clip_pairs(neural_features, image_ids, clip_embeddings):
    """构建神经活动 - CLIP embedding 配对数据集"""
    # neural_features: [n_trials, n_electrodes]
    # image_ids: [n_trials] 每个 trial 对应的图像 ID（0-indexed）
    # clip_embeddings: [n_unique_images, 768]

    # 对相同图像的多次重复取平均（提升信噪比）
    unique_ids = np.unique(image_ids)
    avg_neural = np.zeros((len(unique_ids), neural_features.shape[1]))
    for i, img_id in enumerate(unique_ids):
        mask = image_ids == img_id
        avg_neural[i] = neural_features[mask].mean(axis=0)

    # 对应的 CLIP embeddings
    avg_clip = clip_embeddings[unique_ids]  # [n_unique_images, 768]

    return avg_neural, avg_clip

avg_neural, avg_clip = build_neural_clip_pairs(v1_features, image_ids, clip_embeddings)
print(f"训练数据: neural {avg_neural.shape}, clip {avg_clip.shape}")
```

### 2.9 与 torch_brain 框架的适配

NeuroBridge 基于 POYO 的 torch_brain 框架，需要将 TVSD 数据适配为 `torch_brain` 的标准格式：

```python
# torch_brain 期望的数据格式
# 参考 POYO 在 torch_brain 中的 NLB dataset 实现

from torch.utils.data import Dataset

class TVSDDataset(Dataset):
    """将 TVSD MUA 数据适配为 torch_brain 格式"""

    def __init__(self, mua_dir, stimuli_dir, regions=['V1', 'V4', 'IT'],
                 monkey_id=1, train=True, train_ratio=0.8):
        self.regions = regions

        # 加载多脑区数据
        self.mua_data = {}
        for region in regions:
            mua_path = f"{mua_dir}/monkey{monkey_id}/{region}/mua_data.npy"
            self.mua_data[region] = np.load(mua_path)

        # 加载刺激信息
        self.image_ids = np.load(f"{stimuli_dir}/image_ids.npy")
        self.onset_times = np.load(f"{stimuli_dir}/onset_times.npy")

        # 按图像 ID 划分 train/test（避免图像泄露）
        unique_ids = np.unique(self.image_ids)
        n_train = int(len(unique_ids) * train_ratio)
        if train:
            self.valid_ids = set(unique_ids[:n_train])
        else:
            self.valid_ids = set(unique_ids[n_train:])

        # 筛选有效 trial
        self.trial_mask = np.array([id_ in self.valid_ids
                                    for id_ in self.image_ids])
        self.valid_trials = np.where(self.trial_mask)[0]

    def __len__(self):
        return len(self.valid_trials)

    def __getitem__(self, idx):
        trial_idx = self.valid_trials[idx]
        image_id = self.image_ids[trial_idx]

        # 提取各脑区的 spike/MUA 数据
        neural_data = {}
        for region in self.regions:
            neural_data[region] = self.mua_data[region][trial_idx]
            # shape: [n_electrodes, n_timepoints]

        return {
            'neural_data': neural_data,  # dict of region -> [electrodes, timepoints]
            'image_id': image_id,
            'trial_idx': trial_idx
        }
```

### 2.10 TVSD 重要注意事项

1. **MUA vs 单神经元**：TVSD 提供的是 MUA（Multi-Unit Activity），即多个神经元的混合信号，不是经过 spike sorting 的单神经元数据。这意味着不能直接使用 POYO 的 per-unit embedding，需要**将每个电极通道视为一个"unit"**，或在更粗粒度上进行建模。这是与 IBL/Allen 数据的关键差异，需要在模型设计中明确处理。
2. **数据量预估**：预处理 MUA 数据（3个脑区，2只猴子）估计约200-500GB；原始30kHz数据约2-3 TiB，建议不下载。
3. **THINGS 图像版权**：THINGS 数据库中的图像来自不同版权方，但原始论文和 MonkeySee 均使用了这些图像。建议仅用于学术研究，不公开再发布。
4. **数据集验证**：下载后建议运行 `datalad fsck` 验证文件完整性。

---

## 3. Allen Brain Observatory 数据集

### 3.1 在 NeuroBridge 中的角色

Allen Brain Observatory Neuropixels Visual Coding 在 NeuroBridge 中承担**双重角色**：

- **预训练数据**（★★★）：多脑区同时记录，天然支持全部4种masking策略（temporal/neuron/intra-region/inter-region masking）
- **辅助图像重建数据集**（★★☆）：118张Natural Scenes，每张重复50次，信噪比极高，可作为 TVSD 的补充验证集

**关键局限**：仅118张图像，不能作为图像重建的主要训练数据（CLIP对齐需要足够的图像多样性）。

### 3.2 安装与下载

```bash
pip install allensdk
```

```python
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

cache_dir = '/your/storage/allen_cache'
manifest_path = os.path.join(cache_dir, 'manifest.json')
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
```

完整下载与使用方法详见 `dataset_download_guide_NeuroHorizon.md`（Allen部分）。

### 3.3 NeuroBridge 专用配置

```python
# 优先使用 brain_observatory_1.1 session（含自然图像）
sessions_table = cache.get_session_table()
bo_sessions = sessions_table[
    sessions_table.session_type == 'brain_observatory_1.1'
]

# NeuroBridge预训练：需要多脑区同时记录的session
# 选取至少记录3个以上脑区的session，以支持inter-region masking
multi_region_sessions = bo_sessions[bo_sessions['probe_count'] >= 4]
print(f"支持多脑区masking的session: {len(multi_region_sessions)}")

# NeuroBridge图像重建：使用Natural Scenes（118张，50次重复）
session_id = bo_sessions.index[0]
session = cache.get_session_data(session_id)
natural_scenes = session.get_stimulus_table('natural_scenes')
# 'frame' 字段 0-117 对应118张图像，可用于映射到 CLIP embedding
```

### 3.4 Allen 数据在 NeuroBridge 中的使用策略

- **Phase 1a-b（masking预训练验证）**：使用 Allen 的小规模子集，因为多脑区同时记录支持全部4种masking策略，是验证预训练框架最理想的数据集。
- **Phase 3a（CLIP对齐）**：将118张图像的神经响应与 CLIP embedding 对齐，作为 TVSD 对齐结果的对比验证（小鼠V1 vs 猕猴IT的对齐难度对比本身就是有价值的科学发现）。
- **Phase 3b（图像重建）**：仅作为辅助评估，不作为主要训练数据。

---

## 4. IBL 数据集

### 4.1 在 NeuroBridge 中的角色

IBL Brain-wide Map 在 NeuroBridge 中仅扮演**大规模预训练补充数据**的角色：

- ✅ 适用：Phase 1c 大规模预训练（与Allen/TVSD混合，增加数据多样性和规模）
- ✅ 适用：跨脑区动力学学习（全脑覆盖，inter-region masking）
- ❌ 不适用：图像重建（仅Gabor patch，无语义图像）
- ⚠️ 有疑问：行为任务预训练是否真正有助于视觉图像重建（见第5节）

### 4.2 下载与使用

完整下载方法详见 `dataset_download_guide_NeuroHorizon.md`（IBL部分）。

NeuroBridge 不需要下载 IBL 的 trials 行为数据，仅需：
- `spikes.times`、`spikes.clusters`（核心spike数据）
- `clusters.label`（质量过滤）
- `clusters.brainLocations.acronym`（脑区归属，用于inter-region masking）

```python
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international')

# NeuroBridge 预训练时，可以不下载 trials 数据
ssl = SpikeSortingLoader(pid=pid, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

# 仅保留 good units
good_mask = clusters['label'] == 1
# 按脑区分组，用于 inter-region masking
if 'acronym' in clusters:
    brain_regions = clusters['acronym'][good_mask]
```

---

## 5. 数据集匹配度深度分析

### 5.1 整体评估矩阵

| 评估维度 | TVSD | Allen | IBL |
|---------|------|-------|-----|
| **自监督预训练** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **图像重建（图像多样性）** | ⭐⭐⭐⭐⭐ (22k张) | ⭐⭐ (118张) | ❌ (无语义图像) |
| **脑区层级（视觉通路完整性）** | ⭐⭐⭐⭐⭐ (V1→V4→IT) | ⭐⭐⭐ (主要V1) | ⭐⭐ (含视觉皮层但非重点) |
| **与CLIP语义空间的对齐难度** | 低（IT皮层高层语义）| 高（小鼠V1低层特征）| 极高（无视觉语义）|
| **数据规模（sessions/trials）** | ⭐⭐ (2只猴) | ⭐⭐ (58 sessions) | ⭐⭐⭐⭐⭐ (459 sessions) |
| **数据格式与torch_brain兼容性** | ⚠️ 需适配（MUA非spike-sorted）| ✅ POYO+已验证 | ✅ POYO+已验证 |
| **获取便利性** | ⚠️ 3.2TiB，部分下载可行 | ✅ AllenSDK自动化 | ✅ ONE API自动化 |

### 5.2 TVSD 的核心地位与关键风险

**TVSD 是 NeuroBridge 项目成功的关键**，但存在以下风险需要早期评估：

**风险1：MUA 数据与 spike-level tokenization 的兼容性**

NeuroBridge 基于 POYO 的 spike-level tokenization（每个 spike 是一个 token），但 TVSD 提供的是预处理后的 MUA 信号（连续时间序列，非离散 spike events）。这意味着：

- 选项A：将 MUA 视为连续信号，每个时间点 × 每个电极作为一个 token（类似 binned representation，与 POYO 设计精神不完全一致）
- 选项B：对 MUA 进行阈值处理，提取近似的 spike 时刻（精度不如 spike-sorted 数据，但保持与框架一致）
- 选项C：在 TVSD 上使用简化的 binned MUA 表示，在 Allen/IBL（有 spike-sorted 数据）上使用完整 spike-level tokenization，两种数据通过共享 PerceiverIO 统一处理

**建议**：在 Phase 0 环境搭建阶段，就需要明确 TVSD MUA 数据的处理策略，这不是后期可以忽视的细节，而是直接影响整个预训练框架设计的关键决策。

**风险2：图像数量与CLIP对齐质量**

22,000张图像听起来很多，但相比于 fMRI 领域（MindEye2 使用了 ~30,000个 unique stimuli），仍然在同一量级。关键问题是：**每张图像的重复次数够不够**？MonkeySee 论文中提到每张图像重复2-5次。取平均后的信噪比可能比 Allen（每张图50次重复）低得多。建议优先对高重复次数的图像子集做 CLIP 对齐，验证对齐质量后再扩展到全部22,000张。

**风险3：数据获取时间**

3.2 TiB 的完整数据集，即使使用 DataLad 部分下载（只下 MUA），预计也需要下载数百 GB。在执行计划 Phase 0 中，应将 TVSD 数据的下载和适配列为**关键路径任务**，并发进行，而不是等到 Phase 3 才处理。

### 5.3 Allen Brain Observatory 的关键局限

**118张图像是图像重建任务的硬天花板。**

MindEye2 实现高质量 fMRI 图像重建所用的关键设计是：用大量图像（20,000+）在 CLIP 空间中构建密集的语义覆盖，使得神经表征可以通过 k-NN 检索到语义接近的 CLIP embedding，再驱动 Stable Diffusion。

Allen 的118张图像在 CLIP 空间中只能覆盖极小的语义范围，这意味着：
- 对于训练集中见过的图像，重建质量可能还不错
- 但泛化到新图像时，性能会显著下降
- 图像重建的 top-1 retrieval accuracy 天花板极低（全部图像池只有118张）

**建议**：Allen 数据的图像重建实验应被定位为"概念验证"而非"主要结果"，主要展示不同脑区（V1 vs IT）的贡献差异，而非与 MonkeySee/MindEye 的直接比较。主要的图像重建结果必须在 TVSD 上产生。

### 5.4 IBL 预训练迁移假设的合理性

这是整个 NeuroBridge 数据策略中最需要慎重考虑的问题：**用行为任务（IBL决策任务）数据预训练的神经表征，是否真的有助于视觉图像重建？**

支持迁移的论据：
- 自监督预训练学到的是通用神经动力学模式（timing structure、firing rate statistics、跨神经元相关性），这些特征与特定任务无关
- MtM 和 NEDS 已经证明在 IBL 上预训练后，在其他任务上的线性 probe 性能更好
- 更多数据 = 更好的预训练，即使来自不同领域

反对迁移的论据：
- IBL 记录的是全脑决策回路，而视觉图像重建依赖的是视觉皮层的高层表征
- IBL 预训练可能让模型偏向"决策相关的动力学"，而非"视觉语义相关的动力学"
- Jiang et al. (2025) 发现简单数据积累因异质性损害而非帮助 scaling

**实验建议**：在 Phase 1c 混合预训练后，应该做一个**消融实验**：(a) 仅用视觉数据（Allen+TVSD）预训练 vs (b) 混合全部数据（Allen+TVSD+IBL）预训练，比较两者在 Phase 3 图像重建任务上的性能差异。如果（b）不优于（a），则应完全去掉 IBL，专注视觉数据。

### 5.5 物种差异的隐患

NeuroBridge 混用了两个物种的数据：
- 小鼠（Allen、IBL）：皮层组织结构、神经元密度、视觉系统复杂度与人类差距较大
- 猕猴（TVSD、JiaLab、NLB）：与人类更接近，尤其是腹侧视觉通路

对于图像重建任务，理想情况下应该在**同一物种**的数据上做，因为 CLIP embedding 编码的是人类视觉感知的语义空间。猕猴（特别是 IT 皮层）与人类 CLIP 空间的对应关系比小鼠 V1 更直接。这意味着：

- TVSD（猕猴，V1/V4/IT）→ CLIP 对齐：相对合理
- Allen（小鼠，主要V1）→ CLIP 对齐：挑战更大，需要更多的映射层

在论文中，这种差异本身可以作为一个有趣的科学发现来讨论：不同物种、不同脑区的神经表征与 CLIP 语义空间的对齐难度，以及 IT 皮层在视觉语义编码中的核心作用。

---

## 6. NeuroBridge 推荐执行工作流

### 6.1 数据获取优先级

```
Week 1：
  □ 克隆 TVSD GIN 仓库（只下元数据），浏览文件结构
  □ 下载 Allen NLB MC_Maze（几GB，用于 pipeline 验证）
  □ 搭建 torch_brain 环境，复现 POYO baseline

Week 2：
  □ 开始下载 TVSD MUA 数据（仅 mua/ 目录，估计100-500GB）
  □ 并行下载 THINGS 图像（来自 OSF）
  □ 下载 Allen 前5个 brain_observatory_1.1 session（~15GB）
  □ 编写 TVSD → torch_brain 数据适配器，验证数据流通

Week 3-4：
  □ 下载 Allen 全部58个 session（~146.5GB）
  □ 预提取所有 THINGS 图像的 CLIP embedding（离线处理，~1小时，<1GB）
  □ 确认 TVSD MUA 数据的 spike-level tokenization 处理策略
  □ 开始 IBL 数据下载（先下10-20个 session 做初步验证）
```

### 6.2 各阶段数据集分配

| 项目阶段 | 使用数据集 | 目的 | 注意事项 |
|---------|----------|------|---------|
| Phase 0: Pipeline验证 | NLB MC_Maze | 复现POYO baseline | 确认torch_brain环境正确 |
| Phase 1a: 单任务masking | Allen（小规模，5-10 sessions）| 验证masking预训练基本功能 | 多脑区支持所有4种masking |
| Phase 1b: 多任务masking | Allen（全量，58 sessions）| 验证4种masking策略协同 | 重点验证inter-region masking效果 |
| Phase 1c: 大规模预训练 | Allen + TVSD + IBL | 混合数据大规模预训练 | 使用√N采样避免大数据集主导 |
| Phase 2: 跨session泛化 | Allen + IBL | 验证unit embedding跨session迁移 | TVSD只有2只猴，不适合跨session |
| Phase 3a: CLIP对齐 | TVSD（主）+ Allen Natural Scenes（辅）| 训练Neural Projector | TVSD是主力，Allen仅作对比 |
| Phase 3b: 图像重建 | TVSD（主要评估）+ Allen（辅助）| 重建质量评估，与MonkeySee对比 | Allen结果仅作概念验证 |
| Phase 4: 消融实验 | 所有数据集 | 验证各模块贡献 | 特别验证IBL预训练是否真正有效 |

### 6.3 存储空间规划

| 数据集 | 建议下载量 | 预估存储空间 |
|--------|----------|------------|
| TVSD（仅MUA + 刺激信息，不含原始30kHz）| 全量 | ~200-500 GB |
| THINGS 图像（22k张，标准分辨率）| 全量 | ~10-20 GB |
| THINGS CLIP embeddings（预提取）| 全量 | <1 GB |
| Allen NWB（spike + behavior，不含LFP）| 58 sessions | ~146.5 GB |
| IBL（spike + clusters，不含LFP）| 按需，50-100 sessions | ~50-100 GB |
| NLB MC_Maze | 全量 | ~5 GB |
| **合计** | | **~400-770 GB** |

### 6.4 关键参考资源

- **TVSD 数据集**：https://doi.gin.g-node.org/10.12751/g-node.hc7zlv/
- **TVSD 配套论文**（Neuron 2025）：https://www.cell.com/neuron/abstract/S0896-6273(24)00881-X
- **MonkeySee（NeurIPS 2024）**：https://proceedings.neurips.cc/paper_files/paper/2024/file/aa7eb65738b5bc71c81848fba9111c97-Paper-Conference.pdf
- **Monkey Perceptogram**（TVSD + TITD 后续工作）：https://arxiv.org/abs/2510.07576
- **DataLad GIN 下载指南**：https://handbook.datalad.org/en/latest/basics/101-139-gin.html
- **THINGS 图像数据库**：https://things-initiative.org/
- **torch_brain / POYO 代码库**：https://github.com/mehdiazabou/poyo-1
- **Allen SDK 文档**：https://allensdk.readthedocs.io/
- **IBL ONE API 文档**：https://docs.internationalbrainlab.org/

---

*本文档基于 2025年2月 整理。TVSD 数据集于 2024年11月公开，相关工具链仍在快速更新中，建议定期查阅 GIN 仓库和 MonkeySee 代码库获取最新处理方法。*
