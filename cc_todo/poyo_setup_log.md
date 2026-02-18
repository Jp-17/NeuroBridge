# POYO 模型环境搭建与测试记录

**日期**：2026-02-19
**目标**：在新的 conda 环境中完成 POYO 模型的环境配置、数据集下载，并成功运行端到端训练测试。

---

## 一、背景与策略

### 项目简介
[torch_brain](https://github.com/neuro-galaxy/torch_brain) 是一个面向神经科学的深度学习库，其中 POYO（A Unified, Scalable Framework for Neural Population Decoding）是发表于 NeurIPS 2023 的神经群体解码模型。

### 测试策略
选用 **MC Maze Small** 配置（`train_mc_maze_small.yaml`）作为测试目标，原因：
- 使用最小模型（1.3M 参数，vs 完整版 11.8M）
- 只需下载一个数据集（`pei_pandarinath_nlb_2021`，约 51MB）
- 与官方 CI 集成测试命令完全一致，可靠性高

---

## 二、操作步骤

### Step 1：创建 conda 环境

```bash
conda create -n poyo python=3.10 -y
```

- 环境名：`poyo`
- Python 版本：3.10（与官方 CI 一致）
- 安装路径：`miniconda3/envs/poyo/`

### Step 2：安装依赖

**安装 torch_brain 及其 dev 依赖（本地源码，可编辑模式）：**

```bash
conda run -n poyo pip install -e "/root/autodl-tmp/torch_brain[dev]"
```

主要安装的包：
| 包名 | 版本 | 说明 |
|------|------|------|
| torch | 2.10.0+cu128 | PyTorch（带 CUDA 12.8 支持） |
| torch_brain | 0.1.0 | 本地 dev 安装 |
| temporaldata | 0.1.3 | 时序神经数据格式库 |
| hydra-core | 1.3.2 | 配置管理框架 |
| torchmetrics | 1.8.2 | 评估指标库 |
| brainsets | 0.2.0 (PyPI) | 神经数据集工具（后续替换） |

**安装额外依赖：**

```bash
conda run -n poyo pip install lightning "torch-optimizer==0.3.0" wandb
```

| 包名 | 版本 | 说明 |
|------|------|------|
| lightning | 2.6.1 | PyTorch Lightning 训练框架 |
| torch-optimizer | 0.3.0 | 包含 SparseLamb 优化器（train.py 需要） |
| wandb | 0.25.0 | 实验记录（本次测试禁用） |

**重装 brainsets（从 GitHub 源）：**

PyPI 版本 0.2.0 缺少 `brainsets.datasets` 模块，而 `examples/poyo/datasets/nlb.py` 需要从该模块导入 `PeiPandarinathNLB2021`。官方 CI 使用 GitHub 源，因此替换安装：

```bash
conda run -n poyo pip install --force-reinstall "git+https://github.com/neuro-galaxy/brainsets"
```

安装版本：`0.2.1.dev4+g073ac8ac5`（commit `073ac8ac`）

### Step 3：配置 brainsets 数据目录

```bash
brainsets config \
  --raw-dir /root/autodl-tmp/datasets/raw \
  --processed-dir /root/autodl-tmp/datasets/processed
```

配置文件生成于：`/root/.brainsets.yaml`

### Step 4：下载数据集

```bash
conda run -n poyo brainsets prepare pei_pandarinath_nlb_2021
```

- brainsets 自动创建隔离的 Python 3.11 环境（通过 `uv`），安装 `dandi` 等下载依赖
- 从 DANDI Archive 下载 NWB 原始数据，处理后生成 `.h5` 文件

**数据集信息：**
| 项目 | 内容 |
|------|------|
| 数据集名 | pei_pandarinath_nlb_2021（Neural Latents Benchmark 2021） |
| 任务 | 猕猴 MC Maze 任务（手部 2D 速度解码） |
| 总大小 | 约 51 MB |
| 存储路径 | `autodl-tmp/datasets/processed/pei_pandarinath_nlb_2021/` |
| 生成文件 | `jenkins_maze_train.h5`、`jenkins_maze_test.h5` |

### Step 5：运行训练测试

```bash
conda run -n poyo --cwd .../examples/poyo python train.py \
  --config-name train_mc_maze_small.yaml \
  data_root=/root/autodl-tmp/datasets/processed \
  wandb.enable=false \
  epochs=2 \
  eval_epochs=2 \
  optim.lr_decay_start=0.
```

参数说明：
- `--config-name train_mc_maze_small.yaml`：使用 MC Maze Small 配置（1.3M 模型 + NLB 数据集）
- `data_root=...`：指向自定义数据集路径
- `wandb.enable=false`：禁用 W&B 日志（无需账号）
- `epochs=2`：快速测试，仅跑 2 个 epoch
- `optim.lr_decay_start=0.`：从第 0 个 epoch 开始学习率衰减（适配 2 epoch 测试）

---

## 三、问题与解决

### 问题 1：brainsets PyPI 版本缺少 `datasets` 模块

**现象：**
```
ModuleNotFoundError: No module named 'brainsets.datasets'
hydra.errors.InstantiationException: Error locating target 'datasets.nlb.PoyoNLBDataset'
```

**原因：** PyPI 发布的 `brainsets==0.2.0` 尚未包含 `datasets` 子模块，但 GitHub 主分支已有。

**解决：** 从 GitHub 源重新安装：
```bash
pip install --force-reinstall "git+https://github.com/neuro-galaxy/brainsets"
```

### 问题 2：`brainsets prepare` 需要 `dandi` 模块

**现象：** 使用 `--use-active-env` 运行时报错：
```
ModuleNotFoundError: No module named 'dandi'
RuntimeError: dandi package not present, and is required
```

**解决：** 不使用 `--use-active-env` 标志，让 `brainsets` 自动通过 `uv` 创建隔离环境并安装 `dandi==0.71.3`（这是 `brainsets prepare` 的默认行为）。

---

## 四、测试结果

```
GPU available: True (cuda), used: True

   | Name                 | Type                   | Params
--------------------------------------------------------------
0  | model                | POYO                   | 1.3 M
...

Training on 141 samples, 142 units, 1 sessions

Epoch 1/1 ━━━━━━━━━━━━━━━━━━ 1/1  train_loss: 2.843

Validation metrics:
  jenkins_maze_train    : -0.034835
  average_val_metric    : -0.034835

Test metrics:
  jenkins_maze_train    : -0.024435
  average_test_metric   : -0.024435
```

**结果说明：**
- GPU（CUDA）正常调用
- 模型 1.3M 参数成功构建和训练
- 2 epoch 后 R² ≈ -0.03，属于正常初始值（模型尚未收敛，完整训练需 1000 epoch）
- Checkpoint 已保存至 `examples/poyo/logs/lightning_logs/version_0/checkpoints/`

---

## 五、后续正式训练命令

激活环境后，在 `examples/poyo/` 目录下运行：

```bash
conda activate poyo

# MC Maze Small（最小配置，单数据集）
python train.py --config-name train_mc_maze_small.yaml \
  data_root=/root/autodl-tmp/datasets/processed \
  wandb.enable=false

# POYO-MP（需先下载 perich_miller_population_2018 数据集）
# brainsets prepare perich_miller_population_2018
# python train.py --config-name train_poyo_mp.yaml \
#   data_root=... wandb.enable=false
```

---

## 六、关键路径汇总

| 项目 | 路径 |
|------|------|
| torch_brain 源码 | `autodl-tmp/torch_brain/` |
| POYO 训练脚本 | `autodl-tmp/torch_brain/examples/poyo/train.py` |
| 配置文件目录 | `autodl-tmp/torch_brain/examples/poyo/configs/` |
| 数据集（processed） | `autodl-tmp/datasets/processed/` |
| brainsets 配置 | `/root/.brainsets.yaml` |
| 训练 checkpoint | `autodl-tmp/torch_brain/examples/poyo/logs/` |
| conda 环境 | `miniconda3/envs/poyo/` |
