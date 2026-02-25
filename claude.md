# NeuroBridge 项目 Claude 工作规范

## 项目背景

**NeuroBridge** 是一个基于 **POYO 框架**（本仓库 `torch_brain` 代码库）的神经科学与计算机视觉交叉研究项目，目标是通过多任务掩码预训练实现通用神经表征，并从灵长类动物 Spiking 数据实现 Brain-to-Image 重建。

详细项目信息请参考以下文档：

- 研究方案设计：`cc_core_files/proposal.md`
- POYO 代码库分析：`cc_core_files/code_research.md`
- 数据集选型与规划：`cc_core_files/dataset.md`
- 项目执行计划：`cc_core_files/plan.md`

---

## 当前项目状态

**项目计划优化阶段**

目前正在进一步优化和修正以下项目规划文档：
- 研究方案设计（`cc_core_files/proposal.md`）
- POYO 代码库分析（`cc_core_files/code_research.md`）
- 数据集选型与规划（`cc_core_files/dataset.md`）
- 执行计划制定（`cc_core_files/plan.md`）

上述规划文档完全确定之后，才会开始正式的代码改造/项目执行。

> 待文档进一步修缮后，将在此处标记：
> - 项目详细规划参考 `proposal.md`
> - 项目执行计划参考 `plan.md`（按照 plan.md 进行规划和执行，任务完成后打勾，进行全局项目进展记录）

---

## 重要注意事项

### 已放弃任务（请勿参考）

`cc_todo/20260221-cc-1st/` 文件夹中的内容为**已放弃的历史任务记录**，代码已回退至该次任务之前的状态。

- **任何时候都不要参考该文件夹中的任何内容**
- 如果在执行任务中遇到与其相关的内容，需留心检查是否存在未回退的代码或配置

---

## 任务执行规范

### 1. 任务开始前

- 读取 `progress.md`，了解之前做了什么，有无可借鉴的经验
- 读取 `claude.md`（即本文件），确认当前项目规范和注意事项
- 如果当前任务对应 `cc_core_files/plan.md` 中的某项任务，在 plan.md 对应位置标记任务开始

### 2. 任务执行中

- 遇到问题时，优先查阅 `progress.md` 和 `claude.md` 是否有相关经验记录
- 新建文件夹和文件名使用**英文**
- Markdown 文档内容主要使用**中文**
- Markdown 文档命名在最前面包含产出日期（格式：`YYYYMMDD-文档名.md`）

### 3. plan.md 任务追踪

如果当前执行任务刚好是 `cc_core_files/plan.md` 中的任务，则需要在 plan.md 对应任务位置记录该任务的执行情况，包括：
- 完成状态（完成 / 进行中 / 待开始）
- 已做到什么程度
- 接下来��需要做什么

### 4. 脚本管理

- 脚本文件存放于 `scripts/` 下的合适位置
- **必须**在 `cc_core_files/scripts.md` 记录：脚本功能用途、创建时间、使用方式、存储位置

### 5. 数据管理

- 下载的数据集存放于 `data/` 下的合适位置
- 按数据类型分类管理（如 `raw/`、`processed/`、`generated/` 等）
- **必须**在 `cc_core_files/data.md` 记录：数据处理信息、存储位置、数据含义等

### 6. 结果管理

- 实验分析结果（可视化图表等）存放于 `results/` 下的合适位置
- **必须**在 `cc_core_files/results.md` 记录：产生方式、目的、结果分析等

### 7. 任务完成后

- 及时更新 `progress.md`，记录以下内容：
  - 完成时间（日期-小时-分）
  - 当前任务做了什么
  - 任务结果与效果
  - 执行过程中遇到的问题及解决方法
  - **注意：progress.md 中不记录待开展工作**
- 如果任务对应 `cc_core_files/plan.md` 中的某项任务，在 plan.md 对应位置记录执行情况
- 检查 `claude.md` 是否有过时内容，如有则及时更新
- 将多次任务中遇到的共性问题沉淀为经验，记录到本文件"经验积累"部分

### 8. Git 提交规范

- 每次完成任务或小的模块后，及时执行 `git add & commit & push`
- Git commit 消息使用**中文**
- Git 账户：`jpagkr@163.com`，用户名：`Jp-17`

---

## 目录结构说明

```
.
├── torch_brain/           # POYO 框架核心代码库（尽量不做破坏性修改）
├── cc_core_files/         # NeuroBridge 项目核心文档
│   ├── proposal.md        # 研究方案
│   ├── plan.md            # 执行计划
│   ├── code_research.md   # 代码分析
│   ├── dataset.md         # 数据集规划
│   ├── scripts.md         # 脚本记录（脚本信息必须在此登记）
│   ├── data.md            # 数据记录（数据信息必须在此登记）
│   └── results.md         # 结果记录（结果信息必须在此登记）
├── cc_todo/               # 历史任务记录
│   └── 20260221-cc-1st/   # ⚠️ 已放弃的任务，勿参考，代码已回退
├── scripts/               # 项目脚本
├── data/                  # 数据集
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── generated/         # 生成数据
├── results/               # 实验结果
├── examples/              # POYO 框架示例
├── notebooks/             # Jupyter 教程
└── tests/                 # 单元测试
```

---

## 经验积累

（随任务执行逐步积累，暂无记录）
