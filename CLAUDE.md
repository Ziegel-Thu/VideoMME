# CLAUDE.md — Video 项目约定

## 项目简介

本项目研究视频理解与多模态大模型评测，核心方向：在 VLM 中加 latent visual bottleneck 做 grounded video reasoning。

硬件环境：A100 80GB 单卡。

---

## ⚠️ 实验纪律（必须遵守）

### 1. 先验证再扩大，每一步都要 sanity check
- **写完代码先 overfit 1 batch**：loss 降不到 0 就不要往下走
- **第一个 epoch 完成后立刻跑 sanity check**：不要等全部 epoch 跑完
- **sanity check 必须包含**：blank vision / shuffled vision / 正常对照
- 如果 blank ≈ normal → 模型没在用视觉信息，**立刻停训查原因**

### 2. 不要从零写，要基于参考实现改
- 先读参考论文的代码（VoCo-LLaMA、LIVR、LVR 等），理解它们怎么实现的
- 在已有框架上改，不要自己拼 —— 减少 bug 概率
- 每次写新模块前，先用 rubber-duck agent review 设计

### 3. 理解了再动手
- 新模块写之前，先能口头解释清楚"这个模块做什么、输入输出是什么、为什么这样设计"
- 不确定导师方案某句话的含义时，**停下来问用户**，不要自己猜着往前冲
- 导师说"或 MCQ CE"这种退化选项，要理解它在什么条件下适用

### 4. 数据必须匹配任务
- 训练 bottleneck 时，数据必须**视觉依赖**——不看图答不对题
- temporal QA（答案是时间段）不适合验证 bottleneck，因为靠语言 prior 就能猜
- MCQ / 图片描述 / visual QA 才适合

### 5. 文档实时更新
- **plan.md**：每做完一个 phase / 发现重要问题 / 方向修正时，立即更新
- **实验 README.md**：记录当前配置、已知问题、结果
- **commit message**：说清楚改了什么、为什么改
- 不要等到被问"有在记录吗"才去更新

---

## 文件夹结构

```
video/
├── CLAUDE.md                    # 本文件：行为约束与项目约定
├── paper/                       # 论文 LaTeX 源码仓库（通过脚本下载）
│   └── {名称}/                  # 每篇论文独立目录，含 .tex/.bib/figures
├── scripts/                     # 可复用工具脚本
│   ├── download_paper.sh        # arXiv 论文 LaTeX 下载工具
│   └── parse_training_data.py   # 数据解析脚本
├── experiments/                 # 实验代码（按编号组织）
│   └── {NNN}-{英文短名}/        # 每个实验独立目录
├── data/                        # 训练数据和视频（gitignore）
└── plan.md                      # 研究计划（session state 里）
```

### 命名规范

- 实验文件夹：`{三位数字}-{英文短名}`，如 `002-latent-bottleneck`
- 脚本文件：`snake_case.py` / `kebab-case.sh`
- 论文文件夹：arXiv ID 或英文短名，如 `video-mme`、`2405.21075`

---

## Skills：论文下载

**下载 arXiv 论文的 LaTeX 源码到 `paper/` 目录。**

```bash
bash scripts/download_paper.sh <arXiv_ID_or_URL> [自定义文件夹名]
bash scripts/download_paper.sh 2405.21075 video-mme
```

---

## tmux 使用规则

**所有耗时较长的任务必须在 tmux session 中运行。**

### tmux session 命名规范

- 训练任务：`bn-on` / `bn-off`（ablation 对比）
- 推理/评测：`eval-{描述}`
- 通用长任务：`{描述性短名}`

---

## 实验运行规范

1. **每个实验的代码放在 `experiments/{编号}-xxx/` 下**
2. **每个实验目录应包含**：README.md、主入口脚本、配置
3. **训练日志保存到文件**（`tee` 到 log 文件）
4. **分析结果以数值为主**：关键数值 print 到 stdout

---

## 语言规范

- **所有文档、注释、commit message 使用中文**
- 专业术语保留英文（如 bottleneck、latent、attention 等）
- 代码中的变量名、函数名使用英文

---

## Git 规范

- **及时 commit**：每完成一个有意义的步骤就 commit
- commit message 用中文，简明扼要
- **禁止自动 push**：push 必须由用户手动执行或明确许可
- 不要用 `git add -A`，改用 `git add <具体文件/目录>`
- 不要 `git reset`、`git rebase`、`git push --force`

---

## 代码风格

- Python 代码遵循 PEP 8
- 使用 PyTorch 作为主要深度学习框架
- 配置与代码分离
- 可视化使用 matplotlib，保存为 PNG
