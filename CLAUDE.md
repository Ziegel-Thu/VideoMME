# CLAUDE.md — Video 项目约定

## 项目简介

本项目研究视频理解与多模态大模型评测，核心方向：在 VLM 中加 latent visual bottleneck 做 grounded video reasoning。

硬件环境：A100 80GB 单卡。

---

## ⚠️ 实验纪律（必须遵守）

### 0. 时刻清楚项目目标和当前阶段
- **项目目标**：用 latent visual bottleneck 做 grounded long video reasoning
  - 输入：长视频 + 问题 → 输出：答案 + 证据时间段
  - 方法：latent tokens + bottleneck mask + VoCo 压缩 + Temporal Head
  - Claim：bottleneck 迫使 latent 承载视觉信息，单次前向做 grounding
- **训练路线**：Stage 1 VoCo(可跳) → Stage 2 Bottleneck SFT(MCQ) → Stage 3 Temporal Head(temporal data)
- **启动任何实验前，先确认"当前在哪个 Stage，用什么数据，训什么参数"**

### 1. 训练前必检清单（每次启动训练前逐条确认）

**A. 数据检查**
- [ ] 打印数据总量
- [ ] 需要的标签字段存在吗？（如 L_temp 需要 evidence_segments 不全是 None）
- [ ] 数据是否匹配任务？（L_temp 需要 temporal 标签，L_ans 需要 answer）
- [ ] 打印 3 条完整样本，人工确认格式正确

**B. 模型检查**
- [ ] 哪些参数在训练？哪些冻住了？打印 trainable params
- [ ] 如果加载 checkpoint，验证加载成功（跑 1 step 看 loss 是否合理）
- [ ] mask 是否生效？（用 attention weight 验证）

**C. Loss 检查**
- [ ] 明确写出 loss 公式，每个 loss 对应什么数据
- [ ] 确认每个 loss 项都有非零梯度（不会因为标签全 None 导致某个 loss 永远为 0）

**D. Overfit + Sanity**
- [ ] overfit 1 batch 验证 loss 能降
- [ ] 第 1 epoch 后跑 sanity check，不要等全部 epoch

### 2. tmux 启动命令检查
- [ ] 命令中的路径存在（checkpoint、数据文件、output_dir）
- [ ] 参数完整（K、num_frames、mask_mode、lr、epochs 等都显式指定）
- [ ] 日志 tee 到文件（`2>&1 | tee log_xxx.txt`）
- [ ] 确认 GPU 显存足够（`nvidia-smi` 检查）
- [ ] 不会和正在跑的进程冲突

### 3. 不要从零写，要基于参考实现改
- 先读参考论文的代码（VoCo-LLaMA、LIVR、LVR 等），理解它们怎么实现的
- 在已有框架上改，不要自己拼 —— 减少 bug 概率
- 每次写新模块前，先用 rubber-duck agent review 设计

### 4. 理解了再动手
- 新模块写之前，先能口头解释清楚"这个模块做什么、输入输出是什么、为什么这样设计"
- 不确定导师方案某句话的含义时，**停下来问用户**，不要自己猜着往前冲

### 5. 数据必须匹配任务
- 训练 bottleneck 时，数据必须**视觉依赖**——不看图答不对题
- 训练 temporal head 时，数据必须有 **temporal 标注**
- 每次换数据前确认标签字段非空

### 6. 文档实时更新
- **plan.md**：每做完一个 phase / 发现重要问题 / 方向修正时，立即更新
- **实验 README.md**：记录当前配置、已知问题、结果
- **commit message**：说清楚改了什么、为什么改
- 不要等到被问"有在记录吗"才去更新

### 7. 操作红线
- **不自作主张 kill 正在跑的进程**——先问用户
- **push 由用户手动执行**
- **不自作主张删数据或 checkpoint**——先问用户

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
