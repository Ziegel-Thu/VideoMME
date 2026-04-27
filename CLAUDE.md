# CLAUDE.md — Video 项目约定

## 项目简介

本项目研究视频理解与多模态大模型评测，围绕 Video-MME 等 benchmark 展开实验与论文工作。

硬件环境：A100 80GB / A40 48GB 单卡。

---

## 文件夹结构

```
video/
├── CLAUDE.md                    # 本文件：行为约束与项目约定
├── paper/                       # 论文 LaTeX 源码仓库（通过脚本下载）
│   └── {名称}/                  # 每篇论文独立目录，含 .tex/.bib/figures
├── scripts/                     # 可复用工具脚本
│   └── download_paper.sh        # arXiv 论文 LaTeX 下载工具
├── experiments/                 # 实验代码（按编号组织）
│   └── {NNN}-{英文短名}/        # 每个实验独立目录
├── docs/                        # 文档
└── plan.md                      # 研究计划（全局状态）
```

### 命名规范

- 实验文件夹：`{三位数字}-{英文短名}`，如 `001-video-mme-eval`
- 脚本文件：`snake_case.py` / `kebab-case.sh`
- 论文文件夹：arXiv ID 或英文短名，如 `video-mme`、`2405.21075`

---

## Skills：论文下载

**下载 arXiv 论文的 LaTeX 源码到 `paper/` 目录。**

```bash
# 基本用法
bash scripts/download_paper.sh <arXiv_ID_or_URL> [自定义文件夹名]

# 示例
bash scripts/download_paper.sh 2405.21075 video-mme
bash scripts/download_paper.sh https://arxiv.org/abs/2405.21075
bash scripts/download_paper.sh 2312.12345 my-paper-name
```

支持输入格式：纯 arXiv ID、abs 链接、pdf 链接。自动解压 tar.gz 并检测主 tex 文件。

---

## tmux 使用规则

**所有耗时较长的任务必须在 tmux session 中运行。** 包括但不限于：

- 模型训练 / 推理
- 大规模视频处理
- 批量评测运行
- 任何预计运行超过 5 分钟的任务

### tmux session 命名规范

- 训练任务：`exp-{编号}-train`
- 推理/评测：`exp-{编号}-eval`
- 数据处理：`exp-{编号}-data`
- 通用长任务：`{描述性短名}`

---

## 实验运行规范

1. **每个实验的代码放在 `experiments/{编号}-xxx/` 下**
2. **每个实验目录应包含：**
   - 主入口脚本（`run.py` / `eval.py`）
   - `config.yaml` 或 argparse 配置
   - `README.md`（如何运行、依赖、文件结构）
3. **训练/推理日志应保存到文件**
4. **分析结果以数值日志为主、图表为辅**：
   - 关键数值 print 到 stdout（不能只生成图片）
   - 目标：不打开任何图片就能判断实验是否成功

---

## 语言规范

- **所有文档、注释、commit message 使用中文**
- 专业术语保留英文（如 Video-MME、MLLM、attention、benchmark 等）
- 代码中的变量名、函数名使用英文

---

## Git 规范

- **及时 commit**：每完成一个有意义的步骤就 commit
- commit message 用中文，简明扼要
- **禁止自动 push**：push 必须由用户手动执行或明确许可
- 不要用 `git add -A`，改用 `git add <具体文件/目录>`
- 不要 `git reset`、`git rebase`、`git push --force` 等破坏性操作

---

## 代码风格

- Python 代码遵循 PEP 8
- 使用 PyTorch 作为主要深度学习框架
- 配置与代码分离
- 可视化使用 matplotlib，保存为 PNG 到实验目录
