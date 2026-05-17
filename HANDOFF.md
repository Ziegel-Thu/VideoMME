# 项目交接文档

## 必读文件

1. **`CLAUDE.md`**（项目根目录）—— 项目约定、实验纪律、操作红线。**必须完整读完再动手。**
2. **`experiments/004-voco-segment-compression/README.md`** —— 当前实验的技术细节、shard cache 规范、数据状态、评测结果
3. **`plan.md`**（session state）—— 当前进度和后续计划

## 核心约定

- **不自作主张 kill 进程、改参数、push 代码**——先问用户
- **每完成一项必须先更新 README 再继续下一项**
- **训练/提取只用本地 SSD（`/nvmessd/...`）**，禁止从 NFS 读 teacher cache
- **commit message 用中文**
- **所有长任务在 tmux 中运行**

## 当前正在跑的任务

当前没有需要继续监控的 110K teacher shard 提取任务。gpu8 单机 8×A40 提取已经完成，tmux `extract-110k` 已退出。

## 110K teacher shard cache 完成状态

| 机器 | tmux session | 卡数 | 进度 | 成功 | 跳过 | 错误 | shard | cache 大小 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpu8 | `extract-110k`（已退出） | 8 | 13370/13370 | 13194 | 0 | 176 | 203 | 3.9T |

**日志位置**：`/nvmessd/lifanhong/video/log_extract_teacher_110k.txt`（2.8M）
**输出位置**：`/nvmessd/lifanhong/video/teacher_cache_110k_sharded_256/`
**shard 格式**：`teacher_shard_rank{N}_{idx:03d}.pt`，每个 shard 约 512 条，每条带 `global_idx`
**说明**：输出目录名沿用早期 `teacher_cache_110k_sharded_256`，实际提取命令使用 `--shard_size 512 --prefetch_workers 0 --resume`。错误样本主要是视频 decode/ffmpeg packet 错误，提取进程已跳过并完成。

## 当前活跃 schedule

- 无。110K 提取完成后应停止相关监控 schedule。

## 提取完成后要做的事

1. 用 `/nvmessd/lifanhong/video/teacher_cache_110k_sharded_256/` 启动 110K compressor 训练（B-1L 优先）。
2. 训练前按 `CLAUDE.md` 做数据、模型、loss、overfit sanity 检查。
3. 训练完成后更新 `experiments/004-voco-segment-compression/README.md` 并 commit。

## 已完成的实验结果

**Compressor 10K 评测（统一用 `0_30` 视频目录，200 条 test）：**

| 配置 | epoch1 | epoch2 | epoch3 | epoch4 | epoch5 |
|---|---|---|---|---|---|
| B-1L | 66.5% | 71.0% | **71.0%** | 67.5% | 68.5% |
| B-2L | - | - | 69.0% | - | - |
| D-1L | - | - | 63.0% | 65.0% | 62.5% |
| BD-1L | - | - | 69.5% | - | - |

**Baseline**：Qwen2.5-VL zero-shot logit = 82%
**最佳**：B-1L epoch3 = 71.0%（10K 数据已饱和）

## 后续计划（按优先级）

1. **110K 蒸馏训练**（等 cache 提取完）
2. **段间 cross-attention**：压缩后的段间信息交互
3. **端到端 NLL 训练**：用 cache 中的 dense segments + compressor + frozen LLM → NLL(answer)
4. **Temporal Head**（Stage 3）

## 关键文件清单

| 文件 | 作用 |
|---|---|
| `compressor.py` | Cross-Attention 压缩模块（VoCoCompressor） |
| `train_compressor.py` | compressor 蒸馏训练，shard-only cache，NFS 拒绝 |
| `eval_compressor.py` | 评测（只用 compressed tokens，不混 dense vision） |
| `extract_teacher.py` | teacher 特征提取，支持 `--start_idx/--end_idx/--resume/--prefetch_workers` |
| `pack_teacher_cache.py` | 小文件 cache → shard 转换 |
| `scripts/ensure_data.sh` | 检查 SSD 数据完整性 |

## 存储布局

- **NFS**（代码）：`/beegfs_hdd/data/nfs_share/users/lifanhong/nishome/video/`
- **SSD**（数据）：各机器 `/nvmessd/lifanhong/video/`
  - `llava-video/0_30_s_academic_v0_1/`：12139 个 mp4
  - `llava-video/30_60_s_academic_v0_1/`：10503 个 mp4
  - `parsed/visual_qa_v3_0_60_{train,val,test}.jsonl`
  - `teacher_cache_110k_sharded_256/`：正在提取的 110K cache
- **模型**：`/nvmessd/lifanhong/.cache/modelscope/Qwen/Qwen2___5-VL-7B-Instruct`

## 监控方法

检查三台状态的命令：

```bash
# 检查 session 是否活着（最重要，先查这个）
tmux list-sessions | grep extract  # gpu8 本机
ssh jiagpu4 'tmux list-sessions | grep extract'
ssh jiagpu5 'tmux list-sessions | grep extract'

# 检查进度和内存
grep -oP '\d+/\d+' /nvmessd/lifanhong/video/log_extract_teacher_110k.txt | tail -1
free -h | grep Mem | awk '{print "avail="$7}'
ls /nvmessd/lifanhong/video/teacher_cache_110k_sharded_256/teacher_shard_*.pt | wc -l
```

## 今天犯的错误（供参考，避免重蹈）

1. 评测时测试集不固定——视频目录变了导致 resolve 的 200 条不一样
2. prefetch_workers OOM 三次——每次自作主张改参数重启没问用户
3. 两台分工用 skip_samples 但没考虑 DDP 交错分配导致重叠
4. 给了乐观的时间估计（说 3.8h 实际要 10h）
5. gpu8 死了半小时没发现——只看进度数字没看 session 是否活着
6. 多次不等用户确认就动手改代码/重启任务
