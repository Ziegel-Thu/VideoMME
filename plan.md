# Video 项目迁移到 A100 集群计划

## 当前状态

- **分支**: `a40-cluster`（A40 集群开发）和 `master`（A100 amlt 集群）
- **环境**: 本地 dev VM 已配好 conda `video` + amlt + az cli
- **amlt project**: `bottleneck`（shuwangmain/amulet）
- **正在跑**: `pro-molly` — 110K teacher cache 提取（4×4 A100，prefetch=1，resume）

## 阶段 1：Teacher Cache 提取 ← 当前

- [x] 配好 amlt 环境（storage account、project checkout、target cache）
- [x] 写 `amlt_extract_teacher.yaml`（4 job 并行，start_idx/end_idx 分片）
- [x] 第一次提交（causal-dragon，prefetch=4）→ OOM SIGKILL
- [x] 第二次提交（pro-molly，prefetch=1，resume）
- [ ] 等 pro-molly 4 个 job 全部完成
- [ ] 验证 blob 上 shard 完整性（总样本数 ~103K，对比 A40 上的 102952）

## 阶段 2：Compressor 训练

teacher cache 就绪后，在 A100 集群上训练 cross-attention compressor。

### 2a. B-1L baseline（复现 A40 结果）
- 写 `amlt_train_compressor.yaml`
- 配置：K_seg=8, n_layers=1, loss_type=B, lr=1e-4, epochs=3
- SKU: 80G4-A100-NvLink
- 数据：`/mnt/default/bottleneck/teacher_cache_110k/`
- 需要注意：`--cache_shard_size 512`（避免初始化扫全部 shard 爆内存）
- 目标：复现 A40 上 B-1L epoch1 70.08% (0-30) / 68.48% (0-60)

### 2b. B-2L
- n_layers=2，其余同上
- A40 上 epoch1 loss=1.197 vs B-1L 1.214，精度待评

### 2c. 110K Inter-Segment Attention（新实验）
- 10K 上 inter-seg 没带来收益（66.5% vs 71.0%）
- 但 110K 数据量更大，值得重新验证
- 配置：n_layers=1, inter_layers=1, loss_type=B

### 2d. 110K BD loss ablation
- BD-1L: n_layers=1, loss_type=BD, K_seg=8, lr=1e-4, epochs=3
- 10K 上 BD=69.5% vs B=71.0%，110K 上验证差距是否缩小
- amlt 4×A100 standard

### 2e. K-sweep（下一步重点）
- 固定 loss_type=B, n_layers=2 (当前最佳配置)
- K_seg = 2 / 4 / 8 / 16 / 32
- 110K 数据，amlt 4×A100 standard，epochs=3
- 目的：找到压缩率和精度的最佳平衡点
- 当前 K=8 下 B-2L best ~69% (full eval)，K 增大理论上精度提升但压缩比降低

### 2f. 更多 ablation（后续）
- n_layers sweep: 1/2/3（K 固定为 sweep 最佳值）

### 2g. 架构探索（待讨论后确定）

**Pooling baseline（可直接跑）**：
- 用 weighted average pooling 替代 cross-attention，作为下界 baseline
- 看 cross-attention 到底贡献了多少
- 新建实验跑

**残差压缩（可直接跑）**：
- `compressed = compressor(dense) + pool(dense)`
- cross-attention 只学 residual，pool 提供 baseline representation
- 降低学习难度

**Question-conditioned compression（待讨论）**：
- 当前 compressor 是 question-agnostic，压缩不针对问题
- 方式 A：Q 作为额外 query 拼入 cross-attention
- 方式 B：两层 cross-attention，先 Q→queries 再 queries→vision
- 方式 C：queries = learnable + pool(Q_embed)，加法注入
- Trade-off：不能复用 cache，但蒸馏训练阶段无额外代价
- 倾向先试 C（最简单），再试 B（更有表达力）

**Gated compression**：
- compressor 输出后加 gate：gate = sigmoid(linear(compressed))
- 让模型学会对每个 compressed token 做重要性加权
- 类似 MoE router 思路，不同 token 负责不同类型视觉信息
- 可以和 cross-attention / pooling 组合使用

## 阶段 3：评测

### 3a. MCQ 评测
- 用 `eval_compressor.py` 在 test set 上评
- 对比 zero-shot baseline（logit=82%）
- 0-30s 和 0-60s 分别评

### 3b. Sanity Check
- 确认 compressor 输出的 compressed tokens 确实承载了视觉信息
- zero replacement 检验

## 阶段 4：代码整理与合并

### 需要解决的问题
1. **路径硬编码**：a40-cluster 分支有大量 `/nvmessd/lifanhong/...` 和 `/beegfs_hdd/...` 路径
   - amlt 集群统一用 `/mnt/default/bottleneck/...`
   - 需要确保代码里都用参数传入，不要硬编码
2. **模型路径**：A40 用 modelscope 缓存，amlt 用 HuggingFace `from_pretrained`
   - `extract_teacher.py` 默认 `--model_path Qwen/Qwen2.5-VL-7B-Instruct` 已兼容
   - `train_compressor.py` 和 `eval_compressor.py` 同理
3. **NFS 校验**：`validate_cache_dir` 拒绝 `/beegfs_hdd/` 前缀
   - amlt 上不存在这个问题，但逻辑保留无害
4. **README 更新**：
   - 补充 amlt 集群实验记录
   - 更新数据路径说明

### 合并策略（暂不执行）
- a40-cluster 的核心代码改动（compressor.py、train_compressor.py、eval_compressor.py 等）需要合入 master
- amlt yaml 配置文件留在 master
- A40 专属的日志/路径记录可以不合
- 建议：cherry-pick 代码文件，不合 README 的 A40 日志部分

## 风险与注意事项

1. **Blob 写入并发**：4 个 job 同时写同一个 output_dir，shard 文件名带 rank 前缀不会冲突，但不同 job 的 rank 编号相同（都是 0-3）→ **可能文件名冲突**
   - 实际：不同 job 处理不同 start_idx/end_idx 范围，ShardWriter 的 shard_idx 从现有最大编号+1 开始（`_next_shard_idx`），所以 resume 时不会覆盖
   - 但首次写入时，两个 job 的 rank0 都可能写 `teacher_shard_rank0_000.pt` → 需要验证
2. **Shard 索引一致性**：不同 job 写的 shard 里 global_idx 范围不同，训练时 TeacherCacheDataset 只按 shard 文件名排序加载，不关心 global_idx → 无问题
3. **模型下载时间**：每个 job 启动时都要从 HuggingFace 下载 ~15GB 模型，可能耗时 5-10 分钟
   - 后续可考虑先上传模型到 blob，避免重复下载
