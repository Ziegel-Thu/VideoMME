# 004-voco-segment-compression: VoCo 风格分段压缩

## 概述

完全照搬 VoCo-LLaMA 的设计：
- 视频按 2-4s 切分成段
- 每段独立 forward：`[vision_t, voco_t×K_seg]`，attention mask 让 vision 后的 text 不能跨过 voco 看 vision
- 拼接所有段的 voco activations + question + answer
- **标准 SFT NLL loss**（Cross Entropy），不用 KL 蒸馏

## 设计

### Token 顺序（VoCo 风格 - 方案 B）

```
[vision_seg1] [voco_1×K] [vision_seg2] [voco_2×K] ... [vision_segN] [voco_N×K] [Q] [A]
                                                                                ↑
                                                                       这里 Q/A 之前所有 vision
                                                                       通过 voco 屏障被压缩
```

**Q 在所有 voco 之后** → latent (voco) 是 question-agnostic，可以缓存复用。

### Attention Mask

对每段 voco 起到"屏障"作用：
- voco 之前的 vision tokens 看不到 voco 之后的内容
- voco 之后的 token（其他段 vision、Q、A）不能直接看 voco 之前的 vision，只能通过 voco 看

### Training Loss

标准 NLL on answer tokens：
```python
loss = CrossEntropy(logits[answer_positions], answer_token_ids)
```

不需要 teacher-student KL。

### 关键参数

- 段长度: 2-4 秒
- 每段 voco 数: K_seg = 4-8
- 视频帧率: 1fps
- 30s 视频 → 8-15 段 → ~32-120 voco tokens

## 与 003 (bottleneck SFT) 的关系

003 是单段 + question 在 latent 之前的版本。004 是分段 + question 在 latent 之后的 VoCo 风格。

| 项目 | 003 | 004 |
|------|-----|-----|
| 分段 | ❌ 整体一段 | ✅ 2-4s 一段 |
| Q 位置 | 在 latent 之前 (conditioned) | 在 latent 之后 (agnostic) |
| Loss | NLL on answer | NLL on answer |
| 适合长视频 | ❌（序列 OOM） | ✅（每段独立 forward） |

## 实现要点

1. **Attention mask 实现**: 参考 VoCo-LLaMA `make_voco_mask_llava`
2. **分段 forward + KV cache 拼接**: 每段独立处理后 concatenate
3. **Q 位置**: 把 Q 拼到所有 voco caches 之后
4. **A 位置**: 标准最后位置
5. **Temporal Head (Stage 3)**: question-aware，从 voco hidden states + Q 解码时间段

## 文件结构

```
004-voco-segment-compression/
├── README.md
├── voco_mask.py        # VoCo-style attention mask
├── model.py            # 模型加载 + voco token 注入
├── data.py             # Dataset + 分段 collate
├── train.py            # 单 forward 版（mask 隔离段间 vision）
├── train_concat.py     # 拼接版（KV cache concat，省计算量）
└── amlt.yaml
```

## 实验记录

### Pilot 结果

| 实验 | 版本 | 数据 | 结果 |
|------|------|------|------|
| 本地 5 条 | 单forward | 5×1ep | ✅ loss 4.03 |
| 本地 10 条 | 拼接版 | 10×3ep | ✅ loss 3.83→1.11→0.42 |
| voco-single-200 (集群) | 单forward 4卡 | 200×3ep | ✅ val_loss 0.30→0.25→0.21 |
| voco-concat-v3 (集群) | 拼接版 4卡 | 200×3ep | ❌ ep1 93% crash (DDP bug) |
| **voco-single-10k** (集群) | **单forward 4卡** | **10K×5ep** | **running, ep3 25%** |

### 技术发现

1. **VoCo 原论文用 NLL loss**（不是 KL 蒸馏），是标准 SFT + VoCo mask
2. **单 forward + mask**: 稳定可靠，可用 gradient checkpointing
3. **拼接版**: 计算量小但不能用 gradient checkpointing (和 use_cache 冲突)
4. **KV cache 保留计算图**: 梯度可反传到 voco_embeds

## 已完成

- [x] VoCo 风格 attention mask（向量化实现）
- [x] 单 forward 版 + DDP
- [x] 拼接版 + detach vision 省显存
- [x] Pilot 验证 loss 收敛
- [x] 10K 全量训练提交

## 待办

- [ ] 修复拼接版 DDP crash bug
- [ ] 评测 vs 003 (单段 bottleneck SFT)
- [ ] 预提特征 + 短序列训练（长视频方案）
- [ ] Stage 3: question-aware temporal head
