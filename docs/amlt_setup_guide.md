# amlt 环境配置指南

> 适用于接手 bottleneck 项目的新环境

## 1. Conda 环境

```bash
conda create -n video python=3.10 -y
conda activate video
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.6 peft accelerate qwen-vl-utils decord pillow pandas pyarrow
```

## 2. az cli

```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login --use-device-code
```

## 3. amlt

```bash
pip install amlt --extra-index-url https://msrpypi.azurewebsites.net/stable/learnedtools
```

## 4. amlt 存储凭证

```bash
amlt cred storage set shuwangmain --subscription "762905fc-41fb-4bfb-8e41-478b86cb99ab" --resource-group "system_yeyun"
```

## 5. amlt project checkout

```bash
cd /path/to/video
amlt project checkout bottleneck shuwangmain
```

验证：
```bash
amlt project
# 应显示:
# PROJECT_NAME            bottleneck
# STORAGE_ACCOUNT         shuwangmain
# STORAGE_CONTAINER_NAME  amulet
```

## 6. SSH key

```bash
ssh-keygen -t ed25519 -C "your-email"
# amlt 会自动上传公钥到 job
```

## 7. Git

```bash
git clone git@github.com:Ziegel-Thu/VideoMME.git video
cd video
git config user.name "Your Name"
git config user.email "your-email"
```

## 8. 常用命令

```bash
amlt status <experiment>          # 查看状态
amlt logs view <experiment>       # 查看日志
amlt results download <exp> -o /path/  # 下载结果
amlt run path/to/yaml -d "描述"   # 提交 job
amlt cancel -y <experiment>       # 取消
amlt target info sing -v          # 集群资源
amlt list -n 20                   # 最近实验
```

## 9. 集群信息

| 项目 | 值 |
|------|---|
| 集群 | msrresrchbasicvc |
| workspace | msraairwsws |
| SKU (训练) | 80G4-A100-NvLink (4×A100 NvLink) |
| SKU (eval) | 80G4-A100 (4×A100, 含 NDAMv4) |
| standard | 不被抢占，但队列满时排队久 |
| basic | 可能被抢占，但空闲卡多 |

## 10. Blob 存储布局

```
shuwangmain/amulet/bottleneck/          # amlt 项目存储
├── checkpoints/                         # 所有训练 checkpoint
│   ├── 006_compressor/110k_B{1,2}L_ep{1,2,3}.pt
│   ├── 007_interseg/110k_interseg_B1L_ep{1,2,3}.pt
│   ├── 008_ksweep/110k_K{2,4,8,16}_B2L_ep{1,2,3}.pt
│   ├── 009_pooling/110k_pool_B1L_ep{1,2,3}.pt
│   └── 010_gated/110k_gated_B2L_ep{1,2,3}.pt
├── data/
│   ├── parsed/visual_qa_v2.jsonl        # MCQ 数据
│   ├── parsed/temporal_evidence.jsonl   # temporal 标注
│   ├── videos/                          # LLaVA-Video 0-60s
│   └── stgr_videos/                     # temporal grounding 视频
├── models/Qwen2.5-VL-7B-Instruct/      # 冻结 LLM
├── benchmarks/
│   ├── mvbench/                         # MVBench (videos + json)
│   ├── videomme/                        # Video-MME Short (videos + parquet)
│   └── nextqa/                          # NExT-QA (videos + csv)
└── teacher_cache_110k_v2/               # 预提取 teacher 特征 (216 shards)

shuwangmain/zhengshurui/video_project/   # 个人备份 (同上结构)
```

## 11. 收集正在跑的 eval 结果

sandbox 到期后集群 job 继续跑。收集方法：

```bash
# 查看哪些 pass 了
amlt list -n 50 | grep Pass

# 下载结果
amlt results download <experiment-name> -o /tmp/results/

# 解析结果
python3 -c "
import re, glob
for f in sorted(glob.glob('/tmp/results/*/*/shard*.txt')):
    for line in open(f):
        if '总准确率' in line:
            print(f'{f.split(\"/\")[-2]}: {line.strip()}')
"
```

## 12. 当前正在跑的 Experiment

### 训练 (STD)
| Experiment | 内容 |
|------------|------|
| happy-malamute | K=32 B2L 110K 3ep |
| peaceful-sturgeon | 007 inter-seg B2L 110K 3ep |

### MCQ 全量 eval (BSC, 各实验 eval_compressor.py)
| Experiment | 内容 |
|------------|------|
| square-cod | 010 MCQ ep1+ep2 |
| eager-kit | 009 MCQ ep1+ep2+ep3 |
| nice-ewe | K2 MCQ ep1 |
| wondrous-moose | K4 MCQ ep1 |
| tender-wallaby | K2 MCQ ep3 |
| talented-chigger | K4 MCQ ep3 |
| tender-marmoset | K8 MCQ ep3 |
| driving-python | K16 MCQ ep3 |
| present-caiman | 010 MCQ ep3 |

### MVBench eval (BSC/STD, 新代码 sdpa)
| Experiment | 内容 |
|------------|------|
| huge-snail | 007+K-sweep+010 ep1 (6 job) |
| included-sheep | ep2/ep3 (7 job) |
| optimum-wren | K2/K4/K8 ep3 (3 job) |
| living-wildcat | 009+010 ep1 (STD, 2 job) |

### VME Short eval
| Experiment | 内容 | 代码版本 |
|------------|------|---------|
| adjusted-tadpole | 006 B1L ep2/3 + B2L ep1/3 | 旧代码 sdpa |
| actual-sunfish | 009 ep1/2/3 | 新代码 sdpa |
| sterling-moray | 010 ep1 | 新代码 sdpa |

### NExT-QA eval (BSC, 新代码 sdpa)
| Experiment | 内容 |
|------------|------|
| crack-macaw | 007+K-sweep+010 ep1 (6 job) |
| moral-buzzard | zeroshot+006 B2L (2 job) |
| grand-marten | 009 ep1/2/3 (3 job) |

## 13. ⚠️ 注意事项

- **attn_implementation**: compressor eval 必须用 sdpa。eager 和 sdpa logits 差 ~1.0 (bfloat16 累积误差)，MCQ 准确率差 ~4%，不可比
- **eval 代码版本**:
  - `eval_sdpa_verified/`: 旧代码，只支持 006/007/K-sweep VME Short
  - `013/` 新代码: 支持全部 checkpoint × 全部 benchmark，需要 compressor.py 包含 GatedVoCoCompressor + PoolingCompressor
  - 各实验 `eval_compressor.py`: MCQ 专用，最可信，从未改过
- **model.py**: `get_video_features` 兼容 tuple 和 BaseModelOutputWithPooling 返回值
- **011/012 temporal**: 训练失败 (loss=0)，暂不处理
