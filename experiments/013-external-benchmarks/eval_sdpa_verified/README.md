# eval_sdpa_verified — 旧版 eval 代码（sdpa，集群验证通过）

## 来源
从 commit fc2347f（2026-05-20）提取，当时目录名为 014-external-benchmarks。

## 集群验证记录
- **inspired-bluejay**: VME Short 007+K2/K4/K8/K16 ep1，5 job 全部 pass
  - 007=40.33%, K2=40.78%, K4=42.22%, K8=41.78%, K16=42.22%
- **sharing-tahr**: VME Short 006 B1L+B2L，2 job 全部 pass
  - B1L=42.89%, B2L=43.00%

## 支持范围
- ✅ 006 标准 VoCoCompressor
- ✅ 007 inter-seg (VoCoCompressor + InterSegmentAttention)
- ✅ 008 K-sweep (VoCoCompressor, K=2/4/8/16/32)
- ❌ 009 pooling (PoolingCompressor) — 不支持
- ❌ 010 gated (GatedVoCoCompressor) — 不支持

## 已知问题
- model.py 的 get_video_features 使用 `torch.cat(embeds_list)`，
  在 eager 模式下会返回 tuple 导致 crash。
  但 sdpa 模式下返回 object，不会触发此 bug。
- eval_mvbench.py 在 sdpa 下 007/K-sweep compressor 模式会 0/0（原因未查明）
- eval_videomme_short.py 在 sdpa 下正常工作

## 用法
amlt yaml 中 `code.local_dir` 指向本目录。
