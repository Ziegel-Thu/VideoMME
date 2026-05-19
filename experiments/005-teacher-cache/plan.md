# 005 Teacher Cache 计划

## 目标
预提取 110K 样本的 teacher 特征到 blob，供 006-012 所有实验使用。

## 已完成
- [x] extract_teacher.py + shard_prefix 修复
- [x] p0/p1/p4/p5 提取完成（frank-terrier）
- [x] p2/p3 第一轮部分完成（28 shard each）

## 进行中
- [ ] p2/p3 resume 补完（leading-jay）

## 待做
- [ ] 补完后验证总样本数（目标 ~103K）
- [ ] 记录最终 shard 统计到 README
