"""
MCQ 评测：用 logits 判断选项准确率 + NLL loss

方法：对含答案的完整输入做一次 forward，
  - loss 直接取 model output（需要手动构造 labels）
  - accuracy 看答案首 token 位置的 logit argmax 是否 == GT token

对每个 checkpoint 评测 MCQ accuracy 和 avg loss。
"""
import os, sys, json, glob, torch, argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mvp import setup_model_and_tokenizer, build_bottleneck_mask, build_training_input_video


def make_labels(input_ids, tokenizer):
    """构造 labels：只在 assistant 回复部分计算 loss（和训练一致）"""
    labels = input_ids.clone()
    # 找 assistant 回复的起始 token（<|im_start|>assistant\n 之后）
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    ids = input_ids[0].tolist()

    # 找最后一个 <|im_start|>（对应 assistant turn）
    last_start = -1
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_start_id:
            last_start = i
            break

    if last_start >= 0:
        # mask 掉 <|im_start|> + "assistant" + "\n"（共 3 个 token）之前的所有内容
        labels[0, :last_start + 3] = -100

    pad_id = tokenizer.pad_token_id or 0
    labels[0, input_ids[0] == pad_id] = -100
    return labels


def find_answer_start(input_ids, tokenizer):
    """找到 assistant 回复的首个实质 token 位置（跳过 <|im_start|>assistant\\n）"""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    ids = input_ids[0].tolist()
    last_start = -1
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_start_id:
            last_start = i
            break
    # answer 从 last_start + 3 开始（<|im_start|> assistant \n）
    return last_start + 3 if last_start >= 0 else -1


def eval_checkpoint(model, processor, tokenizer, latent_tokens, latent_token_ids,
                    ckpt_path, data, device, use_bottleneck=True):
    """评测一个 checkpoint"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    embed = model.get_input_embeddings()
    for tid, emb in ckpt["latent_embeddings"].items():
        embed.weight.data[tid] = emb.to(embed.weight.device)
    model.eval()

    lang = model.base_model.model.model.language_model
    correct = 0
    total = 0
    total_loss = 0
    errors = 0

    for sample in tqdm(data, desc=os.path.basename(ckpt_path)):
        try:
            inputs = build_training_input_video(
                processor, tokenizer, latent_tokens,
                sample["video_path"], sample["question"], sample["answer"],
                num_frames=4,
            )
            inputs.pop("_frame_timestamps", None)
            inputs.pop("_video_duration", None)

            # 构造 labels
            labels = make_labels(inputs["input_ids"], tokenizer)
            inputs["labels"] = labels

            inputs_dev = {k: v.to(device) for k, v in inputs.items()
                         if isinstance(v, torch.Tensor)}

            # Bottleneck hooks
            hooks = []
            if use_bottleneck:
                bn_mask = build_bottleneck_mask(
                    inputs["input_ids"], latent_token_ids, enable_bottleneck=True
                ).to(device, dtype=torch.bfloat16)

                def lh(m):
                    def fn(module, args, kwargs):
                        kwargs["attention_mask"] = m
                        return args, kwargs
                    return fn

                for layer in lang.layers:
                    hooks.append(layer.register_forward_pre_hook(lh(bn_mask), with_kwargs=True))

            with torch.no_grad():
                out = model(**inputs_dev)
                if out.loss is not None:
                    total_loss += out.loss.item()

            for h in hooks:
                h.remove()

            # 判断准确率：答案首 token 位置的 logit argmax
            ans_pos = find_answer_start(inputs["input_ids"], tokenizer)
            if ans_pos < 0 or ans_pos >= inputs["input_ids"].shape[1]:
                errors += 1
                continue

            # logits 形状 [1, seq_len, vocab]，位置 ans_pos 的输入预测 ans_pos+1
            # 但 causal LM 里 logits[pos] 预测的是 pos+1 的 token
            # 所以 logits[ans_pos - 1] 预测 ans_pos 位置的 token
            pred_logits = out.logits[0, ans_pos - 1, :]
            pred_token = pred_logits.argmax().item()
            gt_token = inputs["input_ids"][0, ans_pos].item()

            if pred_token == gt_token:
                correct += 1
            total += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [错误] {e}")
            continue

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    if errors > 0:
        print(f"  跳过 {errors} 条出错样本")
    return acc, avg_loss, total


def main(args):
    device = torch.device("cuda")
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(K=args.K)
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    # 加载测试数据
    video_index = {}
    for vdir in ["/home/v-shuzheng/video/data/open-o3-video/videos/stgr",
                 "/home/v-shuzheng/video/data/llava-video"]:
        for mp4 in glob.glob(os.path.join(vdir, "**/*.mp4"), recursive=True):
            video_index[os.path.basename(mp4)] = mp4

    test_data = []
    with open(args.data_path) as f:
        all_data = [json.loads(l) for l in f]

    # 取最后 10% 作为测试集（和训练时一样的 split）
    n_test = min(500, len(all_data) // 10)
    test_samples = all_data[-n_test:]
    for d in test_samples:
        vname = os.path.basename(d.get("video_path", ""))
        if vname in video_index:
            d["video_path"] = video_index[vname]
            test_data.append(d)

    if args.max_samples:
        test_data = test_data[:args.max_samples]
    print(f"测试集: {len(test_data)} 条（来源: 最后 {n_test} 条中有视频的）")

    # 评测每个 checkpoint
    results = []
    for ckpt_path in args.checkpoints:
        print(f"\n{'='*50}")
        print(f"Checkpoint: {os.path.basename(ckpt_path)}")
        print(f"{'='*50}")
        acc, loss, n = eval_checkpoint(
            model, processor, tokenizer, latent_tokens, latent_token_ids,
            ckpt_path, test_data, device, use_bottleneck=args.bottleneck,
        )
        print(f"  Accuracy: {acc:.4f} ({int(acc * n)}/{n})")
        print(f"  Avg Loss: {loss:.4f}")
        results.append({
            "checkpoint": os.path.basename(ckpt_path),
            "accuracy": acc, "loss": loss, "n": n,
        })

    print(f"\n{'='*50}")
    print("汇总")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['checkpoint']:30s} acc={r['accuracy']:.4f}  loss={r['loss']:.4f}  n={r['n']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data_path",
                        default="/home/v-shuzheng/video/data/parsed/visual_qa_v2.jsonl")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--bottleneck", action="store_true")
    args = parser.parse_args()
    main(args)
