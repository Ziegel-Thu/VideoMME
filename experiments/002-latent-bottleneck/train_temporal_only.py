"""
Stage 3: 从 bottleneck checkpoint 加载，冻 LoRA，只训 temporal head
用 temporal_evidence 数据（有时间标注）
"""
import os, sys, json, glob, torch, torch.nn.functional as F, argparse
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mvp import setup_model_and_tokenizer, build_bottleneck_mask, build_training_input_video
from train_video import TemporalHead

class TemporalDataset(Dataset):
    def __init__(self, data_path, video_dirs, num_frames=4, max_samples=None):
        video_index = {}
        for vdir in video_dirs:
            for mp4 in glob.glob(os.path.join(vdir, "**/*.mp4"), recursive=True):
                video_index[os.path.basename(mp4)] = mp4

        self.samples = []
        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                vname = os.path.basename(d["video_path"])
                if vname in video_index and d.get("evidence_segments"):
                    d["video_path"] = video_index[vname]
                    self.samples.append(d)
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.num_frames = num_frames
        print(f"TemporalDataset: {len(self.samples)} samples (all have temporal labels)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_temporal(batch, processor, tokenizer, latent_tokens, num_frames, latent_token_ids):
    from qwen_vl_utils import process_vision_info
    import decord
    from PIL import Image
    import tempfile

    texts, all_vid, all_img = [], [], []
    temporal_labels = []
    latent_str = "".join(latent_tokens)

    for item in batch:
        try:
            vr = decord.VideoReader(item["video_path"])
            total = len(vr)
            fps = vr.get_avg_fps()
            duration = total / fps
            indices = [int(i * total / num_frames) for i in range(num_frames)]
            frames = [vr[idx].asnumpy() for idx in indices]
            frame_times = [idx / fps for idx in indices]
        except:
            frames = [Image.new("RGB", (224,224), "black")] * num_frames
            duration, frame_times = 30.0, [i*4.0 for i in range(num_frames)]

        tmp_paths = []
        for i, f in enumerate(frames):
            if not isinstance(f, Image.Image):
                f = Image.fromarray(f)
            p = os.path.join(tempfile.gettempdir(), f"_tf_{os.getpid()}_{i}.jpg")
            f.save(p)
            tmp_paths.append(p)

        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": tmp_paths, "fps": 1.0},
                {"type": "text", "text": f"{item['question']}\n{latent_str}"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": item["answer"]},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        img_in, vid_in = process_vision_info(messages)
        all_img.extend(img_in or [])
        all_vid.extend(vid_in or [])
        for p in tmp_paths:
            os.remove(p)

        # Temporal labels
        t_labels = []
        for ft in frame_times:
            is_ev = any(ts <= ft <= te for ts, te in item["evidence_segments"])
            t_labels.append(1.0 if is_ev else 0.0)
        temporal_labels.append(t_labels)

    inputs = processor(text=texts, images=all_img or None, videos=all_vid or None,
                       return_tensors="pt", padding=True)
    # Labels for L_ans
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    for b in range(input_ids.shape[0]):
        ids = input_ids[b].tolist()
        last_start = max(i for i, t in enumerate(ids) if t == im_start_id)
        labels[b, :last_start+3] = -100
        pad_id = tokenizer.pad_token_id or 0
        labels[b, input_ids[b] == pad_id] = -100
    inputs["labels"] = labels
    inputs["_temporal_labels"] = temporal_labels
    return inputs

def train(args):
    device = torch.device("cuda")

    # 加载模型 + bottleneck checkpoint
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(K=args.K)
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    ckpt = torch.load(args.bottleneck_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    embed = model.get_input_embeddings()
    for tid, emb in ckpt["latent_embeddings"].items():
        embed.weight.data[tid] = emb.to(embed.weight.device)
    print("Bottleneck checkpoint 加载完成")

    # 冻 LoRA + latent embedding，只训 temporal head
    for name, param in model.named_parameters():
        param.requires_grad = False
    print("LoRA + LLM 全部冻结")

    # Temporal Head
    temporal_head = TemporalHead(3584, num_bins=args.num_frames).to(device)
    print(f"Temporal Head 参数: {sum(p.numel() for p in temporal_head.parameters()):,}")

    # 数据
    dataset = TemporalDataset(
        args.data_path,
        ["/home/v-shuzheng/video/data/open-o3-video/videos/stgr",
         "/home/v-shuzheng/video/data/llava-video"],
        num_frames=args.num_frames, max_samples=args.max_samples,
    )
    def collate(batch):
        return collate_temporal(batch, processor, tokenizer, latent_tokens,
                                args.num_frames, latent_token_ids)

    n_val = min(500, len(dataset)//10)
    train_set, val_set = random_split(dataset, [len(dataset)-n_val, n_val],
                                       generator=torch.Generator().manual_seed(42))
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    optimizer = torch.optim.AdamW(temporal_head.parameters(), lr=args.lr)
    lang = model.base_model.model.model.language_model
    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        temporal_head.train()
        total_loss = 0
        n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            t_labels = batch.pop("_temporal_labels")
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Bottleneck hooks
            bn_mask = build_bottleneck_mask(
                batch["input_ids"], latent_token_ids, enable_bottleneck=True
            ).to(device, dtype=torch.bfloat16)
            def lh(m):
                def fn(module, args, kwargs):
                    kwargs["attention_mask"] = m; return args, kwargs
                return fn
            hooks = [layer.register_forward_pre_hook(lh(bn_mask), with_kwargs=True)
                     for layer in lang.layers]

            # 捕获最后一层 hidden state
            captured = {}
            def capture_hook(module, input, output):
                captured["hidden"] = output[0]
            hooks.append(lang.layers[-1].register_forward_hook(capture_hook))

            with torch.no_grad():
                model(**batch)

            for h in hooks:
                h.remove()

            if "hidden" not in captured:
                continue

            hidden = captured["hidden"]
            latent_set = set(latent_token_ids)
            loss = torch.tensor(0.0, device=device)
            cnt = 0

            for b in range(batch["input_ids"].shape[0]):
                if t_labels[b] is None:
                    continue
                ids = batch["input_ids"][b].tolist()
                lat_pos = [i for i, t in enumerate(ids) if t in latent_set]
                if not lat_pos:
                    continue
                lat_hidden = hidden[b, lat_pos, :].unsqueeze(0).float()
                logits = temporal_head(lat_hidden)
                target = torch.tensor([t_labels[b]], dtype=torch.float, device=device)
                loss = loss + F.binary_cross_entropy_with_logits(logits, target)
                cnt += 1

            if cnt > 0:
                loss = loss / cnt
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(n, 1)
        print(f"  Epoch {epoch+1}: train_loss={avg:.4f}")

        # Val
        temporal_head.eval()
        val_loss, vn = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                t_labels = batch.pop("_temporal_labels")
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                bn_mask = build_bottleneck_mask(
                    batch["input_ids"], latent_token_ids, enable_bottleneck=True
                ).to(device, dtype=torch.bfloat16)
                def lh(m):
                    def fn(module, args, kwargs):
                        kwargs["attention_mask"] = m; return args, kwargs
                    return fn
                hooks = [layer.register_forward_pre_hook(lh(bn_mask), with_kwargs=True)
                         for layer in lang.layers]
                captured = {}
                def capture_hook(module, input, output):
                    captured["hidden"] = output[0]
                hooks.append(lang.layers[-1].register_forward_hook(capture_hook))
                model(**batch)
                for h in hooks:
                    h.remove()
                if "hidden" not in captured:
                    continue
                hidden = captured["hidden"]
                for b in range(batch["input_ids"].shape[0]):
                    if t_labels[b] is None:
                        continue
                    ids = batch["input_ids"][b].tolist()
                    lat_pos = [i for i, t in enumerate(ids) if t in latent_set]
                    if not lat_pos:
                        continue
                    lat_hidden = hidden[b, lat_pos, :].unsqueeze(0).float()
                    logits = temporal_head(lat_hidden)
                    target = torch.tensor([t_labels[b]], dtype=torch.float, device=device)
                    val_loss += F.binary_cross_entropy_with_logits(logits, target).item()
                    vn += 1

        val_loss = val_loss / max(vn, 1)
        print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}")

        torch.save(temporal_head.state_dict(),
                   os.path.join(args.output_dir, f"temporal_head_ep{epoch+1}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(temporal_head.state_dict(),
                       os.path.join(args.output_dir, "temporal_head_best.pt"))
            print(f"  ★ New best (val_loss={val_loss:.4f})")

    print(f"\nDone. best_val={best_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bottleneck_ckpt", required=True)
    parser.add_argument("--data_path", default="/home/v-shuzheng/video/data/parsed/temporal_evidence.jsonl")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "outputs_temporal_only"))
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    train(args)
