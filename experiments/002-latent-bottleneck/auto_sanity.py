"""训练完成后自动 sanity check"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mvp import setup_model_and_tokenizer, build_bottleneck_mask
from train_mvp import ImageQADataset, collate_fn_factory
from torch.utils.data import DataLoader

def run_sanity(checkpoint_path, max_samples=30):
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(K=8)
    latent_tokens = [f"<latent_{i}>" for i in range(8)]
    device = next(model.parameters()).device

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    embed = model.get_input_embeddings()
    for tid, emb in ckpt["latent_embeddings"].items():
        embed.weight.data[tid] = emb.to(embed.weight.device)
    model.eval()

    dataset = ImageQADataset(
        os.path.join(os.path.dirname(__file__), "../../data/parsed/visual_qa.jsonl"),
        processor, tokenizer, latent_tokens, max_samples=max_samples,
    )
    collate = collate_fn_factory(processor, tokenizer, latent_tokens)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)

    lang = model.base_model.model.model.language_model
    results = {"normal": [], "zero_no_bn": [], "bn_normal": [], "bn_zero": []}

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        ids = batch["input_ids"][0]
        vmask = (ids >= 151652) & (ids <= 151656)
        bn = build_bottleneck_mask(batch["input_ids"], latent_token_ids,
                                    enable_bottleneck=True).to(device, dtype=torch.bfloat16)

        def zh(module, args, kwargs):
            if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                e = kwargs["inputs_embeds"].clone()
                e[0, vmask] = 0.0
                kwargs["inputs_embeds"] = e
            return args, kwargs

        def lh(m):
            def fn(module, args, kwargs):
                kwargs["attention_mask"] = m
                return args, kwargs
            return fn

        with torch.no_grad():
            results["normal"].append(model(**batch).loss.item())

        h = lang.register_forward_pre_hook(zh, with_kwargs=True)
        with torch.no_grad():
            results["zero_no_bn"].append(model(**batch).loss.item())
        h.remove()

        hooks = [layer.register_forward_pre_hook(lh(bn), with_kwargs=True) for layer in lang.layers]
        with torch.no_grad():
            results["bn_normal"].append(model(**batch).loss.item())
        for h in hooks:
            h.remove()

        hz = lang.register_forward_pre_hook(zh, with_kwargs=True)
        hooks = [layer.register_forward_pre_hook(lh(bn), with_kwargs=True) for layer in lang.layers]
        with torch.no_grad():
            results["bn_zero"].append(model(**batch).loss.item())
        hz.remove()
        for h in hooks:
            h.remove()

    print("=" * 50)
    print("SANITY CHECK RESULTS")
    print("=" * 50)
    for c, l in results.items():
        print(f"  {c:15s}: {sum(l)/len(l):.4f}")

    zd = sum(results["zero_no_bn"])/len(results["zero_no_bn"]) - sum(results["normal"])/len(results["normal"])
    bd = sum(results["bn_zero"])/len(results["bn_zero"]) - sum(results["bn_normal"])/len(results["bn_normal"])
    print(f"\n  Zero Δ (no BN): {zd:+.4f}  {'✅ PASS' if zd > 0.1 else '❌ FAIL'}")
    print(f"  Zero Δ (BN):    {bd:+.4f}  {'✅ PASS' if bd > 0.1 else '❌ FAIL'}")
    return zd > 0.1 and bd > 0.1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_samples", type=int, default=30)
    args = parser.parse_args()
    passed = run_sanity(args.checkpoint, args.max_samples)
    sys.exit(0 if passed else 1)
