import torch
import torch.nn as nn
import gc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# 1. 获取 Transformer blocks
# -----------------------------
def get_blocks(model):
    name = model.__class__.__name__
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # LLaMA / Qwen
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # Bloom / Falcon / BigCode
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder.layers  # OPT
    raise NotImplementedError(f"Unsupported model type: {name}")


# -----------------------------
# 2. 在线统计器（不存 tensor）
# -----------------------------
class OnlineStats:
    def __init__(self):
        self.count = 0
        self.max = 0.0
        self.sum = 0.0
        self.sumsq = 0.0
        self.samples = []  # 只存少量用于 percentile（可控）

    def update(self, x: torch.Tensor):
        x = x.float().flatten()
        self.count += x.numel()
        self.max = max(self.max, x.abs().max().item())
        self.sum += x.abs().sum().item()
        self.sumsq += (x ** 2).sum().item()

        # 只采样一小部分用于 percentile
        if len(self.samples) < 200_000:
            idx = torch.randperm(x.numel())[: min(1024, x.numel())]
            self.samples.append(x[idx].cpu())

    def finalize(self):
        mean = self.sum / self.count
        std = (self.sumsq / self.count - mean**2) ** 0.5
        samples = torch.cat(self.samples) if self.samples else None
        p999 = torch.quantile(samples.abs(), 0.999).item() if samples is not None else 0
        outlier_ratio = (samples.abs() > p999).float().mean().item() if samples is not None else 0
        return {
            "max": self.max,
            "std": std,
            "max_div_std": self.max / (std + 1e-6),
            "p999": p999,
            "outlier_ratio": outlier_ratio,
        }


# -----------------------------
# 3. 主分析逻辑
# -----------------------------
@torch.no_grad()
def analyze_model(
    model_name,
    text="Hello, this is a calibration example.",
    n_tokens=512,
    device="cuda",
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens)
    inputs = inputs.to(device)

    layers = get_blocks(model)

    layer_stats = []

    # 先跑 embedding + layer0 输入
    hidden_states = model.model.embed_tokens(inputs["input_ids"])

    for i, layer in enumerate(layers):
        print(f"Analyzing layer {i} ...")
        layer = layer.to(device)

        stats = {}

        # 注册 hook（只统计，不存）
        handles = []
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                stats[name] = OnlineStats()

                def hook_fn(mod, inp, out, s=stats[name]):
                    s.update(inp[0])

                handles.append(m.register_forward_hook(hook_fn))

        # forward
        hidden_states = layer(hidden_states)[0]

        for h in handles:
            h.remove()

        # 权重统计
        weight_stats = {}
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                w = m.weight.detach().float()
                weight_stats[name] = {
                    "w_max": w.abs().max().item(),
                    "w_std": w.std().item(),
                }

        # 汇总
        layer_result = {
            "layer": i,
            "activation": {k: v.finalize() for k, v in stats.items()},
            "weight": weight_stats,
        }
        layer_stats.append(layer_result)

        layer = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    return layer_stats


# -----------------------------
# 4. 可视化
# -----------------------------
def plot_stats(layer_stats, key="max_div_std"):
    xs, ys = [], []
    for layer in layer_stats:
        for name, s in layer["activation"].items():
            xs.append(layer["layer"])
            ys.append(s[key])

    plt.figure(figsize=(10, 4))
    plt.scatter(xs, ys, s=10)
    plt.xlabel("Layer Index")
    plt.ylabel(key)
    plt.title(f"Activation statistic: {key}")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    stats = analyze_model(
        model_name="meta-llama/Llama-2-7b-hf",
        text="Hello world! This is a test input for calibration.",
    )

    plot_stats(stats, key="max_div_std")
    plot_stats(stats, key="outlier_ratio")
