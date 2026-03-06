import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping
from llmcompressor.utils import dispatch_for_generation

# llmcompressor == 0.9.0
MODEL_ID = "/data/Models/Qwen3-VL-8B-Instruct"

# 1. 显式定义映射：解决 Qwen3VL 复杂的层级命名问题
# 注意：我们将 qkv_proj 拆开，因为报错显示模型中它们是独立的
custom_mappings = [
    # 路径 A: 输入层归一化 -> Attention 投影
    AWQMapping(
        smooth_layer="re:.*\.input_layernorm$", 
        balance_layers=[
            "re:.*\.self_attn\.q_proj$", 
            "re:.*\.self_attn\.k_proj$", 
            "re:.*\.self_attn\.v_proj$"
        ]
    ),
    # 路径 B: 注意力后归一化 -> MLP 门控与上升投影
    AWQMapping(
        smooth_layer="re:.*\.post_attention_layernorm$", 
        balance_layers=[
            "re:.*\.mlp\.gate_proj$", 
            "re:.*\.mlp\.up_proj$"
        ]
    ),
    # 路径 C: MLP 上升投影 -> MLP 下降投影 (AWQ 常用链式平滑)
    AWQMapping(
        smooth_layer="re:.*\.mlp\.up_proj$", 
        balance_layers=["re:.*\.mlp\.down_proj$"]
    )
]

# Load model.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 2048  # 建议先调小以确保 4080 显存不溢出

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess_function(example):
    messages = []
    for message in example["messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )

ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)

def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }

# 2. 配置 AWQ 量化配方
recipe = AWQModifier(
    mappings=custom_mappings,
    ignore=[
        "re:.*embed_tokens",
        "re:model\.visual.*",  # 视觉模块不参与 AWQ
        "re:visual.*",
        "lm_head",
    ],
    duo_scaling=True,
    config_groups={
        "group_0": {
            # 明确指定所有线性层目标
            "targets": [
                "re:.*\.q_proj$", "re:.*\.k_proj$", "re:.*\.v_proj$", "re:.*\.o_proj$",
                "re:.*\.gate_proj$", "re:.*\.up_proj$", "re:.*\.down_proj$"
            ],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "group_size": 32,
                "strategy": "group",
                "dynamic": False,
                "actorder": None,
                "observer": "mse",
            },
        }
    },
)

# Apply AWQ quantization.
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
# 确保在推理时也将数据移动到 GPU
inputs = processor(text="Hello my name is", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-AWQ-W4A16-mse-seq"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)