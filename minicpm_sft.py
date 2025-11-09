# train_minicpm_v_sft_lora.py
import json, os
from dataclasses import dataclass
from typing import Dict, List

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image

from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ========= Config =========
MODEL_ID = "openbmb/MiniCPM-V-2_6"       # 或 openbmb/MiniCPM-V-4_5 / -4 视显存而定
DATA_JSONL = "minicpm_sft_llava150k.jsonl"
OUTPUT_DIR = "minicpm_v_lora_sft"
BATCH_SIZE = 2          # QLoRA 下单卡 24GB 建议 2~4；多卡可调大
GRAD_ACC = 16
LR = 2e-4
EPOCHS = 1              # 先跑通一轮；实战 1~3
MAX_LEN = 2048
WARMUP = 0.03
USE_QLORA = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= Load model & processor =========
bnb_config = None
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# MiniCPM-V 的 AutoProcessor 里已经包含图像变换与文本模板
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32,
    quantization_config=bnb_config,
    trust_remote_code=True
)

if USE_QLORA:
    model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"], # 视觉/文本注意力投影名称，具体以权重名为准
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ========= Dataset =========
class MiniSFT(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = [json.loads(x) for x in open(jsonl_path, "r", encoding="utf-8")]
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ex = self.items[idx]
        img = Image.open(ex["image"]).convert("RGB")
        # conversations: [{"from":"user","value":...},{"from":"assistant","value":...}]
        # 让 processor 处理成输入序列；MiniCPM-V 的 processor 支持多模态对话格式
        conversation = ex["conversations"]
        # 格式示例（若需要自定义，参照模型卡/官方脚本的对话模板）
        inputs = processor(
            images=img,
            text=conversation,
            return_tensors="pt",
            max_length=MAX_LEN,
            padding="longest",
            truncation=True
        )
        # 语言模型监督：labels 与 input_ids 对齐，mask 掉用户部分（只让模型学 assistant）
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        labels = input_ids.clone()
        # 简单规则：把第一轮 user 的 token 标为 -100（不计 loss）
        # 这里示例写法，真实可根据 processor 返回的 role 标记精确 mask
        # 保险起见：仅保留最后一轮 assistant 的 token 计算 loss
        if "role_ids" in inputs:  # 有些实现会返回角色标记
            role_ids = inputs["role_ids"][0]
            labels[role_ids!=2] = -100  # 2 假设是 assistant 角色 id
        else:
            # 粗略做法：保留后半段（可能是 assistant），示例而已
            half = labels.numel()//2
            labels[:half] = -100

        return {
            "pixel_values": inputs.get("pixel_values", None),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate(batch: List[Dict]):
    # 视觉 + 文本 padding
    pixel_values = None
    if batch[0]["pixel_values"] is not None:
        pixel_values = torch.stack([b["pixel_values"][0] for b in batch], dim=0)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

train_ds = MiniSFT(DATA_JSONL)
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=4, collate_fn=collate)

# ========= Optim & Scheduler =========
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
num_steps = EPOCHS * (len(loader) // GRAD_ACC + 1)
warmup = int(WARMUP * num_steps)
scheduler = get_cosine_schedule_with_warmup(optim, warmup, num_steps)

# ========= Train =========
model.train()
global_step, accu = 0, 0
scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

for ep in range(EPOCHS):
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
            out = model(**batch)
            loss = out.loss / GRAD_ACC

        scaler.scale(loss).backward()
        accu += 1
        if accu % GRAD_ACC == 0:
            scaler.step(optim); scaler.update(); scheduler.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 20 == 0:
                print(f"step {global_step}/{num_steps}  loss={loss.item()*GRAD_ACC:.4f}")

# ========= Save LoRA adapter =========
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Saved LoRA adapter to", OUTPUT_DIR)

# ======== Inference Demo ========
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image
import torch

BASE = "openbmb/MiniCPM-V-2_6"
ADAPTER = "minicpm_v_lora_sft"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(ADAPTER, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                            device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER).eval()

img = Image.open("demo.jpg").convert("RGB")
messages = [
    {"from":"user","value":"请描述这张图片的主要内容。"}
]

inputs = processor(images=img, text=messages, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
