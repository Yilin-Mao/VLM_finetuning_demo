import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

print('imported modules successfully.')
# --------- 配置 ----------
MODEL_ID = "openai/clip-vit-base-patch16"
BATCH_SIZE = 32            # 显存不够可降到 32/16
LR = 5e-5                  # 仅训练 LoRA，学习率可略大于全参微调
EPOCHS = 3
WARMUP_RATIO = 0.05
MAX_SAMPLES = None         # 调试可设 5000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TEMPERATURE = 0.07         # 对比学习温度

random.seed(SEED)
torch.manual_seed(SEED)

# --------- 数据集 ----------
# Flickr30k: 每条包含 'image'(PIL) 和 'caption' (list of dict, 含 'raw')

# 确保 revision 参数是唯一的关键差异
PARQUET_REVISION = "refs/convert/parquet"

# 这是正确的远程加载方式
try:
    dataset = load_dataset(
        "nlphuji/flickr30k",
        split="test",
        revision=PARQUET_REVISION
    )
    print("数据集加载成功。")
except Exception as e:
    # 如果仍然报错，请尝试下一步
    print(f"加载失败，错误: {e}")

# dataset = load_dataset("nlphuji/flickr30k/refs/convert/parquet", split="train")
# dataset = load_dataset("flickr30k", split="train")
print(f"Dataset loaded with {len(dataset)} samples.")

if MAX_SAMPLES:
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

# 随机取一条 caption；也可以扩展为多 caption 复制样本
def pick_caption(example):
    sents = example["caption"]
    if isinstance(sents, list) and len(sents) > 0:
        ran_num = random.randint(0, len(sents)-1)
        cap = random.choice(sents)[ran_num]
    else:
        cap = ""
    return {"image": example["image"], "text": cap}

dataset = dataset.map(pick_caption, remove_columns=dataset.column_names)
# 现在 dataset 只有 "image": PIL.Image, "text": str
print("Captions picked.")

# --------- 模型与处理器 ----------
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE)
model.to(DEVICE)
print("Model and processor loaded.")

# 冻结全参（peft 会把 LoRA 层设为可训练）
for p in model.parameters():
    p.requires_grad = False

# 给视觉/文本编码器的自注意力投影层加 LoRA
# 模块名来自 transformers 的 CLIP 实现：
# text_model.encoder.layers.*.self_attn.(q_proj|k_proj|v_proj|out_proj)
# vision_model.encoder.layers.*.self_attn.(q_proj|k_proj|v_proj|out_proj)
target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
peft_cfg = LoraConfig(
    r=16,                    # rank，可调 8/16/32
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules,
    task_type="FEATURE_EXTRACTION",   # 对比学习风格，不是生成
)
model = get_peft_model(model, peft_cfg)
# 打印可训练参数比例
trainable, total = 0, 0
for n, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable params: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M ({100*trainable/total:.2f}%)")

# --------- DataLoader & collator ----------
def collate_fn(batch):
    images = [x["image"] for x in batch]
    texts = [x["text"] for x in batch]
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return inputs

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, collate_fn=collate_fn, pin_memory=True
)
print("DataLoader prepared.")

# --------- 优化器与调度 ----------
# 只优化可训练(LoRA)参数
optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
num_training_steps = EPOCHS * math.ceil(len(loader))
num_warmup = int(WARMUP_RATIO * num_training_steps)
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup, num_training_steps)

scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

# --------- 对比学习损失（双向 InfoNCE） ----------
def clip_contrastive_loss(image_embeds, text_embeds, temperature=TEMPERATURE):
    # L2 normalize
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds  = F.normalize(text_embeds,  dim=-1)
    logits = (image_embeds @ text_embeds.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2.0

# --------- 训练 ----------
global_step = 0
model.train()
for epoch in range(1, EPOCHS+1):
    running = 0.0
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
            # 直接拿特征（无需分类头）
            image_embeds = model.get_image_features(pixel_values=batch["pixel_values"])
            text_embeds  = model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = clip_contrastive_loss(image_embeds, text_embeds)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        running += loss.item()
        global_step += 1

        if step % 5 == 0:
            print(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss {running/50:.4f}")
            running = 0.0

# --------- 保存 LoRA 适配器 ----------
SAVE_DIR = "lora_clip_adapter"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
print(f"LoRA adapter saved to: {SAVE_DIR}")
