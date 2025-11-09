# train_clip_scratch.py
# 从零搭建并训练一个最小可用的 CLIP（双塔 + 对比学习）
# 依赖: pip install torch torchvision datasets transformers accelerate pillow
import math, random, os
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16
from PIL import Image

from datasets import load_dataset
from transformers import BertTokenizerFast, get_cosine_schedule_with_warmup

# ===================== 配置 =====================
@dataclass
class Config:
    image_size: int = 224
    img_embed_dim: int = 768         # ViT-B/16 的 d_model
    txt_embed_dim: int = 512         # 文本塔 hidden size
    proj_dim: int = 512              # 共享多模态空间维度
    txt_layers: int = 6
    txt_heads: int = 8
    txt_ffn: int = 2048
    max_len: int = 77
    batch_size: int = 128
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    temperature: float = 0.07
    seed: int = 42
    dataset_max_samples: int | None = None   # 调试时可设小一点
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

# ===================== 数据集 =====================
# Flickr30k: 每条包含 "image" (PIL) 与 "sentences" (list of dict, 'raw' 为文本)
ds = load_dataset("flickr30k", split="train")
if cfg.dataset_max_samples:
    ds = ds.select(range(min(cfg.dataset_max_samples, len(ds))))

def pick_one_caption(example):
    sents = example["sentences"]
    cap = random.choice(sents)["raw"] if sents else ""
    return {"image": example["image"], "text": cap}

ds = ds.map(pick_one_caption, remove_columns=ds.column_names)

# 图像预处理（与 ViT 输入一致）
img_tf = transforms.Compose([
    transforms.Resize(cfg.image_size, interpolation=Image.BICUBIC),
    transforms.CenterCrop(cfg.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

# 文本分词（使用 BERT tokenizer 省去自己训练 BPE；权重不共享，只作为分词器）
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def collate(batch: List[Dict]):
    images = [img_tf(x["image"].convert("RGB")) for x in batch]
    texts = [x["text"] for x in batch]

    pixel_values = torch.stack(images, dim=0)  # [B, 3, H, W]
    # 使用 [CLS] ... [SEP]，pad 到 max_len
    tok = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=cfg.max_len,
        return_tensors="pt",
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": tok.input_ids,
        "attention_mask": tok.attention_mask,
    }

loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, collate_fn=collate)

# ===================== 文本塔（从零实现） =====================
class TextTransformer(nn.Module):
    def __init__(self, vocab, d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_len=77):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask):
        # input_ids: [B, L], attention_mask: [B, L] (1=keep, 0=pad)
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).repeat(B, 1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        # 需要将 mask 转成 transformer 期望的 True=pad
        src_key_padding_mask = (attention_mask == 0)  # [B, L], True 表示要 mask
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln(x)
        # 取 [CLS] 位置（BERT tokenizer 的 cls=101）作为句向量；若无 cls，可用第一个 token
        cls_index = (input_ids == tokenizer.cls_token_id).long().argmax(dim=1)
        sent_embed = x[torch.arange(B), cls_index]  # [B, d_model]
        return sent_embed

# ===================== 视觉塔（随机初始化 ViT） =====================
class VisionBackbone(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        # vit_b_16: d_model=768, num_heads=12, depth=12
        # weights=None 确保随机初始化
        self.vit = vit_b_16(weights=None)
        self.out_dim = out_dim

    def forward(self, pixel_values):
        # torchvision ViT 前向输出类别 token 的特征向量 [B, 1000] 若使用默认分类头；
        # 我们需要中间的 encoder 输出。修改：拿 vit 的 encoder 输出的 cls token。
        # 简化做法：复用 vit.forward_features 获取 [B, 768]
        x = self.vit._process_input(pixel_values)          # patchify
        n = x.shape[0]
        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.layers(x)                    # Transformer blocks
        x = self.vit.encoder.ln(x)
        cls = x[:, 0]                                     # [B, 768]
        return cls

# ===================== CLIP 主体（双塔 + 投影） =====================
class CLIPScratch(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.vision = VisionBackbone(out_dim=cfg.img_embed_dim)
        self.text = TextTransformer(
            vocab=vocab_size, d_model=cfg.txt_embed_dim,
            n_heads=cfg.txt_heads, n_layers=cfg.txt_layers,
            d_ff=cfg.txt_ffn, max_len=cfg.max_len
        )
        self.img_proj = nn.Linear(cfg.img_embed_dim, cfg.proj_dim)
        self.txt_proj = nn.Linear(cfg.txt_embed_dim, cfg.proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / cfg.temperature)))

    def encode_image(self, pixel_values):
        x = self.vision(pixel_values)
        x = self.img_proj(x)
        x = F.normalize(x, dim=-1)
        return x

    def encode_text(self, input_ids, attention_mask):
        x = self.text(input_ids, attention_mask)
        x = self.txt_proj(x)
        x = F.normalize(x, dim=-1)
        return x

    def forward(self, pixel_values, input_ids, attention_mask):
        zi = self.encode_image(pixel_values)
        zt = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        logits = logit_scale * (zi @ zt.t())  # [B, B]
        return logits

def clip_loss(logits):
    # 双向 InfoNCE
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2

# ===================== 实例化模型/优化器 =====================
vocab_size = tokenizer.vocab_size
model = CLIPScratch(cfg, vocab_size).to(cfg.device)

params = [p for p in model.parameters() if p.requires_grad]
optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

num_steps = cfg.epochs * math.ceil(len(loader))
num_warmup = int(cfg.warmup_ratio * num_steps)
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup, num_steps)

scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device == "cuda"))

# ===================== 训练 =====================
model.train()
global_step = 0
for epoch in range(1, cfg.epochs + 1):
    running = 0.0
    for step, batch in enumerate(loader, 1):
        pixel_values = batch["pixel_values"].to(cfg.device, non_blocking=True)
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(cfg.device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.device=="cuda")):
            logits = model(pixel_values, input_ids, attention_mask)  # [B, B]
            loss = clip_loss(logits)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        running += loss.item()
        global_step += 1

        if step % 50 == 0:
            print(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss {running/50:.4f} | scale={model.logit_scale.exp().item():.3f}")
            running = 0.0

# ===================== 保存 =====================
os.makedirs("clip_scratch_ckpt", exist_ok=True)
torch.save(model.state_dict(), "clip_scratch_ckpt/model.pt")
print("Saved to clip_scratch_ckpt/model.pt")
