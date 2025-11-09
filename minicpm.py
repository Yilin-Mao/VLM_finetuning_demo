# minicpmv_scratch_tiny.py
# 教学版: 从零搭建“MiniCPM-V 风格”主干并做小规模训练（图像条件下生成caption）
# 依赖: pip install torch torchvision datasets transformers pillow

import math, random, os
from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.vision_transformer import vit_b_16
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from transformers import BertTokenizerFast, get_cosine_schedule_with_warmup

# ============ Config ============
@dataclass
class Cfg:
    image_size: int = 224
    vit_dim: int = 768              # ViT-B/16 hidden size
    n_vis_tokens: int = 64          # Resampler输出视觉token数（典型64）
    llm_dim: int = 512              # 小型LLM隐藏维
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 2048
    max_txt_len: int = 64           # 生成端最大文本长度（教学）
    batch_size: int = 16
    epochs: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
cfg = Cfg()
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

# ============ 数据（Flickr30k） ============
ds = load_dataset("flickr30k", split="train")
def pick_one_caption(ex):
    sents = ex["sentences"]
    cap = random.choice(sents)["raw"] if sents else "a photo."
    return {"image": ex["image"], "text": cap}
ds = ds.map(pick_one_caption, remove_columns=ds.column_names)

img_tf = transforms.Compose([
    transforms.Resize(cfg.image_size, interpolation=Image.BICUBIC),
    transforms.CenterCrop(cfg.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def collate(batch: List[Dict]):
    images = [img_tf(x["image"].convert("RGB")) for x in batch]
    texts  = [x["text"] for x in batch]
    pixel_values = torch.stack(images, dim=0)                 # [B,3,H,W]
    tok = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=cfg.max_txt_len, return_tensors="pt"
    )
    # LM 训练的labels（shifted inside model）
    labels = tok.input_ids.clone()
    return {
        "pixel_values": pixel_values,
        "input_ids": tok.input_ids,
        "attention_mask": tok.attention_mask,
        "labels": labels
    }

loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, collate_fn=collate)

# ============ 视觉编码器（ViT特征） ============
class VisionBackbone(nn.Module):
    def __init__(self, d=cfg.vit_dim):
        super().__init__()
        self.vit = vit_b_16(weights=None)  # 随机初始化；想快可改用预训练
        self.d = d
    def forward(self, pixel_values):
        # 取 encoder 中所有 patch token（不只CLS），以供Resampler跨注意力读取
        x = self.vit._process_input(pixel_values)  # patchify -> [B, N, d]
        B = x.size(0)
        cls = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)             # [B, N+1, d]
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.layers(x)             # [B, T, d]
        x = self.vit.encoder.ln(x)
        # 返回所有token（含CLS），让Resampler自由选择
        return x                                    # [B, T, d]

# ============ Perceiver-Resampler（单层简化版） ============
# learnable queries Q 通过 cross-attn 读取 K,V=vision tokens
class PerceiverResampler(nn.Module):
    def __init__(self, d_in, d_out, n_queries=cfg.n_vis_tokens, n_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_out) / math.sqrt(d_out))
        self.q_proj = nn.Linear(d_out, d_out)
        self.k_proj = nn.Linear(d_in,  d_out)
        self.v_proj = nn.Linear(d_in,  d_out)
        self.out_proj = nn.Linear(d_out, d_out)
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_out),
            nn.Linear(d_out, int(d_out*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_out*mlp_ratio), d_out),
        )
        self.ln = nn.LayerNorm(d_out)

    def forward(self, vision_tokens):             # [B, T, d_in]
        B, T, _ = vision_tokens.shape
        q = self.q_proj(self.queries).unsqueeze(0).expand(B, -1, -1)  # [B, Q, d_out]
        k = self.k_proj(vision_tokens)                                  # [B, T, d_out]
        v = self.v_proj(vision_tokens)                                  # [B, T, d_out]

        # reshape to heads
        def split_heads(x):  # [B, L, d] -> [B, h, L, dh]
            return x.view(B, x.size(1), self.n_heads, self.d_head).transpose(1,2)
        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
        attn = (qh @ kh.transpose(-1, -2)) / math.sqrt(self.d_head)     # [B,h,Q,T]
        attn = attn.softmax(dim=-1)
        out = attn @ vh                                                 # [B,h,Q,dh]
        out = out.transpose(1,2).contiguous().view(B, q.size(1), -1)    # [B,Q,d_out]
        out = self.out_proj(out)
        # 残差+MLP
        out = out + self.mlp(self.ln(out))
        return out                                                      # [B,Q,d_out]

# ============ 小型 Causal LLM（文本生成） ============
class TinyCausalLM(nn.Module):
    def __init__(self, vocab, d=cfg.llm_dim, n_layers=cfg.n_layers, n_heads=cfg.n_heads, ffn=cfg.ffn_dim):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(cfg.max_txt_len + cfg.n_vis_tokens + 4, d)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,
                                               dim_feedforward=ffn, activation="gelu",
                                               batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, visual_tokens, input_ids, labels=None):
        """
        visual_tokens: [B, Q, d]   (由Resampler+投影得到)
        input_ids:     [B, L]
        """
        B, Q, d = visual_tokens.shape
        L = input_ids.size(1)
        txt = self.tok_emb(input_ids)                                   # [B,L,d]
        x = torch.cat([visual_tokens, txt], dim=1)                      # [B,Q+L,d]

        # 位置编码（简单绝对位置）
        pos = torch.arange(Q+L, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)

        # 因果mask：文本部分不能看未来；视觉token可被全文可见
        # 构造 (Q+L)x(Q+L) 的上三角 mask，仅对文本区域生效
        S = Q + L
        causal = torch.ones(S, S, device=x.device, dtype=torch.bool).triu(1)
        # 允许视觉token彼此与文本任意交互（不做未来屏蔽），仅对纯文本的未来屏蔽严格：
        # 这里简化：仍对全序列用causal mask，属于保守处理（教学）
        x = self.blocks(x, mask=causal)
        x = self.ln(x)
        logits = self.head(x[:, Q:, :])                                 # 只预测文本部分 [B,L,V]

        loss = None
        if labels is not None:
            # 左移一位做LM loss（teacher forcing）
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                                   labels[:, 1:].reshape(-1), ignore_index=0)
        return logits, loss

# ============ 组装“MiniCPM-V风格”主干 ============
class MiniCPMVTiny(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.vision = VisionBackbone(cfg.vit_dim)
        self.resampler = PerceiverResampler(d_in=cfg.vit_dim, d_out=cfg.llm_dim,
                                            n_queries=cfg.n_vis_tokens, n_heads=8)
        self.proj = nn.Linear(cfg.llm_dim, cfg.llm_dim)  # 这里保持维度一致，真实可带非线性
        self.lm = TinyCausalLM(vocab=vocab_size, d=cfg.llm_dim,
                               n_layers=cfg.n_layers, n_heads=cfg.n_heads, ffn=cfg.ffn_dim)

    def forward(self, pixel_values, input_ids, labels=None):
        vis_tokens = self.vision(pixel_values)                   # [B,T,vit_dim]
        vis_comp   = self.resampler(vis_tokens)                  # [B,Q,llm_dim]
        vis_comp   = self.proj(vis_comp)                         # [B,Q,llm_dim]
        logits, loss = self.lm(vis_comp, input_ids, labels)
        return logits, loss

# ============ 实例化与训练 ============
vocab = tokenizer.vocab_size
model = MiniCPMVTiny(cfg, vocab).to(cfg.device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
steps = cfg.epochs * math.ceil(len(loader))
warm = int(cfg.warmup_ratio * steps)
sch = get_cosine_schedule_with_warmup(opt, warm, steps)
scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device=="cuda"))

model.train()
gstep = 0
for ep in range(1, cfg.epochs+1):
    run = 0.0
    for it, batch in enumerate(loader, 1):
        pix = batch["pixel_values"].to(cfg.device, non_blocking=True)
        ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        labels = batch["labels"].to(cfg.device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device=="cuda")):
            _, loss = model(pix, ids, labels)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sch.step()

        run += loss.item(); gstep += 1
        if it % 50 == 0:
            print(f"Epoch {ep} | Step {it}/{len(loader)} | Loss {run/50:.4f}")
            run = 0.0

os.makedirs("minicpmv_tiny_ckpt", exist_ok=True)
torch.save(model.state_dict(), "minicpmv_tiny_ckpt/model.pt")
print("Saved to minicpmv_tiny_ckpt/model.pt")
