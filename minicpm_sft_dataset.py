# build_minicpm_sft_dataset.py
from datasets import load_dataset
from PIL import Image
import os, json, uuid

OUT_JSONL = "minicpm_sft_llava150k.jsonl"
IMG_ROOT = "llava_images"  # 会自动硬链接/下载到本地缓存
os.makedirs(IMG_ROOT, exist_ok=True)

ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")

def to_mini_item(ex):
    # LLaVA 格式字段：'image', 'conversations'（list，含from/ value）
    # 取第一轮问答；多轮可自行展开
    conv = ex["conversations"]
    if not conv or len(conv) < 2: 
        return None
    # LLaVA 用字符串路径或 COCO 名称，datasets 会自动抓取
    img_path = ex["image"]
    # 保证图片可被本地读取（datasets cache 中），也可 copy 到 IMG_ROOT
    return {
        "id": str(uuid.uuid4()),
        "image": img_path,               # 或者改成绝对路径
        "conversations": [
            {"from": "user", "value": conv[0]["value"]},
            {"from": "assistant", "value": conv[1]["value"]}
        ]
    }

items = []
for ex in ds:
    it = to_mini_item(ex)
    if it: items.append(it)

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

print("Wrote:", OUT_JSONL, "size:", len(items))
