"""Chấm điểm ảnh shop chính hãng (Adidas, Nike) để chọn mẫu âm thật cho train / đo.

- Bỏ banner khuyến mãi (cỡ 1000×1000, 1200×1200: khung LazMall + ô giảm giá) và ảnh đã có trong poc/.../lazmall.
- Mỗi ảnh: điểm lớn nhất của probe hiện tại (work/probe.pt) → mẫu âm khó; OWLv2 phân loại áo / giày / phụ kiện.
Ghi work/store_scores.json. Chạy: .venv\\Scripts\\python.exe -X utf8 score_store_images.py
"""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, Owlv2ForObjectDetection, Owlv2Processor

ROOT = Path(r"D:\Code\logo-obstruct-detection")
WORK = ROOT / "work"
POC = Path(r"D:\Code\TH.AntiFakeExt\poc")
STORES = [Path(r"D:\Code\KGM\dataset\adidas1631618372-images"), Path(r"D:\Code\KGM\dataset\nike-flagship-store-images")]
BANNER_SIZES = {(1000, 1000), (1200, 1200)}
DEV, DT = "cuda", torch.float16
SIZE, PATCH = 518, 14; G = SIZE // PATCH; LAYERS = (4, 8, 12)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
CATS = {"apparel": ["a t-shirt", "a shirt", "a jacket", "a hoodie", "pants", "shorts", "a dress", "a sports bra"],
        "shoe": ["a shoe", "a sneaker", "a sandal"],
        "accessory": ["a bag", "a backpack", "socks", "a cap", "a ball"]}

dino = AutoModel.from_pretrained(POC / "models/dinov2-base", dtype=DT).to(DEV).eval()
sd = torch.load(WORK / "probe.pt")


class Probe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.register_buffer("mu", torch.zeros(d)); self.register_buffer("sd", torch.ones(d))

    def forward(self, x):
        return self.net((x.float() - self.mu) / self.sd).squeeze(-1)


probe = Probe(sd["mu"].numel()).to(DEV); probe.load_state_dict(sd); probe.eval()
proc = Owlv2Processor.from_pretrained(POC / "models/owlv2-base")
owl = Owlv2ForObjectDetection.from_pretrained(POC / "models/owlv2-base", dtype=DT).to(DEV).eval()
phr = [p for v in CATS.values() for p in v]
cat_of = [c for c, v in CATS.items() for _ in v]
tok = proc(text=[phr], return_tensors="pt")

lazmall = {p.name for p in (POC / "data/logo_eval/images/lazmall").iterdir()}
paths = []
for d in STORES:
    for p in sorted(d.iterdir()):
        if p.name in lazmall:
            continue
        with Image.open(p) as im:
            if im.size in BANNER_SIZES:
                continue
        paths.append(p)
print(len(paths), "ảnh sau khi bỏ banner và ảnh trùng lazmall")

out, t0 = {}, time.time()
with torch.inference_mode():
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB"); im.thumbnail((1024, 1024))
        img = np.asarray(im); h, w = img.shape[:2]
        s = SIZE / max(h, w); nh, nw = round(h * s), round(w * s)
        lb = np.zeros((SIZE, SIZE, 3), np.uint8); lb[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy((lb.astype(np.float32) / 255 - MEAN) / STD).permute(2, 0, 1)[None].to(DEV, DT)
        hs = dino(pixel_values=x, output_hidden_states=True).hidden_states
        pr = probe(torch.cat([hs[l][:, 1:] for l in LAYERS], -1)[0]).sigmoid().cpu().numpy().reshape(G, G)
        pr = pr[:math.ceil(nh / PATCH), :math.ceil(nw / PATCH)]
        pv = proc(images=im, return_tensors="pt")["pixel_values"].to(DEV, DT)
        o = owl(input_ids=tok["input_ids"].to(DEV), attention_mask=tok["attention_mask"].to(DEV), pixel_values=pv)
        sc = o.logits[0].float().sigmoid().max(0).values.cpu().numpy()      # điểm lớn nhất mỗi cụm từ
        cs = {c: float(max(sc[j] for j in range(len(phr)) if cat_of[j] == c)) for c in CATS}
        out[str(p)] = {"probe_max": float(pr.max()), "probe_area_090": float((pr >= 0.9).mean()),
                       "cat": max(cs, key=cs.get), "cat_scores": cs, "store": p.parent.name}
        if (k + 1) % 250 == 0:
            print(f"{k + 1}/{len(paths)} · {(time.time() - t0) / (k + 1) * 1000:.0f} ms/ảnh", flush=True)
(WORK / "store_scores.json").write_text(json.dumps(out), encoding="utf-8")
print("xong", len(out))
