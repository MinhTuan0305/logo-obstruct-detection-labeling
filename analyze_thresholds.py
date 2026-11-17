"""Phân tích model probe đã lưu (work/probe.pt): ảnh sạch val nào bị báo, và tỉ lệ bắt / báo thừa theo ngưỡng."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from transformers import AutoModel

ROOT = Path(r"D:\Code\logo-obstruct-detection")
WORK = ROOT / "work"; SYN = WORK / "synth"
OUT = WORK  # tờ ảnh ảnh sạch bị báo nhiều nhất: work/clean_fp_top12.jpg
DEV, DT = "cuda", torch.float16
SIZE, PATCH = 518, 14; G = SIZE // PATCH; LAYERS = (4, 8, 12)
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
dino = AutoModel.from_pretrained(r"D:\Code\TH.AntiFakeExt\poc\models\dinov2-base", dtype=DT).to(DEV).eval()
sd = torch.load(WORK / "probe.pt")


class Probe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.register_buffer("mu", torch.zeros(d)); self.register_buffer("sd", torch.ones(d))

    def forward(self, x):
        return self.net((x.float() - self.mu) / self.sd).squeeze(-1)


probe = Probe(sd["mu"].numel()).to(DEV); probe.load_state_dict(sd); probe.eval()


def letterbox(img):
    h, w = img.shape[:2]; s = SIZE / max(h, w); nh, nw = round(h * s), round(w * s)
    out = np.zeros((SIZE, SIZE, 3), np.uint8); out[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, (nh, nw)


@torch.no_grad()
def full_heat(img):
    lb, (nh, nw) = letterbox(img)
    x = torch.from_numpy((lb.astype(np.float32) / 255 - MEAN) / STD).permute(2, 0, 1)[None].to(DEV, DT)
    hs = dino(pixel_values=x, output_hidden_states=True).hidden_states
    f = torch.cat([hs[l][:, 1:] for l in LAYERS], -1)[0]
    p = probe(f).sigmoid().cpu().numpy().reshape(G, G)
    v = np.zeros((G, G), bool); v[:math.ceil(nh / PATCH), :math.ceil(nw / PATCH)] = True; p[~v] = 0
    up = cv2.resize(p, (SIZE, SIZE))[:nh, :nw]
    return cv2.resize(up, (img.shape[1], img.shape[0]))


# 1) ảnh sạch val theo nguồn (74 ảnh giày đứng đầu danh sách val, còn lại là áo LogoDet)
clean = sorted(SYN.glob("val_*_4.jpg"))
rows = []
for p in clean:
    k = int(p.stem.split("_")[1])
    img = np.asarray(Image.open(p).convert("RGB"))
    rows.append((p, "giày" if k < 74 else "áo", float(full_heat(img).max())))
for src in ("giày", "áo"):
    m = np.array([r[2] for r in rows if r[1] == src])
    if not len(m):
        continue
    print(f"ảnh sạch val ({src}, {len(m)}): " + " · ".join(f"≥{t}: {np.mean(m >= t):.0%}" for t in (0.8, 0.9, 0.94, 0.98)))
worst = sorted(rows, key=lambda r: -r[2])[:12]
tiles = []
for p, src, mx in worst:
    img = np.asarray(Image.open(p).convert("RGB")); hm = full_heat(img)
    o = (img * 0.55 + (cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]) * 0.45).astype(np.uint8)
    im = Image.fromarray(o); im.thumbnail((300, 300)); t = Image.new("RGB", (300, 320), "white"); t.paste(im, (0, 20))
    ImageDraw.Draw(t).text((3, 3), f"{src} {mx:.3f} {p.stem[:26]}", fill=0); tiles.append(np.asarray(t))
while len(tiles) % 6:
    tiles.append(np.full((320, 300, 3), 255, np.uint8))
Image.fromarray(np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])).save(OUT / "clean_fp_top12.jpg", quality=85)


# 2) ảnh thật: bắt được / báo thừa theo ngưỡng
def box_iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def grow(b, f, W, H):
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * f, (b[3] - b[1]) * f
    return [max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)]


for name, d, ann in (("dataset", "dataset", "real_boxes.json"), ("dataset2", "dataset2", "dataset2_boxes.json")):
    A = json.loads((ROOT / "annotations" / ann).read_text(encoding="utf-8"))["images"]
    heats = {k: full_heat(np.asarray(Image.open(ROOT / d / k).convert("RGB"))) for k in A}
    for t in (0.8, 0.9, 0.94, 0.98):
        hit = tot = fa = 0; byk = {}
        for k, gts in A.items():
            hm = heats[k]; H, W = hm.shape
            for g in gts:
                x0, y0, x1, y1 = map(int, g["bbox"]); ok = hm[y0:y1, x0:x1].max() >= t
                tot += 1; hit += ok; byk.setdefault(g["kind"], [0, 0]); byk[g["kind"]][0] += ok; byk[g["kind"]][1] += 1
            n, lab, st, _ = cv2.connectedComponentsWithStats((hm >= t).astype(np.uint8))
            for i in range(1, n):
                x, y, w, h, a = st[i]
                if a >= 0.0004 * H * W and not any(box_iou([x, y, x + w, y + h], grow(g["bbox"], 1.3, W, H)) > 0 for g in gts):
                    fa += 1
        print(f"{name} ngưỡng {t}: bắt {hit}/{tot} = {hit / tot:.0%} · báo thừa {fa} · "
              + " ".join(f"{kk} {a}/{b}" for kk, (a, b) in byk.items()))

# 3) ảnh shop chính hãng giữ riêng (test, không che): mọi khung đều là báo thừa
SEL = ROOT / "annotations" / "store_selection.json"
if SEL.exists():
    sel = json.loads(SEL.read_text(encoding="utf-8"))
    sc = json.loads((WORK / "store_scores.json").read_text(encoding="utf-8")) if (WORK / "store_scores.json").exists() else {}
    heats = {}
    for p in sel["test"]:
        im = Image.open(p).convert("RGB"); im.thumbnail((1024, 1024)); heats[p] = full_heat(np.asarray(im))
    for t in (0.8, 0.9, 0.94, 0.98):
        by = {}
        for p, hm in heats.items():
            c = sc.get(p, {}).get("cat", "?")
            n, lab, st, _ = cv2.connectedComponentsWithStats((hm >= t).astype(np.uint8))
            nb = sum(st[i][4] >= 0.0004 * hm.size for i in range(1, n))
            a = by.setdefault(c, [0, 0, 0]); a[0] += hm.max() >= t; a[1] += 1; a[2] += nb
        tot = [sum(v[i] for v in by.values()) for i in range(3)]
        print(f"shop test ngưỡng {t}: {tot[0]}/{tot[1]} ảnh ({tot[0] / tot[1]:.0%}) bị báo, {tot[2]} khung · "
              + " ".join(f"{c} {a}/{n}" for c, (a, n, _) in sorted(by.items())))
