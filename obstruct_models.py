"""Nạp và chạy các model phát hiện vùng che: probe DINOv2 (work/probe.pt), RF-DETR (work/rfdetr_out/…), kết hợp.

Dùng chung cho notebook gán nhãn gợi ý (chuan_bi_du_lieu_that.ipynb) và các script khác.

    from obstruct_models import Models
    m = Models(probe=True, rfdetr=True)
    img = m.load("anh.jpg")                    # RGB, cạnh dài ≤ 1024 (như lúc train)
    vung = m.ket_hop(img)                      # [{"bbox": [x0,y0,x1,y1], "diem_rfdetr":…, "diem_probe":…}, …]
"""
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
# DINOv2-base (facebook/dinov2-base). Máy khác: đặt biến môi trường DINO_DIR trỏ tới thư mục model đã tải.
DINO_DIR = Path(os.environ.get("DINO_DIR", r"D:\Code\TH.AntiFakeExt\poc\models\dinov2-base"))
PROBE_PT = WORK / "probe.pt"
RFDETR_CKPT = WORK / "rfdetr_out" / "checkpoint_best_total.pth"
MAX_SIDE = 1024
SIZE, PATCH, LAYERS = 518, 14, (4, 8, 12)
G = SIZE // PATCH
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class _Probe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.register_buffer("mu", torch.zeros(d)); self.register_buffer("sd", torch.ones(d))

    def forward(self, x):
        return self.net((x.float() - self.mu) / self.sd).squeeze(-1)


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


class Models:
    def __init__(self, probe=True, rfdetr=True, rfdetr_ckpt=RFDETR_CKPT, probe_pt=PROBE_PT):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.dino = self.probe = self.rf = None
        if probe:
            from transformers import AutoModel
            self.dino = AutoModel.from_pretrained(DINO_DIR, dtype=self.dt).to(self.dev).eval()
            sd = torch.load(probe_pt, map_location=self.dev)
            self.probe = _Probe(sd["mu"].numel()).to(self.dev)
            self.probe.load_state_dict(sd); self.probe.eval()
        if rfdetr:
            from rfdetr import RFDETRSmall
            self.rf = RFDETRSmall(pretrain_weights=str(rfdetr_ckpt))

    @staticmethod
    def load(src):
        """Ảnh RGB (numpy), thu nhỏ cạnh dài về ≤ 1024 như lúc train (đổi khung về ảnh gốc: nhân W_gốc / img.shape[1])."""
        im = Image.open(src).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        return np.asarray(im)

    @torch.inference_mode()
    def heat(self, img):
        """Xác suất 'bị che' của probe cho từng điểm ảnh (0..1), cùng cỡ img."""
        h, w = img.shape[:2]; s = SIZE / max(h, w); nh, nw = round(h * s), round(w * s)
        lb = np.zeros((SIZE, SIZE, 3), np.uint8); lb[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy((lb.astype(np.float32) / 255 - MEAN) / STD).permute(2, 0, 1)[None].to(self.dev, self.dt)
        hs = self.dino(pixel_values=x, output_hidden_states=True).hidden_states
        q = self.probe(torch.cat([hs[l][:, 1:] for l in LAYERS], -1)[0]).sigmoid().float().cpu().numpy().reshape(G, G)
        v = np.zeros_like(q, bool); v[:math.ceil(nh / PATCH), :math.ceil(nw / PATCH)] = True; q[~v] = 0
        return cv2.resize(cv2.resize(q, (SIZE, SIZE))[:nh, :nw], (w, h))

    @staticmethod
    def boxes_from_heat(full, thr, min_area=0.0001):
        n, lab, st, _ = cv2.connectedComponentsWithStats((full >= thr).astype(np.uint8))
        H, W = full.shape
        return [{"bbox": [float(st[i][0]), float(st[i][1]), float(st[i][0] + st[i][2]), float(st[i][1] + st[i][3])],
                 "diem_probe": round(float(full[lab == i].max()), 3)}
                for i in range(1, n) if st[i][4] >= min_area * H * W]

    def rfdetr(self, img, thr=0.3, nms_iou=0.5):
        d = self.rf.predict(Image.fromarray(img), threshold=thr)
        ks = sorted(({"bbox": [float(v) for v in b], "diem_rfdetr": round(float(s), 3)} for b, s in zip(d.xyxy, d.confidence)),
                    key=lambda k: -k["diem_rfdetr"])
        out = []
        for k in ks:
            if all(iou(k["bbox"], g["bbox"]) < nms_iou for g in out):
                out.append(k)
        return out

    def ket_hop(self, img, thr_rf=0.3, thr_probe=0.9, heat=None, tra_ca_bi_loai=False):
        """RF-DETR đề xuất, probe kiểm lại: giữ khung nếu điểm probe lớn nhất trong khung ≥ thr_probe."""
        heat = self.heat(img) if heat is None else heat
        giu, loai = [], []
        for k in self.rfdetr(img, thr_rf):
            x0, y0, x1, y1 = int(max(0, k["bbox"][0])), int(max(0, k["bbox"][1])), int(k["bbox"][2]) + 1, int(k["bbox"][3]) + 1
            k["diem_probe"] = round(float(heat[y0:y1, x0:x1].max()), 3) if heat[y0:y1, x0:x1].size else 0.0
            (giu if k["diem_probe"] >= thr_probe else loai).append(k)
        return (giu, loai) if tra_ca_bi_loai else giu
