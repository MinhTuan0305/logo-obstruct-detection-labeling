"""Nạp và chạy các model phát hiện vùng che: probe DINOv2 (work/probe.pt), RF-DETR (work/rfdetr_real_out/…), kết hợp.

Dùng chung cho notebook gán nhãn gợi ý (chuan_bi_du_lieu_that.ipynb) và các script khác.

    from obstruct_models import Models
    m = Models(probe=True, rfdetr=True)
    img = m.load("anh.jpg")                    # RGB, cạnh dài ≤ 1024 (như lúc train)
    vung = m.ket_hop(img)                      # [{"bbox": [x0,y0,x1,y1], "diem_rfdetr":…, "diem_probe":…}, …]

Lọc theo vùng sản phẩm (LOC_SAN_PHAM): OWLv2 tìm giày / áo / túi, chỉ giữ khung che nằm trong khung sản phẩm
(nới 8%). Không thấy sản phẩm nào thì giữ nguyên mọi khung. Tắt: LOC_SAN_PHAM = False, hoặc Models(loc_san_pham=False),
hoặc từng lần gọi: m.rfdetr(img, loc_san_pham=False).
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
# RF-DETR fine-tune thêm trên ảnh che thật + mẫu âm shop (bản cũ chỉ dữ liệu tổng hợp: work/rfdetr_out/…)
RFDETR_CKPT = WORK / "rfdetr_real_out" / "checkpoint_best_total.pth"
# OWLv2-base (google/owlv2-base-patch16-ensemble), chỉ đọc. Máy khác: đặt biến môi trường OWL_DIR.
OWL_DIR = Path(os.environ.get("OWL_DIR", r"D:\Code\TH.AntiFakeExt\poc\models\owlv2-base"))

# ---------- lọc theo vùng sản phẩm: bật / tắt ở đây ----------
LOC_SAN_PHAM = True                                  # True: bỏ khung che nằm ngoài sản phẩm
SAN_PHAM_PHRASES = ["shoe", "sneaker", "shirt", "clothing", "bag", "shoe box"]
NGUONG_SAN_PHAM = 0.1                                # điểm OWLv2 tối thiểu để tính là sản phẩm (0.15 làm sót giày trong ảnh ghép)
NOI_SAN_PHAM = 1.08                                  # nới khung sản phẩm 8% mỗi chiều
TI_LE_TRONG = 0.5                                    # giữ khung che nếu ≥ 50% diện tích nằm trong vùng sản phẩm
# --------------------------------------------------------------
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


class TimSanPham:
    """OWLv2 tìm vùng sản phẩm. Gọi: sp = TimSanPham(); khung = sp(img) → [[x0,y0,x1,y1], …] (đã nới), [] nếu không thấy."""

    def __init__(self, phrases=SAN_PHAM_PHRASES, thr=NGUONG_SAN_PHAM, noi=NOI_SAN_PHAM, owl_dir=OWL_DIR):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.proc = Owlv2Processor.from_pretrained(owl_dir)
        self.owl = Owlv2ForObjectDetection.from_pretrained(owl_dir, dtype=self.dt).to(self.dev).eval()
        self.tok = self.proc(text=[list(phrases)], return_tensors="pt")
        self.thr, self.noi = thr, noi

    @torch.inference_mode()
    def __call__(self, img):
        h, w = img.shape[:2]; S = max(h, w)   # OWLv2 pad ảnh vuông ở dưới/phải
        pv = self.proc(images=Image.fromarray(img), return_tensors="pt")["pixel_values"].to(self.dev, self.dt)
        o = self.owl(input_ids=self.tok["input_ids"].to(self.dev), attention_mask=self.tok["attention_mask"].to(self.dev),
                     pixel_values=pv)
        sc = o.logits[0].float().sigmoid().max(-1).values.cpu().numpy()
        cx, cy, bw, bh = (o.pred_boxes[0].float().cpu().numpy() * S).T
        out = []
        for i in np.nonzero(sc >= self.thr)[0]:
            if bw[i] * bh[i] < 0.02 * w * h:   # bỏ khung quá nhỏ (chi tiết, không phải cả sản phẩm)
                continue
            hw, hh = bw[i] * self.noi / 2, bh[i] * self.noi / 2
            out.append([float(max(0, cx[i] - hw)), float(max(0, cy[i] - hh)), float(min(w, cx[i] + hw)), float(min(h, cy[i] + hh))])
        return out


def mat_na_san_pham(khung_sp, shape):
    """Mặt nạ 0/1 (float32) = hợp các khung sản phẩm; không có khung nào → cả ảnh là 1 (không lọc)."""
    m = np.zeros(shape[:2], np.float32)
    for x0, y0, x1, y1 in khung_sp:
        m[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))] = 1
    return m if m.any() else np.ones(shape[:2], np.float32)


def trong_san_pham(bbox, mat_na, ti_le=TI_LE_TRONG):
    """True nếu ≥ ti_le diện tích khung nằm trong mặt nạ sản phẩm."""
    H, W = mat_na.shape
    x0, y0 = int(max(0, bbox[0])), int(max(0, bbox[1]))
    x1, y1 = int(min(W, math.ceil(bbox[2]))), int(min(H, math.ceil(bbox[3])))
    if x1 <= x0 or y1 <= y0:
        return False
    return float(mat_na[y0:y1, x0:x1].mean()) >= ti_le


class Models:
    def __init__(self, probe=True, rfdetr=True, rfdetr_ckpt=RFDETR_CKPT, probe_pt=PROBE_PT, loc_san_pham=None):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.dino = self.probe = self.rf = self.sp = None
        self.loc_san_pham = LOC_SAN_PHAM if loc_san_pham is None else loc_san_pham
        if self.loc_san_pham:
            self.sp = TimSanPham()
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

    def mat_na_san_pham(self, img):
        """Mặt nạ vùng sản phẩm của img (cả ảnh = 1 nếu không thấy sản phẩm). Nạp OWLv2 nếu chưa nạp."""
        if self.sp is None:
            self.sp = TimSanPham()
        return mat_na_san_pham(self.sp(img), img.shape)

    def _loc(self, ks, img, loc_san_pham, mat_na):
        """Tách ks thành (trong sản phẩm, ngoài sản phẩm)."""
        if not (self.loc_san_pham if loc_san_pham is None else loc_san_pham) or not ks:
            return ks, []
        mat_na = self.mat_na_san_pham(img) if mat_na is None else mat_na
        trong = [k for k in ks if trong_san_pham(k["bbox"], mat_na)]
        return trong, [dict(k, ly_do="ngoai_san_pham") for k in ks if not trong_san_pham(k["bbox"], mat_na)]

    def rfdetr(self, img, thr=0.3, nms_iou=0.5, loc_san_pham=None, mat_na=None, tra_ca_bi_loai=False):
        """Khung RF-DETR điểm ≥ thr, gộp khung gần trùng. loc_san_pham=None → theo cài đặt của Models."""
        d = self.rf.predict(Image.fromarray(img), threshold=thr)
        ks = sorted(({"bbox": [float(v) for v in b], "diem_rfdetr": round(float(s), 3)} for b, s in zip(d.xyxy, d.confidence)),
                    key=lambda k: -k["diem_rfdetr"])
        out = []
        for k in ks:
            if all(iou(k["bbox"], g["bbox"]) < nms_iou for g in out):
                out.append(k)
        giu, loai = self._loc(out, img, loc_san_pham, mat_na)
        return (giu, loai) if tra_ca_bi_loai else giu

    def ket_hop(self, img, thr_rf=0.3, thr_probe=0.9, heat=None, tra_ca_bi_loai=False, loc_san_pham=None, mat_na=None):
        """RF-DETR đề xuất (đã lọc vùng sản phẩm nếu bật), probe kiểm lại: giữ khung nếu điểm probe lớn nhất trong khung ≥ thr_probe.
        Khung bị loại có "ly_do": "ngoai_san_pham" hoặc "probe_thap"."""
        heat = self.heat(img) if heat is None else heat
        de_xuat, loai = self.rfdetr(img, thr_rf, loc_san_pham=loc_san_pham, mat_na=mat_na, tra_ca_bi_loai=True)
        giu = []
        for k in de_xuat:
            x0, y0, x1, y1 = int(max(0, k["bbox"][0])), int(max(0, k["bbox"][1])), int(k["bbox"][2]) + 1, int(k["bbox"][3]) + 1
            k["diem_probe"] = round(float(heat[y0:y1, x0:x1].max()), 3) if heat[y0:y1, x0:x1].size else 0.0
            if k["diem_probe"] >= thr_probe:
                giu.append(k)
            else:
                loai.append(dict(k, ly_do="probe_thap"))
        return (giu, loai) if tra_ca_bi_loai else giu
