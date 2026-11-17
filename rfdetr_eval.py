"""Đo RF-DETR đã fine-tune trên ảnh thật, cùng thước đo với probe (analyze_thresholds.py).

- Ngưỡng gợi ý: chọn trên 60 ảnh shop chính hãng (val, không che) sao cho ~10% ảnh có khung báo.
- dataset/ (24 ảnh giày, 89 khung) và dataset2/ (20 ảnh áo, 75 khung): bắt được / khung thừa theo ngưỡng.
  Một vùng che "bắt được" nếu có khung dự đoán (điểm ≥ ngưỡng) trùng nó (IoU ≥ 0.1 hoặc tâm nằm trong).
- Shop test (150 ảnh sạch): số ảnh có khung, tổng số khung (đều là báo thừa).
Ghi tờ ảnh work/rfdetr_eval_{dataset,dataset2,shop}.jpg ở ngưỡng gợi ý.

Chạy: .venv\\Scripts\\python.exe -X utf8 rfdetr_eval.py [--ckpt work/rfdetr_real_out/checkpoint_best_total.pth] [--model small]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
THRS = (0.3, 0.4, 0.5, 0.6, 0.7)


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1])); inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def match(p, g):
    return iou(p, g) >= 0.1 or inside(((p[0] + p[2]) / 2, (p[1] + p[3]) / 2), g) or inside(((g[0] + g[2]) / 2, (g[1] + g[3]) / 2), p)


def grow(b, f, W, H):
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * f, (b[3] - b[1]) * f
    return [max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)]


def predict_all(model, paths):
    out = {}
    for p in paths:
        im = Image.open(p).convert("RGB")
        d = model.predict(im, threshold=0.05)
        out[str(p)] = {"size": im.size, "boxes": d.xyxy.tolist(), "scores": d.confidence.tolist()}
    return out


def eval_annotated(pred, ann_dir, ann):
    rows = []
    for t in THRS:
        hit = tot = fa = 0
        by = {}
        for name, gts in ann.items():
            r = pred[str(ann_dir / name)]
            W, H = r["size"]
            pb = [b for b, s in zip(r["boxes"], r["scores"]) if s >= t]
            for g in gts:
                ok = any(match(b, g["bbox"]) for b in pb)
                hit += ok; tot += 1
                k = by.setdefault(g["kind"], [0, 0]); k[0] += ok; k[1] += 1
            fa += sum(not any(iou(b, grow(g["bbox"], 1.3, W, H)) > 0 or match(b, g["bbox"]) for g in gts) for b in pb)
        rows.append((t, hit, tot, fa, by))
        print(f"  ngưỡng {t}: bắt {hit}/{tot} = {hit / tot:.0%} · khung thừa {fa} · "
              + " ".join(f"{k} {a}/{n}" for k, (a, n) in by.items()))
    return rows


def sheet(pred, items, thr, out, gts=None, cols=6):
    tiles = []
    for p in items:
        r = pred[str(p)]
        im = Image.open(p).convert("RGB"); d = ImageDraw.Draw(im); lw = max(2, im.width // 250)
        for g in (gts or {}).get(Path(p).name, []):
            d.rectangle(g["bbox"], outline=(255, 255, 255), width=lw)
        for b, s in zip(r["boxes"], r["scores"]):
            if s >= thr:
                d.rectangle(b, outline=(255, 0, 255), width=lw); d.text((b[0] + lw, b[1] + lw), f"{s:.2f}", fill=(255, 0, 255))
        im.thumbnail((320, 320)); t = Image.new("RGB", (320, 340), "white"); t.paste(im, (0, 20))
        ImageDraw.Draw(t).text((3, 3), Path(p).name[:40], fill=0); tiles.append(np.asarray(t))
    while len(tiles) % cols:
        tiles.append(np.full((340, 320, 3), 255, np.uint8))
    Image.fromarray(np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])).save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(WORK / "rfdetr_real_out" / "checkpoint_best_total.pth"))
    ap.add_argument("--model", choices=["nano", "small"], default="small")
    a = ap.parse_args()
    from rfdetr import RFDETRNano, RFDETRSmall
    model = (RFDETRSmall if a.model == "small" else RFDETRNano)(pretrain_weights=a.ckpt)

    sel = json.loads((ROOT / "annotations/store_selection.json").read_text(encoding="utf-8"))
    A1 = json.loads((ROOT / "annotations/real_boxes.json").read_text(encoding="utf-8"))["images"]
    A2 = json.loads((ROOT / "annotations/dataset2_boxes.json").read_text(encoding="utf-8"))["images"]
    pv = predict_all(model, sel["val"])
    mx = np.array([max(r["scores"], default=0.0) for r in pv.values()])
    thr = float(min(max(np.quantile(mx, 0.90), 0.3), 0.9))
    print(f"Ngưỡng gợi ý (shop val, 60 ảnh sạch, ~10% ảnh có khung): {thr:.2f}")

    p1 = predict_all(model, [ROOT / "dataset" / n for n in A1])
    p2 = predict_all(model, [ROOT / "dataset2" / n for n in A2])
    pt = predict_all(model, sel["test"])
    print("dataset (24 ảnh giày, 89 khung):"); eval_annotated(p1, ROOT / "dataset", A1)
    print("dataset2 (20 ảnh áo, 75 khung):"); eval_annotated(p2, ROOT / "dataset2", A2)
    print("Shop test (150 ảnh sạch):")
    for t in THRS + (round(thr, 2),):
        n_img = sum(any(s >= t for s in r["scores"]) for r in pt.values())
        n_box = sum(sum(s >= t for s in r["scores"]) for r in pt.values())
        print(f"  ngưỡng {t}: {n_img}/150 ảnh có khung · {n_box} khung thừa")
    sheet(p1, [ROOT / "dataset" / n for n in A1], thr, WORK / "rfdetr_eval_dataset.jpg", A1)
    sheet(p2, [ROOT / "dataset2" / n for n in A2], thr, WORK / "rfdetr_eval_dataset2.jpg", A2, cols=5)
    worst = sorted(pt, key=lambda p: -max(pt[p]["scores"], default=0))[:12]
    sheet(pt, worst, thr, WORK / "rfdetr_eval_shop.jpg")
    (WORK / "rfdetr_preds.json").write_text(json.dumps({"val": pv, "dataset": p1, "dataset2": p2, "test": pt, "thr": thr}), encoding="utf-8")
    print("tờ ảnh: work/rfdetr_eval_{dataset,dataset2,shop}.jpg · dự đoán: work/rfdetr_preds.json")


if __name__ == "__main__":
    main()
