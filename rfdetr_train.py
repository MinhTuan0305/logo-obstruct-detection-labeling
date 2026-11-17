"""Fine-tune RF-DETR (Apache-2.0, backbone DINOv2) phát hiện vùng che.

Dữ liệu:
- tổng hợp: work/synth/*.jpg + work/synth_{train,val}.json (notebook obstruct_detection.ipynb, mục 3);
- (tùy chọn) ảnh THẬT đã gán nhãn: <real>/coco_{train,val}.json (chuan_bi_du_lieu_that.ipynb, bước 5).
Một lớp: "obstruction". Ảnh không khung (ảnh sạch) được dùng làm mẫu âm.

Khi có --real: train = tổng hợp + ảnh thật train lặp lại --real-repeat lần (để ~3.000 ảnh tổng hợp không lấn át);
valid = ảnh thật val → dừng sớm và chọn checkpoint theo ẢNH THẬT, không theo mAP tổng hợp.

Chạy (PowerShell, trong D:\\Code\\logo-obstruct-detection):
  .venv\\Scripts\\python.exe -X utf8 rfdetr_train.py --bench                       # đo tốc độ, ước tính thời gian
  .venv\\Scripts\\python.exe -X utf8 rfdetr_train.py --epochs 20                   # chỉ dữ liệu tổng hợp → work/rfdetr_out/
  .venv\\Scripts\\python.exe -X utf8 rfdetr_train.py --real data_real_train --init work/rfdetr_out/checkpoint_best_total.pth
                                                                                   # + ảnh thật, train tiếp từ model hiện có → work/rfdetr_real_out/
  ... --resume <output>/checkpoint.pth                                             # chạy tiếp lần train bị dừng
Sau đó đo trên ảnh thật: rfdetr_eval.py --ckpt <output>/checkpoint_best_total.pth
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
SYN = WORK / "synth"


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def _append(dst, src, src_dir, d, prefix, repeat=1, n=None):
    """Thêm ảnh + khung của src (COCO) vào dst, hardlink ảnh vào thư mục d (tên mới để không trùng)."""
    ims = src["images"][:n] if n else src["images"]
    anns = defaultdict_list(src["annotations"])
    for rep in range(repeat):
        for im in ims:
            iid = len(dst["images"])
            name = f"{prefix}{rep}_{im['id']}{Path(im['file_name']).suffix}" if prefix else Path(im["file_name"]).name
            _link(Path(src_dir) / im["file_name"], d / name)
            dst["images"].append({"id": iid, "file_name": name, "width": im["width"], "height": im["height"]})
            for a in anns.get(im["id"], []):
                dst["annotations"].append({**a, "id": len(dst["annotations"]), "image_id": iid, "category_id": 0})


def defaultdict_list(anns):
    out = {}
    for a in anns:
        out.setdefault(a["image_id"], []).append(a)
    return out


def build_dataset(out: Path, n_train=None, n_val=None, real=None, real_repeat=5):
    """out/{train,valid,test}/ với hardlink tới ảnh (không tốn dung lượng)."""
    if out.exists():
        shutil.rmtree(out)
    cat = [{"id": 0, "name": "obstruction", "supercategory": "none"}]
    syn_tr = json.loads((WORK / "synth_train.json").read_text(encoding="utf-8"))
    syn_va = json.loads((WORK / "synth_val.json").read_text(encoding="utf-8"))
    r_tr = r_va = None
    if real:
        real = Path(real)
        r_tr = json.loads((real / "coco_train.json").read_text(encoding="utf-8"))
        r_va = json.loads((real / "coco_val.json").read_text(encoding="utf-8"))
    for split in ("train", "valid", "test"):
        d = out / split; d.mkdir(parents=True)
        coco = {"images": [], "annotations": [], "categories": cat}
        if split == "train":
            _append(coco, syn_tr, SYN, d, "", n=n_train)
            if r_tr:
                _append(coco, r_tr, real, d, "real_tr", repeat=real_repeat)
        elif split == "valid" and r_va and r_va["images"]:
            _append(coco, r_va, real, d, "real_va")
        else:
            _append(coco, syn_va, SYN, d, "", n=n_val)
        (d / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
        print(f"{split}: {len(coco['images'])} ảnh, {len(coco['annotations'])} khung")
    if r_tr:
        print(f"(ảnh thật: train {len(r_tr['images'])} × {real_repeat} lần, val {len(r_va['images'])} ảnh dùng để dừng sớm / chọn checkpoint)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["nano", "small"], default="small")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="batch hiệu dụng = batch × accum")
    ap.add_argument("--lr", type=float, default=None, help="mặc định 1e-4; 5e-5 khi có --init")
    ap.add_argument("--workers", type=int, default=2, help="số tiến trình nạp dữ liệu (nút thắt chính trên Windows)")
    ap.add_argument("--real", default=None, help="thư mục ảnh thật có coco_train.json / coco_val.json (vd data_real_train)")
    ap.add_argument("--real-repeat", type=int, default=5, help="số lần lặp ảnh thật train trong mỗi epoch")
    ap.add_argument("--init", default=None, help="checkpoint để train tiếp (vd work/rfdetr_out/checkpoint_best_total.pth)")
    ap.add_argument("--out", default=None, help="thư mục kết quả (mặc định work/rfdetr_out, hoặc work/rfdetr_real_out khi có --real)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--bench", action="store_true", help="1 epoch trên 160 ảnh train + 40 ảnh val để ước tính thời gian")
    a = ap.parse_args()

    from rfdetr import RFDETRNano, RFDETRSmall

    bench = a.bench
    ds = build_dataset(WORK / ("rfdetr_bench_ds" if bench else "rfdetr_ds"), 160 if bench else None, 40 if bench else None,
                       real=None if bench else a.real, real_repeat=a.real_repeat)
    out = Path(a.out) if a.out else WORK / ("rfdetr_bench_out" if bench else "rfdetr_real_out" if a.real else "rfdetr_out")
    lr = a.lr or (5e-5 if a.init else 1e-4)
    init = str(Path(a.init).resolve()) if a.init else None
    os.chdir(WORK)  # trọng số pretrained (rf-detr-*.pth) tải về work/
    Cls = RFDETRSmall if a.model == "small" else RFDETRNano
    model = Cls(pretrain_weights=init) if init else Cls()
    t0 = time.time()
    model.train(dataset_dir=str(ds), output_dir=str(out), epochs=1 if bench else a.epochs, batch_size=a.batch,
                grad_accum_steps=a.accum, lr=lr, num_workers=a.workers, tensorboard=False, run_test=False,
                checkpoint_interval=5, early_stopping=not bench, early_stopping_patience=5, resume=a.resume)
    dt = time.time() - t0
    if bench:
        n_tr = len(json.loads((WORK / "synth_train.json").read_text(encoding="utf-8"))["images"])
        n_va = len(json.loads((WORK / "synth_val.json").read_text(encoding="utf-8"))["images"])
        per_img = dt / (160 + 40 * 0.35)   # ước lượng: đánh giá val nhanh hơn train ~3 lần
        ep = per_img * (n_tr + n_va * 0.35)
        print(f"\n[bench] 1 epoch nhỏ (160 train + 40 val): {dt:.0f} s → ~{per_img:.2f} s/ảnh")
        print(f"[bench] ước tính 1 epoch đầy đủ ({n_tr} train + {n_va} val): {ep / 60:.0f} phút "
              "(lần train thật trước: ~7 phút/epoch, vì lượt đo nhỏ bị chi phí khởi động lấn át)")
        for e in (10, 20, 30):
            print(f"[bench] {e} epoch: ~{e * ep / 3600:.1f} giờ")
    else:
        print(f"xong sau {dt / 3600:.2f} giờ → {out}")


if __name__ == "__main__":
    main()
