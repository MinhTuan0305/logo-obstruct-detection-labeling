"""Chọn ảnh shop chính hãng (Adidas / Nike) từ work/store_scores.json → annotations/store_selection.json.

- Chia theo CỤM MÃ SẢN PHẨM (mã ảnh // 1000): các ảnh của cùng một sản phẩm (mã liền nhau) nằm cùng một phía.
- train (300): áo 150, giày 90, phụ kiện 60; ~70% là mẫu âm khó (probe hiện tại ≥ 0.9), ~30% ngẫu nhiên.
- val (60) và test (150): ngẫu nhiên theo loại, không thiên về mẫu khó → ước lượng báo thừa thật.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:\Code\logo-obstruct-detection")
S = json.loads((ROOT / "work/store_scores.json").read_text(encoding="utf-8"))
SEED = 0
QUOTA = {"train": {"apparel": 150, "shoe": 90, "accessory": 60},
         "val": {"apparel": 30, "shoe": 20, "accessory": 10},
         "test": {"apparel": 75, "shoe": 50, "accessory": 25}}
HARD_FRAC, HARD_T = 0.7, 0.9


def group(p):
    stem = Path(p).stem
    return stem[:-3] if stem.isdigit() else stem   # mã ảnh // 1000


groups = defaultdict(list)
for p in S:
    groups[group(p)].append(p)
keys = sorted(groups)
random.Random(SEED).shuffle(keys)
# 70% cụm cho train, 10% val, 20% test
n = len(keys)
side = {k: ("train" if i < 0.7 * n else "val" if i < 0.8 * n else "test") for i, k in enumerate(keys)}

rng = random.Random(SEED)
sel = {"train": [], "val": [], "test": []}
for split, q in QUOTA.items():
    for cat, k in q.items():
        pool = [p for p in S if side[group(p)] == split and S[p]["cat"] == cat]
        rng.shuffle(pool)
        if split == "train":
            hard = [p for p in pool if S[p]["probe_max"] >= HARD_T]
            easy = [p for p in pool if S[p]["probe_max"] < HARD_T]
            nh = min(len(hard), round(k * HARD_FRAC))
            pick = hard[:nh] + easy[:k - nh]
            pick += [p for p in pool if p not in pick][:k - len(pick)]
        else:
            pick = pool[:k]
        sel[split] += sorted(pick)
        print(f"{split:5s} {cat:9s} {len(pick):3d} (cụm có {len(pool)} ảnh)"
              + (f" · mẫu khó {sum(S[p]['probe_max'] >= HARD_T for p in pick)}" if split == "train" else ""))
assert not (set(sel["train"]) & set(sel["test"])) and not (set(sel["train"]) & set(sel["val"]))
sel["_note"] = ("Ảnh Adidas / Nike official store, không che. Bỏ banner 1000×1000 / 1200×1200 và ảnh đã có trong poc lazmall. "
                "Chia theo cụm mã sản phẩm (mã // 1000). Sinh bằng select_store_images.py (seed 0).")
(ROOT / "annotations/store_selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
print({k: len(v) for k, v in sel.items() if k != "_note"})
