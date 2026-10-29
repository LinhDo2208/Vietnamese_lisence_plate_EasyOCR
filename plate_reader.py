"""Doc bien so bang EasyOCR (model 'plate' da cai bang setup_model.py) + theo doi/bo phieu."""
import os
import re
from collections import Counter

import cv2
import torch

CHARS_MAC_DINH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # dung khi khong doc duoc plate.yaml
MAU_TREN = re.compile(r"^\d{2}[A-Z][A-Z0-9]?$")          # vi du 59G1
MAU_DUOI = re.compile(r"^\d{4,5}$")                       # vi du 63188
MAU_1DONG = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{4,5}$")  # bien 1 dong
MAU_DAYDU = MAU_1DONG                                     # ket qua cuoi (2 dong da ghep)
# cac muc cat le (ngang, doc) theo ti le anh, de loai vien bien va nhieu o mep
LE = [(0.0, 0.0), (0.04, 0.04), (0.08, 0.06), (0.12, 0.08)]


def doc_bo_ky_tu(model):
    # doc bo ky tu tu yaml cua model de luon khop voi luc train
    p = os.path.join(os.path.expanduser("~"), ".EasyOCR", "user_network", f"{model}.yaml")
    try:
        import yaml
        with open(p, encoding="utf-8") as f:
            return str(yaml.safe_load(f)["character_list"])
    except Exception:
        return CHARS_MAC_DINH


class PlateOCR:
    def __init__(self, passes=3, gpu=True, model="plate"):
        import easyocr
        # detector=False: chi dung bo nhan dang, khong can tai model CRAFT
        self.reader = easyocr.Reader(["en"], gpu=gpu and torch.cuda.is_available(),
                                     recog_network=model, detector=False)
        self.chars = doc_bo_ky_tu(model)  # doc tat ca ky tu da train
        self.le = LE[: max(1, min(passes, len(LE)))]

    def _mot_lan(self, img, mx, my):
        h, w = img.shape[:2]
        c = img[int(h * my): max(int(h * (1 - my)), int(h * my) + 4),
                int(w * mx): max(int(w * (1 - mx)), int(w * mx) + 4)]
        # chi chuyen xam, khong tien xu ly them de khop cach train
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
        hh, ww = g.shape[:2]
        kq = self.reader.recognize(g, horizontal_list=[[0, ww, 0, hh]], free_list=[],
                                   allowlist=self.chars)
        # chi giu ky tu nam trong bo ky tu da train
        return "".join(ch for ch in "".join(t for _, t, _ in kq).upper() if ch in self.chars)

    def _dong(self, img, mau):
        ds = [self._mot_lan(img, mx, my) for mx, my in self.le]
        hop = [d for d in ds if mau.match(d)]  # uu tien ket qua dung dinh dang
        pool = hop or [d for d in ds if d]
        return Counter(pool).most_common(1)[0][0] if pool else ""

    def read(self, crop):
        """Nhan anh crop bien so (BGR), tra ve chuoi bien so (khong dau cach, khong gach)."""
        h, w = crop.shape[:2]
        if w / h < 2.5:  # bien 2 dong: cat doi, doc rieng tung dong
            return self._dong(crop[: h // 2], MAU_TREN) + self._dong(crop[h // 2:], MAU_DUOI)
        return self._dong(crop, MAU_1DONG)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    """Ghep khung giua cac frame bang IoU, moi khung giu cac ket qua doc de bo phieu."""
    def __init__(self, max_age=15, iou_thr=0.3, max_votes=10):
        self.tracks, self.max_age, self.iou_thr, self.max_votes = [], max_age, iou_thr, max_votes

    def step(self):
        # goi 1 lan moi frame: tang tuoi, bo khung da mat qua lau
        for t in self.tracks:
            t["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

    def match(self, box):
        best, best_iou = None, self.iou_thr
        for t in self.tracks:
            v = iou(t["box"], box)
            if v > best_iou:
                best, best_iou = t, v
        if best is None:
            best = {"box": box, "votes": [], "age": 0}
            self.tracks.append(best)
        best["box"], best["age"] = box, 0
        return best

    def add_vote(self, t, text):
        if text:
            t["votes"] = (t["votes"] + [text])[-self.max_votes:]

    @staticmethod
    def best(t):
        valid = [v for v in t["votes"] if MAU_DAYDU.match(v)]
        pool = valid or t["votes"]
        return Counter(pool).most_common(1)[0][0] if pool else ""