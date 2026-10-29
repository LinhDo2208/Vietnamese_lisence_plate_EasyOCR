"""Phat hien bien so bang YOLOv8 (best.pt). Chi lo viec tim khung, khong doc chu."""
import torch

# Va cho PyTorch >= 2.6 khi dung ultralytics ban cu (checkpoint do chinh ban train)
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO  # noqa: E402


class PlateDetector:
    def __init__(self, weights="best.pt", imgsz=640, conf=0.3):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, frame):
        """Tra ve list (x1, y1, x2, y2, conf), toa do da cat trong khung anh."""
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        r = results[0] if isinstance(results, (list, tuple)) else results
        h, w = frame.shape[:2]
        boxes = []
        if r.boxes is None:
            return boxes
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        return boxes


def cat_bien(frame, box, pad=5, min_size=20):
    """Cat bien so tu frame (them pad pixel quanh khung). Tra ve None neu qua nho."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box[:4]
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    if (cx2 - cx1) < min_size or (cy2 - cy1) < min_size:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()