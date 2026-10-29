"""
Nhan dien + doc bien so xe. Goi plate_detector (tim khung) va plate_reader (doc chu).

Vi du:
    python main.py --source anh.jpg --weights best.pt
    python main.py --source video.mp4 --weights runs/detect/train5/weights/best.pt
    python main.py --source video.mp4 --ocr-every 5 --ocr-passes 2   (nhanh hon)
    python main.py --source anh.jpg --no-ocr                         (chi detect)
"""
import matplotlib
matplotlib.use("Agg")  # tranh loi Tk tren Windows

import argparse
from pathlib import Path

import cv2

from plate_detector import PlateDetector, cat_bien  # import truoc de va torch.load
from plate_reader import PlateOCR, Tracker

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="duong dan anh hoac video")
    p.add_argument("--weights", default="best_640.pt", help="duong dan file best.pt")
    p.add_argument("--out", default="output", help="thu muc ket qua")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.3, help="nguong tin cay")
    p.add_argument("--pad", type=int, default=5, help="so pixel nho them quanh bien so khi crop")
    p.add_argument("--min-size", type=int, default=20, help="bo qua crop nho hon (pixel)")
    p.add_argument("--crop-every", type=int, default=1,
                   help="video: chi luu crop moi N frame (1 = moi frame)")
    p.add_argument("--no-ocr", action="store_true", help="tat doc bien, chi detect")
    p.add_argument("--ocr-every", type=int, default=3,
                   help="video: doc bien moi N frame, frame con lai dung ket qua da bo phieu")
    p.add_argument("--ocr-passes", type=int, default=3,
                   help="so muc cat le thu khi doc (1-4), cang nhieu cang chinh xac va cang cham")
    p.add_argument("--cpu", action="store_true", help="ep OCR chay tren CPU")
    return p.parse_args()


def ve_khung(frame, box, label):
    x1, y1, x2, y2 = box[:4]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    if y1 - th - 10 >= 0:  # nhan phia tren khung
        ly1, ly2, ty = y1 - th - 10, y1, y1 - 5
    else:  # sat mep tren thi dua nhan xuong duoi khung
        ly1, ly2, ty = y2, y2 + th + 10, y2 + th + 5
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 4, ly2), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def xu_ly_frame(frame, detector, ocr, tracker, args, crops_dir, tag,
                save_crops=True, do_ocr=True):
    """Detect + doc bien, ve khung kem ket qua len frame (tai cho), luu crop."""
    sach = frame.copy()  # frame goc chua ve khung, de crop sach
    boxes = detector.detect(sach)
    if tracker is not None:
        tracker.step()
    saved = 0
    for i, box in enumerate(boxes):
        crop = cat_bien(sach, box, args.pad, args.min_size)

        # Doc bien: doc khi den luot, hoac khi khung nay chua co ket qua nao
        text = ""
        if ocr is not None:
            t = tracker.match(box[:4])
            if crop is not None and (do_ocr or not t["votes"]):
                tracker.add_vote(t, ocr.read(crop))
            text = tracker.best(t)
            if text and do_ocr:
                print(f"  [{tag}] bien {i}: {text}")

        ve_khung(frame, box, f"{text} {box[4]:.2f}" if text else f"license_plate {box[4]:.2f}")

        if save_crops and crop is not None:
            ten = f"{tag}_plate{i}_{text + '_' if text else ''}conf{box[4]:.2f}.jpg"
            cv2.imwrite(str(crops_dir / ten), crop)
            saved += 1
    return saved


def chay_anh(src, detector, ocr, args, out_dir, crops_dir):
    frame = cv2.imread(str(src))
    if frame is None:
        raise SystemExit(f"Khong doc duoc anh: {src}")
    n = xu_ly_frame(frame, detector, ocr, Tracker() if ocr else None, args, crops_dir, src.stem)
    out_path = out_dir / f"{src.stem}_detected.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"Anh ket qua : {out_path}")
    print(f"So crop luu : {n} -> {crops_dir}")


def chay_video(src, detector, ocr, args, out_dir, crops_dir):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Khong mo duoc video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = out_dir / f"{src.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = Tracker() if ocr else None
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        saved += xu_ly_frame(frame, detector, ocr, tracker, args, crops_dir, f"f{idx:06d}",
                             save_crops=(idx % max(1, args.crop_every) == 0),
                             do_ocr=(idx % max(1, args.ocr_every) == 0))
        writer.write(frame)
        idx += 1
        if idx % 50 == 0:
            print(f"  frame {idx}/{total if total else '?'}  |  crop: {saved}")

    cap.release()
    writer.release()
    print(f"Video ket qua: {out_path}")
    print(f"So frame xu ly: {idx}")
    print(f"So crop luu   : {saved} -> {crops_dir}")


def main():
    args = parse_args()
    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Khong tim thay: {src}")

    out_dir = Path(args.out)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    detector = PlateDetector(args.weights, args.imgsz, args.conf)
    ocr = None if args.no_ocr else PlateOCR(passes=args.ocr_passes, gpu=not args.cpu)

    ext = src.suffix.lower()
    if ext in IMG_EXT:
        chay_anh(src, detector, ocr, args, out_dir, crops_dir)
    elif ext in VID_EXT:
        chay_video(src, detector, ocr, args, out_dir, crops_dir)
    else:
        raise SystemExit(f"Dinh dang khong ho tro: {ext}")


if __name__ == "__main__":
    main()