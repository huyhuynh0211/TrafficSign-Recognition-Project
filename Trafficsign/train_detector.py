# ============================================================
#  train_detector.py
#  Fine-tune YOLOv8n — 1 class "traffic-sign"
#  Chạy sau khi đã có thư mục gtsdb_yolo/ từ prepare_gtsdb.py
#
#  Cách dùng:
#    python train_detector.py
#    python train_detector.py --epochs 30
#
#  Output: traffic_sign_detector.pt  (ngay trong thư mục Trafficsign/)
# ============================================================

import os, sys, shutil, argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",         default="gtsdb_yolo/dataset.yaml")
    p.add_argument("--epochs",       type=int,   default=25)
    p.add_argument("--imgsz",        type=int,   default=640)
    p.add_argument("--batch",        type=int,   default=16)
    p.add_argument("--workers",      type=int,   default=8)
    p.add_argument("--out",          default="traffic_sign_detector.pt")
    p.add_argument("--no-amp",       action="store_true",
                   help="Tắt Automatic Mixed Precision (dùng khi có lỗi fp16)")
    p.add_argument("--no-cache",     action="store_true",
                   help="Tắt cache dataset vào RAM")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Kiểm tra dataset ────────────────────────────────────
    if not os.path.isfile(args.data):
        sys.exit(
            f"[Lỗi] Không tìm thấy '{args.data}'\n"
            "Hãy chạy prepare_gtsdb.py trước."
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[Lỗi] pip install ultralytics")

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    amp    = (not args.no_amp) and torch.cuda.is_available()
    cache  = "ram" if not args.no_cache else False

    print(f"[Train] Device : {'GPU cuda:0' if device == 0 else 'CPU'}")
    print(f"[Train] AMP    : {'ON (fp16)' if amp else 'OFF'}")
    print(f"[Train] Cache  : {cache if cache else 'OFF'}")
    print(f"[Train] Dataset: {args.data}")
    print(f"[Train] Epochs : {args.epochs}  |  ImgSz: {args.imgsz}  |  Batch: {args.batch}")
    print(f"[Train] Workers: {args.workers}")
    print(f"[Train] Ước tính: ~5–10 phút trên RTX 5070 Ti\n")

    # ── Train ───────────────────────────────────────────────
    model = YOLO("yolov8n.pt")
    model.train(
        data         = args.data,
        epochs       = args.epochs,
        imgsz        = args.imgsz,
        batch        = args.batch,
        device       = device,
        workers      = args.workers,

        # ── Tối ưu GPU ───────────────────────────────────
        amp          = amp,        # Automatic Mixed Precision → ~2x nhanh hơn
        cache        = cache,      # Cache toàn dataset vào RAM (dataset nhỏ ~600 ảnh)
        close_mosaic = 5,          # Tắt mosaic 5 epoch cuối để ổn định convergence
        save_period  = 5,          # Lưu checkpoint mỗi 5 epoch

        # ── Stopping ─────────────────────────────────────
        patience     = 10,

        # ── Project / name ───────────────────────────────
        project      = "detector_train",
        name         = "traffic_sign",
        exist_ok     = True,

        # ── Augmentation phù hợp biển báo ────────────────
        hsv_h        = 0.015,
        hsv_s        = 0.5,
        hsv_v        = 0.4,
        degrees      = 8,
        translate    = 0.1,
        scale        = 0.4,
        fliplr       = 0.0,        # TẮT flip ngang (biển báo có hướng)
        mosaic       = 1.0,
        copy_paste   = 0.1,        # Copy-paste augmentation
        erasing      = 0.3,        # Random erasing để tránh over-fit background
    )

    # ── [FIX] Lấy đường dẫn best.pt từ trainer (không hardcode) ──
    # YOLO lưu vào runs/detect/ hoặc <project>/<name>/ tùy version
    # model.trainer.best là Path object chắc chắn trỏ đúng file
    best_path = None
    if hasattr(model, "trainer") and model.trainer is not None:
        candidate = str(model.trainer.best)
        if os.path.isfile(candidate):
            best_path = candidate

    # Fallback: tìm trong project/name/weights/
    if best_path is None:
        fallback = os.path.join("detector_train", "traffic_sign", "weights", "best.pt")
        if os.path.isfile(fallback):
            best_path = fallback

    if best_path:
        shutil.copy(best_path, args.out)
        print(f"\n[Done] Model đã lưu → {args.out}")
        print(f"       (Nguồn: {best_path})")
        print(f"       yolo_worker.py sẽ tự dùng nó thay vì yolov8n.pt")
    else:
        # Thử tìm trong toàn bộ detector_train/
        found = []
        for root, _, files in os.walk("detector_train"):
            for f in files:
                if f == "best.pt":
                    found.append(os.path.join(root, f))
        if found:
            # Lấy file mới nhất
            newest = max(found, key=os.path.getmtime)
            shutil.copy(newest, args.out)
            print(f"\n[Done] Model đã lưu → {args.out}  (tìm thấy tại: {newest})")
        else:
            print(f"\n[Cảnh báo] Không tìm thấy best.pt!")
            print(f"           Tìm thủ công tại: detector_train/")


if __name__ == "__main__":
    main()