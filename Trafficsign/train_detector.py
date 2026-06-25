# ============================================================
#  train_detector.py
#  Fine-tune YOLOv8n — 1 class "traffic-sign"
#  Chạy sau khi đã có thư mục gtsdb_yolo/ từ prepare_gtsdb.py
#
#  Cách dùng:
#    python train_detector.py
#    python train_detector.py --epochs 30
#
#  Output: traffic_sign_detector.pt
#  → Copy file này vào thư mục Trafficsign/ là xong
# ============================================================

import os, sys, shutil, argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="gtsdb_yolo/dataset.yaml")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--out",    default="traffic_sign_detector.pt")
    return p.parse_args()

def main():
    args = parse_args()

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
    print(f"[Train] Device: {'GPU cuda:0' if device == 0 else 'CPU'}")
    print(f"[Train] Dataset: {args.data}")
    print(f"[Train] Epochs: {args.epochs} | ImgSz: {args.imgsz} | Batch: {args.batch}")
    print(f"[Train] Ước tính: ~5–10 phút trên RTX 5070 Ti\n")

    model = YOLO("yolov8n.pt")
    model.train(
        data      = args.data,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = device,
        patience  = 10,
        project   = "detector_train",
        name      = "traffic_sign",
        exist_ok  = True,
        # Augmentation phù hợp biển báo
        hsv_h     = 0.015,
        hsv_s     = 0.5,
        hsv_v     = 0.4,
        degrees   = 8,
        translate = 0.1,
        scale     = 0.4,
        fliplr    = 0.0,   # tắt flip ngang (biển báo có hướng)
        mosaic    = 1.0,
    )

    best = os.path.join("detector_train", "traffic_sign", "weights", "best.pt")
    if os.path.isfile(best):
        shutil.copy(best, args.out)
        print(f"\n[Done] Model đã lưu → {args.out}")
        print(f"       Copy file này vào thư mục Trafficsign/ là xong!")
        print(f"       yolo_worker.py sẽ tự dùng nó thay vì yolov8n.pt")
    else:
        print(f"\n[Cảnh báo] Không tìm thấy best.pt, kiểm tra: detector_train/traffic_sign/weights/")

if __name__ == "__main__":
    main()