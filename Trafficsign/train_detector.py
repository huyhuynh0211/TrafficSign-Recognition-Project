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
                   help="Tắt Automatic Mixed Precision")
    p.add_argument("--no-cache",     action="store_true",
                   help="Tắt cache dataset vào RAM")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.data):
        sys.exit(
            f"Error '{args.data}'\n"
            "run prepare_gtsdb.py first."
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

    model = YOLO("yolov8n.pt")
    model.train(
        data         = args.data,
        epochs       = args.epochs,
        imgsz        = args.imgsz,
        batch        = args.batch,
        device       = device,
        workers      = args.workers,

        amp          = amp,        
        cache        = cache,      
        close_mosaic = 5,         
        save_period  = 5, 

        patience     = 10,

        project      = "detector_train",
        name         = "traffic_sign",
        exist_ok     = True,

        hsv_h        = 0.015,
        hsv_s        = 0.5,
        hsv_v        = 0.4,
        degrees      = 8,
        translate    = 0.1,
        scale        = 0.4,
        fliplr       = 0.0,     
        mosaic       = 1.0,
        copy_paste   = 0.1,
        erasing      = 0.3,
    )

    best_path = None
    if hasattr(model, "trainer") and model.trainer is not None:
        candidate = str(model.trainer.best)
        if os.path.isfile(candidate):
            best_path = candidate

    if best_path is None:
        fallback = os.path.join("detector_train", "traffic_sign", "weights", "best.pt")
        if os.path.isfile(fallback):
            best_path = fallback

    if best_path:
        shutil.copy(best_path, args.out)
        print(f"\n[Done] Model đã lưu → {args.out}")
        print(f"       (Nguồn: {best_path})")
    else:
        found = []
        for root, _, files in os.walk("detector_train"):
            for f in files:
                if f == "best.pt":
                    found.append(os.path.join(root, f))
        if found:
            newest = max(found, key=os.path.getmtime)
            shutil.copy(newest, args.out)
            print(f"\n[Done] Model đã lưu → {args.out}  (tìm thấy tại: {newest})")
        else:
            print(f"\n[Cảnh báo] Không tìm thấy best.pt!")
            print(f"           Tìm thủ công tại: detector_train/")


if __name__ == "__main__":
    main()