# ============================================================
#  yolo_worker.py  — chạy riêng, không import TensorFlow
#  Được gọi bởi test_single_yolo.py qua subprocess
#  Trả về JSON: list of {x1,y1,x2,y2,yolo_conf}
# ============================================================
import sys, json, os
os.environ['YOLO_VERBOSE'] = 'False'

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--iou",    type=float, default=0.45)
    args = parser.parse_args()

    from ultralytics import YOLO
    import cv2, numpy as np

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(json.dumps({"error": f"Cannot read: {args.image}"}))
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ih, iw  = img_bgr.shape[:2]

    yolo = YOLO("yolov8n.pt")
    results = yolo(img_rgb, conf=args.conf, iou=args.iou, verbose=False)
    boxes_all = results[0].boxes

    SIGN_CLASSES = {9, 11}   # traffic light, stop sign (COCO)
    sign_boxes = [b for b in boxes_all if int(b.cls) in SIGN_CLASSES]
    if not sign_boxes:
        sign_boxes = list(boxes_all)  # fallback: tất cả object

    out = []
    for b in sign_boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        out.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "yolo_conf": round(float(b.conf[0]), 4),
            "coco_class": int(b.cls[0]),
            "fallback": False,
        })

    if not out:
        # Không detect được gì → fallback cả ảnh
        out.append({
            "x1": 0, "y1": 0, "x2": iw, "y2": ih,
            "yolo_conf": 0.0,
            "coco_class": -1,
            "fallback": True,
        })

    print(json.dumps(out))

if __name__ == "__main__":
    main()