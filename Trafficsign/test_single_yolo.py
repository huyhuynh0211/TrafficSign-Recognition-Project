# ============================================================
#  test_single_yolo.py
#  Dùng YOLOv8n để detect vùng biển báo trong ảnh,
#  sau đó crop từng vùng và phân loại bằng model Keras.
#
#  Cài đặt:
#    pip install ultralytics tensorflow opencv-python
#
#  Cách dùng:
#    python test_single_yolo.py --model best_model.keras --image path/to/image.png
#    python test_single_yolo.py --model best_model.keras --image path/to/image.png --conf 0.3
#    python test_single_yolo.py --model best_model.keras --image path/to/image.png --save
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import sys
import cv2
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# ============================================================
# GTSRB — 43 nhãn biển báo tiếng Việt
# ============================================================
SIGNS = {
    0:  "Giới hạn tốc độ 20 km/h",
    1:  "Giới hạn tốc độ 30 km/h",
    2:  "Giới hạn tốc độ 50 km/h",
    3:  "Giới hạn tốc độ 60 km/h",
    4:  "Giới hạn tốc độ 70 km/h",
    5:  "Giới hạn tốc độ 80 km/h",
    6:  "Hết giới hạn tốc độ 80 km/h",
    7:  "Giới hạn tốc độ 100 km/h",
    8:  "Giới hạn tốc độ 120 km/h",
    9:  "Cấm vượt",
    10: "Cấm xe trên 3.5 tấn vượt",
    11: "Đường ưu tiên tại giao lộ tiếp theo",
    12: "Đường ưu tiên",
    13: "Nhường đường",
    14: "Dừng lại",
    15: "Cấm xe",
    16: "Cấm xe trên 3.5 tấn",
    17: "Cấm đi vào",
    18: "Nguy hiểm chung",
    19: "Đường cong nguy hiểm bên trái",
    20: "Đường cong nguy hiểm bên phải",
    21: "Đường cong liên tiếp",
    22: "Đường gồ ghề",
    23: "Đường trơn trượt",
    24: "Đường hẹp bên phải",
    25: "Công trường",
    26: "Đèn giao thông",
    27: "Người đi bộ",
    28: "Trẻ em qua đường",
    29: "Xe đạp băng qua",
    30: "Cẩn thận băng/tuyết",
    31: "Động vật hoang dã băng qua",
    32: "Hết tất cả giới hạn tốc độ và cấm vượt",
    33: "Rẽ phải phía trước",
    34: "Rẽ trái phía trước",
    35: "Chỉ được đi thẳng",
    36: "Đi thẳng hoặc rẽ phải",
    37: "Đi thẳng hoặc rẽ trái",
    38: "Đi bên phải",
    39: "Đi bên trái",
    40: "Bắt buộc đi vòng xuyến",
    41: "Hết cấm vượt",
    42: "Hết cấm vượt đối với xe trên 3.5 tấn",
}

DEFAULT_SIZE = 30  # kích thước input mặc định nếu model không khai báo

# Màu vẽ bounding box theo confidence: cao → xanh lá, thấp → đỏ
COLOR_HIGH   = (0, 200, 0)
COLOR_MEDIUM = (0, 165, 255)
COLOR_LOW    = (0, 0, 220)

# ============================================================
# HELPERS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8n detect → Keras classify — test ảnh đơn"
    )
    parser.add_argument("--model",  required=True, help="Đường dẫn file .keras / .h5 của model phân loại")
    parser.add_argument("--image",  required=True, help="Đường dẫn ảnh cần test")
    parser.add_argument("--conf",   type=float, default=0.25,
                        help="Ngưỡng confidence của YOLO (mặc định: 0.25)")
    parser.add_argument("--iou",    type=float, default=0.45,
                        help="Ngưỡng IoU NMS của YOLO (mặc định: 0.45)")
    parser.add_argument("--top3",   action="store_true",
                        help="Hiển thị top-3 dự đoán thay vì chỉ top-1")
    parser.add_argument("--save",   action="store_true",
                        help="Lưu ảnh kết quả có vẽ bounding box")
    parser.add_argument("--out",    default="output_single.jpg",
                        help="Tên file ảnh kết quả (mặc định: output_single.jpg)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", default=True,
                        help="Tắt chuẩn hoá [0,1] (dùng nếu model train trên pixel [0,255])")
    return parser.parse_args()


def get_input_size(model):
    """Đọc (height, width) từ input shape của model; fallback về DEFAULT_SIZE."""
    try:
        shape = model.input.shape
    except AttributeError:
        shape = model.input_shape

    h = int(shape[1]) if shape[1] is not None else DEFAULT_SIZE
    w = int(shape[2]) if shape[2] is not None else DEFAULT_SIZE

    if shape[1] is None or shape[2] is None:
        print(f"[Cảnh báo] Model có dynamic input shape {tuple(shape)}. "
              f"Dùng {DEFAULT_SIZE}x{DEFAULT_SIZE}.", file=sys.stderr)
    return h, w


def pick_color(conf: float):
    if conf >= 0.70:
        return COLOR_HIGH
    elif conf >= 0.45:
        return COLOR_MEDIUM
    return COLOR_LOW


def draw_result(img_bgr, x1, y1, x2, y2, cls_id, conf_cls, yolo_conf, label):
    """Vẽ bounding box + nhãn lên ảnh BGR."""
    color = pick_color(conf_cls)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    text = f"#{cls_id} {label[:25]} {conf_cls:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    # Nền nhãn
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img_bgr, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
    cv2.putText(img_bgr, text, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # YOLO score nhỏ ở góc dưới box
    yolo_text = f"YOLO:{yolo_conf:.2f}"
    cv2.putText(img_bgr, yolo_text, (x1 + 2, y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # ── 1. Kiểm tra file đầu vào ────────────────────────────
    if not os.path.isfile(args.image):
        sys.exit(f"[Lỗi] Không tìm thấy ảnh: '{args.image}'")
    if not os.path.isfile(args.model):
        sys.exit(f"[Lỗi] Không tìm thấy model: '{args.model}'")

    # ── 2. Load YOLO ─────────────────────────────────────────
    print("Đang tải YOLOv8n …")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[Lỗi] Chưa cài ultralytics. Chạy: pip install ultralytics")

    # Dùng YOLOv8n pretrained trên COCO — detect "stop sign" (class 11 trong COCO)
    # và toàn bộ object để tìm vùng có thể là biển báo.
    # Ta detect tất cả rồi lọc class liên quan đến traffic sign.
    yolo = YOLO("yolov8n.pt")   # tự tải về nếu chưa có

    # Class COCO có liên quan đến giao thông (để lọc bớt nhiễu)
    # 9=traffic light, 11=stop sign; nếu không tìm thấy thì fallback detect all
    COCO_SIGN_CLASSES = {9, 11}

    # ── 3. Load Keras model ──────────────────────────────────
    print(f"Đang tải Keras model: {args.model} …")
    try:
        clf_model = tf.keras.models.load_model(args.model)
    except Exception as e:
        sys.exit(f"[Lỗi] Không load được model: {e}")

    h, w = get_input_size(clf_model)
    print(f"  Input size: {h}×{w}  |  Normalize: {args.normalize}\n")

    # ── 4. Đọc ảnh ──────────────────────────────────────────
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        sys.exit(f"[Lỗi] Không đọc được ảnh: '{args.image}'")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ih, iw = img_bgr.shape[:2]

    # ── 5. YOLO detect ───────────────────────────────────────
    print(f"[YOLO] Đang detect … (conf={args.conf}, iou={args.iou})")
    results = yolo(img_rgb, conf=args.conf, iou=args.iou, verbose=False)
    boxes_all = results[0].boxes

    # Lọc: ưu tiên class traffic sign; nếu không có thì dùng tất cả
    sign_boxes = [b for b in boxes_all if int(b.cls) in COCO_SIGN_CLASSES]
    if not sign_boxes:
        print("  [YOLO] Không tìm thấy traffic sign cụ thể → dùng tất cả detection.")
        sign_boxes = list(boxes_all)

    if not sign_boxes:
        print("  [YOLO] Không phát hiện được bất kỳ object nào trong ảnh.")
        print("  ➜ Gợi ý: hạ ngưỡng --conf (vd: --conf 0.1) hoặc kiểm tra ảnh đầu vào.")
        # Fallback: predict cả ảnh gốc
        print("\n[Fallback] Phân loại toàn bộ ảnh …")
        crop = cv2.resize(img_rgb, (w, h)).astype(np.float32)
        if args.normalize:
            crop /= 255.0
        pred  = clf_model.predict(np.expand_dims(crop, 0), verbose=0)[0]
        top_n = np.argsort(pred)[::-1][:3 if args.top3 else 1]
        print(f"\n{'─'*55}")
        print(f"  Ảnh gốc (fallback, không có YOLO box)")
        for rank, cid in enumerate(top_n, 1):
            bar = "█" * int(pred[cid] * 20)
            print(f"  #{rank}  [{cid:2d}] {SIGNS.get(cid,'Unknown')}")
            print(f"       Confidence: {pred[cid]:.2%}  {bar}")
        return

    print(f"  [YOLO] Phát hiện {len(sign_boxes)} object(s).\n")

    # ── 6. Crop → Classify từng box ──────────────────────────
    print("=" * 65)
    print(f"  {'Box':<5} {'Vùng detect':<25} {'Nhãn dự đoán':<30} {'Conf':>6}")
    print("─" * 65)

    result_img = img_bgr.copy()

    for idx, box in enumerate(sign_boxes):
        # Toạ độ pixel (xyxy)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Clamp vào kích thước ảnh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        yolo_conf = float(box.conf[0])

        crop_rgb = img_rgb[y1:y2, x1:x2]
        if crop_rgb.size == 0:
            print(f"  Box {idx+1:<3} | (vùng trống, bỏ qua)")
            continue

        # Preprocess crop
        crop_resized = cv2.resize(crop_rgb, (w, h)).astype(np.float32)
        if args.normalize:
            crop_resized /= 255.0
        inp = np.expand_dims(crop_resized, 0)

        # Predict
        pred   = clf_model.predict(inp, verbose=0)[0]
        top_n  = np.argsort(pred)[::-1][:3 if args.top3 else 1]
        cls_id = int(top_n[0])
        conf   = float(pred[cls_id])
        label  = SIGNS.get(cls_id, "Unknown")

        region_str = f"({x1},{y1})→({x2},{y2})"
        print(f"  Box {idx+1:<3} | {region_str:<25} | [{cls_id:2d}] {label[:28]:<30} | {conf:.1%}")

        if args.top3 and len(top_n) > 1:
            for rank, cid in enumerate(top_n[1:], 2):
                bar = "░" * int(pred[cid] * 20)
                print(f"         | {'':25} | #{rank} [{cid:2d}] {SIGNS.get(cid,'?')[:28]:<28} | {pred[cid]:.1%} {bar}")

        # Vẽ lên ảnh
        draw_result(result_img, x1, y1, x2, y2, cls_id, conf, yolo_conf, label)

    print("=" * 65)

    # ── 7. Lưu kết quả ──────────────────────────────────────
    if args.save:
        cv2.imwrite(args.out, result_img)
        print(f"\n[Saved] Ảnh kết quả → {args.out}")
    else:
        print("\n(Thêm --save để lưu ảnh kết quả có bounding box)")

    # ── 8. Hiển thị ảnh (nếu có GUI) ─────────────────────────
    try:
        cv2.imshow("Traffic Sign Detection + Classification", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass   # môi trường headless (server/Colab) không cần show


if __name__ == "__main__":
    main()