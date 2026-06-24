# ============================================================
#  test_single_yolo.py  (v2 — fix segfault)
#  Pipeline: YOLOv8n detect (subprocess) → crop → Keras classify
#
#  Fix segfault: YOLO và TensorFlow chạy trong 2 process riêng.
#  Yêu cầu: yolo_worker.py phải nằm cùng thư mục với file này.
#
#  Cài đặt:  pip install ultralytics tensorflow opencv-python
#
#  Cách dùng:
#    python test_single_yolo.py --model best_model.keras --image Test_5/45.png
#    python test_single_yolo.py --model best_model.keras --image Test_5/45.png --save
#    python test_single_yolo.py --model best_model.keras --image Test_5/45.png --conf 0.15 --top3
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import subprocess
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

DEFAULT_SIZE = 30

COLOR_HIGH   = (0, 200, 0)
COLOR_MEDIUM = (0, 165, 255)
COLOR_LOW    = (0, 0, 220)

# ============================================================
# HELPERS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8n detect (subprocess) → Keras classify — test ảnh đơn"
    )
    parser.add_argument("--model",  required=True, help="File .keras / .h5 của model phân loại")
    parser.add_argument("--image",  required=True, help="Đường dẫn ảnh cần test")
    parser.add_argument("--conf",   type=float, default=0.25, help="Ngưỡng confidence YOLO (mặc định 0.25)")
    parser.add_argument("--iou",    type=float, default=0.45, help="Ngưỡng IoU NMS YOLO (mặc định 0.45)")
    parser.add_argument("--top3",   action="store_true",      help="Hiển thị top-3 dự đoán")
    parser.add_argument("--save",   action="store_true",      help="Lưu ảnh kết quả có vẽ bounding box")
    parser.add_argument("--out",    default="output_single.jpg", help="Tên file ảnh đầu ra")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", default=True,
                        help="Tắt chuẩn hoá [0,1]")
    return parser.parse_args()


def get_input_size(model):
    try:
        shape = model.input.shape
    except AttributeError:
        shape = model.input_shape
    h = int(shape[1]) if shape[1] is not None else DEFAULT_SIZE
    w = int(shape[2]) if shape[2] is not None else DEFAULT_SIZE
    if shape[1] is None or shape[2] is None:
        print(f"[Cảnh báo] Dynamic input shape → dùng {DEFAULT_SIZE}×{DEFAULT_SIZE}", file=sys.stderr)
    return h, w


def run_yolo_worker(image_path, conf, iou):
    """
    Gọi yolo_worker.py trong subprocess riêng (không import TF).
    Trả về list of dict: [{x1,y1,x2,y2,yolo_conf,fallback}, ...]
    """
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_worker.py")
    if not os.path.isfile(worker):
        sys.exit(
            f"[Lỗi] Không tìm thấy yolo_worker.py tại: {worker}\n"
            "Hãy đảm bảo yolo_worker.py nằm cùng thư mục với test_single_yolo.py"
        )

    cmd = [sys.executable, worker,
           "--image", image_path,
           "--conf",  str(conf),
           "--iou",   str(iou)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        sys.exit("[Lỗi] yolo_worker.py timeout sau 120s")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        sys.exit(f"[Lỗi] yolo_worker.py thất bại (exit {result.returncode}):\n{stderr}")

    # Lấy dòng JSON cuối cùng trong stdout (YOLO có thể in thêm log)
    stdout = result.stdout.strip()
    json_line = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") or line.startswith("{"):
            json_line = line
            break

    if json_line is None:
        sys.exit(f"[Lỗi] yolo_worker.py không trả về JSON.\nStdout:\n{stdout}\nStderr:\n{result.stderr}")

    return json.loads(json_line)


def pick_color(conf):
    if conf >= 0.70: return COLOR_HIGH
    if conf >= 0.45: return COLOR_MEDIUM
    return COLOR_LOW


def draw_result(img_bgr, x1, y1, x2, y2, cls_id, conf_cls, yolo_conf, label, fallback=False):
    color = pick_color(conf_cls)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    prefix = "[fallback] " if fallback else ""
    text = f"{prefix}#{cls_id} {label[:22]} {conf_cls:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img_bgr, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
    cv2.putText(img_bgr, text, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if not fallback:
        cv2.putText(img_bgr, f"YOLO:{yolo_conf:.2f}", (x1 + 2, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"[Lỗi] Không tìm thấy ảnh: '{args.image}'")
    if not os.path.isfile(args.model):
        sys.exit(f"[Lỗi] Không tìm thấy model: '{args.model}'")

    # ── 1. YOLO detect (subprocess riêng — không dùng TF) ───
    print(f"[YOLO] Đang detect … (conf={args.conf}, iou={args.iou})")
    detections = run_yolo_worker(args.image, args.conf, args.iou)
    n_det = sum(1 for d in detections if not d["fallback"])
    if n_det:
        print(f"[YOLO] Phát hiện {n_det} object(s).")
    else:
        print("[YOLO] Không detect được object → dùng toàn bộ ảnh (fallback).")

    # ── 2. Load Keras model (trong process chính) ────────────
    print(f"\n[Model] Đang tải: {args.model} …")
    try:
        clf_model = tf.keras.models.load_model(args.model)
    except Exception as e:
        sys.exit(f"[Lỗi] Không load được model: {e}")

    h, w = get_input_size(clf_model)
    print(f"[Model] Input size: {h}×{w}  |  Normalize: {args.normalize}\n")

    # ── 3. Đọc ảnh gốc ──────────────────────────────────────
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result_img = img_bgr.copy()

    # ── 4. Crop → Classify từng box ──────────────────────────
    print("=" * 65)
    print(f"  {'Box':<5} {'Vùng':<25} {'Nhãn dự đoán':<28} {'Conf':>5}")
    print("─" * 65)

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        yolo_conf       = det["yolo_conf"]
        fallback        = det["fallback"]

        crop_rgb = img_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop_rgb, (w, h)).astype(np.float32)
        if args.normalize:
            crop_resized /= 255.0
        inp = np.expand_dims(crop_resized, 0)

        pred   = clf_model.predict(inp, verbose=0)[0]
        top_n  = np.argsort(pred)[::-1][:3 if args.top3 else 1]
        cls_id = int(top_n[0])
        conf   = float(pred[cls_id])
        label  = SIGNS.get(cls_id, "Unknown")

        region = "toàn ảnh (fallback)" if fallback else f"({x1},{y1})→({x2},{y2})"
        print(f"  Box {idx+1:<3} | {region:<25} | [{cls_id:2d}] {label[:26]:<28} | {conf:.1%}")

        if args.top3:
            for rank, cid in enumerate(top_n[1:], 2):
                print(f"  {'':5} | {'  #'+str(rank):<25} | [{cid:2d}] {SIGNS.get(cid,'?')[:26]:<28} | {pred[cid]:.1%}")

        draw_result(result_img, x1, y1, x2, y2, cls_id, conf, yolo_conf, label, fallback)

    print("=" * 65)

    # ── 5. Lưu / hiển thị ────────────────────────────────────
    if args.save:
        cv2.imwrite(args.out, result_img)
        print(f"\n[Saved] → {args.out}")
    else:
        print("\n(Thêm --save để lưu ảnh kết quả có bounding box)")

    try:
        cv2.imshow("Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()