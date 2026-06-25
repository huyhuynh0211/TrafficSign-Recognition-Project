import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import json
import platform
import subprocess
import sys
import cv2
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SIGNS = {
    0:  "Gioi han toc do 20 km/h",
    1:  "Gioi han toc do 30 km/h",
    2:  "Gioi han toc do 50 km/h",
    3:  "Gioi han toc do 60 km/h",
    4:  "Gioi han toc do 70 km/h",
    5:  "Gioi han toc do 80 km/h",
    6:  "Het gioi han toc do 80 km/h",
    7:  "Gioi han toc do 100 km/h",
    8:  "Gioi han toc do 120 km/h",
    9:  "Cam vuot",
    10: "Cam xe tren 3.5 tan vuot",
    11: "Duong uu tien tai giao lo tiep theo",
    12: "Duong uu tien",
    13: "Nhuong duong",
    14: "Dung lai",
    15: "Cam xe",
    16: "Cam xe tren 3.5 tan",
    17: "Cam di vao",
    18: "Nguy hiem chung",
    19: "Duong cong nguy hiem ben trai",
    20: "Duong cong nguy hiem ben phai",
    21: "Duong cong lien tiep",
    22: "Duong go ghe",
    23: "Duong tron truot",
    24: "Duong hep ben phai",
    25: "Cong truong",
    26: "Den giao thong",
    27: "Nguoi di bo",
    28: "Tre em qua duong",
    29: "Xe dap bang qua",
    30: "Can than bang/tuyet",
    31: "Dong vat hoang da bang qua",
    32: "Het tat ca gioi han toc do va cam vuot",
    33: "Re phai phia truoc",
    34: "Re trai phia truoc",
    35: "Chi duoc di thang",
    36: "Di thang hoac re phai",
    37: "Di thang hoac re trai",
    38: "Di ben phai",
    39: "Di ben trai",
    40: "Bat buoc di vong xuyen",
    41: "Het cam vuot",
    42: "Het cam vuot doi voi xe tren 3.5 tan",
}

DEFAULT_SIZE = 30
COLOR_HIGH   = (0, 200, 0)
COLOR_MEDIUM = (0, 165, 255)
COLOR_LOW    = (0, 0, 220)


def _has_display():
    """
    Kiểm tra môi trường có GUI display không.
    Tránh cv2.waitKey(0) treo vô tận khi chạy trên server / SSH không có GUI.
    """
    if platform.system() == "Windows":
        return True
    if platform.system() == "Darwin":
        return True
    return bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8n detect → Keras classify")
    parser.add_argument("--model",  required=True)
    parser.add_argument("--image",  required=True)
    parser.add_argument("--conf",   type=float, default=0.28)
    parser.add_argument("--iou",    type=float, default=0.45)
    parser.add_argument("--save",   action="store_true")
    parser.add_argument("--out",    default="output_single.jpg")

    parser.add_argument("--no-gui", action="store_true",
                        help="Tắt cửa sổ popup ảnh (hữu ích khi chạy SSH/server)")

    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument("--normalize",    dest="normalize", action="store_true", default=True)
    normalize_group.add_argument("--no-normalize", dest="normalize", action="store_false")
    return parser.parse_args()


def get_input_size(model, fallback=DEFAULT_SIZE):
    try:
        shape = model.input.shape
    except AttributeError:
        shape = model.input_shape
    h = int(shape[1] if shape[1] is not None else fallback)
    w = int(shape[2] if shape[2] is not None else fallback)
    return h, w


def run_yolo_worker(image_path, conf, iou):
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_worker.py")
    if not os.path.isfile(worker):
        sys.exit(f"[Lỗi] Không tìm thấy {worker}")
    cmd = [sys.executable, worker,
           "--image", image_path,
           "--conf",  str(conf),
           "--iou",   str(iou)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        sys.exit(f"[Lỗi] yolo_worker thất bại:\n{res.stderr.strip()}")

    for line in reversed(res.stdout.strip().splitlines()):
        stripped = line.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            return json.loads(stripped)
    sys.exit("[Lỗi] Không lấy được JSON từ YOLO.")


def draw_result(img_bgr, x1, y1, x2, y2, cls_id, conf_cls, yolo_conf, label, fallback=False):
    color = (COLOR_HIGH   if conf_cls >= 0.7  else
             COLOR_MEDIUM if conf_cls >= 0.45 else
             COLOR_LOW)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    text = f"{'[fb] ' if fallback else ''}#{cls_id} {label[:22]} {conf_cls:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img_bgr, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
    cv2.putText(img_bgr, text, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if not fallback:
        cv2.putText(img_bgr, f"Y:{yolo_conf:.2f}", (x1 + 2, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    if not os.path.isfile(args.image):
        sys.exit(f"[Lỗi] Không tìm thấy ảnh: {args.image}")

    print(f"[YOLO] Đang detect … (conf={args.conf}, iou={args.iou})")
    detections = run_yolo_worker(args.image, args.conf, args.iou)

    n_yolo = sum(1 for d in detections if d.get("method","").startswith("custom_yolo"))
    n_hsv  = sum(1 for d in detections if d.get("method","") == "hsv_color")
    n_fb   = sum(1 for d in detections if d["fallback"])

    if n_yolo:
        print(f"[YOLO] YOLO detect: {n_yolo} biển báo.")
    elif n_hsv:
        print(f"[YOLO] YOLO không detect được → HSV fallback: {n_hsv} vùng màu.")
    else:
        print(f"[YOLO] Fallback.")    

    print(f"\n[Model] Đang tải: {args.model} …")
    clf_model = tf.keras.models.load_model(args.model)
    h, w = get_input_size(clf_model)
    print(f"[Model] Input: {h}×{w} | Normalize: {args.normalize}\n")

    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res_img = img_bgr.copy()

    print("=" * 65)
    for idx, det in enumerate(detections):
        x1       = det["x1"]
        y1       = det["y1"]
        x2       = det["x2"]
        y2       = det["y2"]
        yolo_conf = det["yolo_conf"]
        fallback  = det["fallback"]

        if not fallback:
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw // 2, y1 + bh // 2
            size   = max(bw, bh)
            margin = int(size * 0.15)
            new_size = size + margin * 2

            nx1 = max(0, cx - new_size // 2)
            ny1 = max(0, cy - new_size // 2)
            nx2 = min(img_rgb.shape[1], cx + new_size // 2)
            ny2 = min(img_rgb.shape[0], cy + new_size // 2)
        else:
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2

        crop_rgb = img_rgb[ny1:ny2, nx1:nx2]

        crop_resized = cv2.resize(crop_rgb, (w, h)).astype(np.float32)
        if args.normalize:
            crop_resized /= 255.0

        pred   = clf_model.predict(np.expand_dims(crop_resized, 0), verbose=0)[0]
        top3   = np.argsort(pred)[::-1][:3]
        cls_id = int(top3[0])
        conf   = float(pred[cls_id])
        label  = SIGNS.get(cls_id, "Unknown")

        method = det.get("method", "unknown")
        method_tag = (
            "[YOLO]" if method.startswith("custom_yolo") else
            "[HSV] " if method == "hsv_color" else
            "[FB]  "
        )
        region = "toàn ảnh (fallback)" if fallback else f"({x1},{y1})→({x2},{y2})"
        conf_str = f"YOLO Conf: {yolo_conf:.2f}" if yolo_conf > 0 else "HSV (no YOLO conf)"
        print(f"  Box {idx + 1:<3} {method_tag} | {region:<25} | {conf_str}")
        print()

        for rank, cid in enumerate(top3, start=1):
            c_conf = float(pred[cid])
            c_label = SIGNS.get(int(cid), "Unknown")
            bar = "#" * int(c_conf * 25)
            print(f"    #{rank}  Class {cid:2d} — {c_label}")
            print(f"           Rate: {c_conf:.2%}  {bar}")
            print()
        print("─" * 65)

        draw_result(res_img, x1, y1, x2, y2, cls_id, conf, yolo_conf, label, fallback)

    if args.save:
        cv2.imwrite(args.out, res_img)
        print(f"\n[Saved] → {args.out}")
    show_gui = (not args.no_gui) and _has_display()

    if show_gui:
        try:
            cv2.imshow("Ket qua", res_img)
            print("\n[INFO] Đang hiển thị ảnh. Bấm phím bất kỳ trên cửa sổ ảnh để thoát...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"\nKhông thể hiển thị cửa sổ ảnh: {e}")
    else:
        reason = "flag --no-gui được bật" if args.no_gui else "không tìm thấy DISPLAY/WAYLAND"
        print(f"\n[INFO] Bỏ qua hiển thị GUI ({reason}).")


if __name__ == "__main__":
    main()