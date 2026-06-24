# ============================================================
#  test_folder_yolo.py
#  Dùng YOLOv8n để detect vùng biển báo trong từng ảnh
#  của một folder, sau đó phân loại bằng model Keras.
#  Hỗ trợ so sánh với ground truth (CSV) và in báo cáo.
#
#  Cài đặt:
#    pip install ultralytics tensorflow opencv-python
#
#  Cách dùng:
#    python test_folder_yolo.py --model best_model.keras --folder Test/
#    python test_folder_yolo.py --model best_model.keras --folder Test_3/ --csv Test_3/labels.csv
#    python test_folder_yolo.py --model best_model.keras --folder Test/ --save-dir results_yolo/
#    python test_folder_yolo.py --model best_model.keras --folder Test/ --conf 0.15 --top3
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import csv
import sys
import time
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

IMG_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
COCO_SIGN_CLASSES = {9, 11}   # traffic light, stop sign

COLOR_HIGH   = (0, 200, 0)
COLOR_MEDIUM = (0, 165, 255)
COLOR_LOW    = (0, 0, 220)

# ============================================================
# HELPERS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8n detect → Keras classify — test cả folder"
    )
    parser.add_argument("--model",    required=True, help="File .keras / .h5 của model phân loại")
    parser.add_argument("--folder",   required=True, help="Folder chứa ảnh cần test")
    parser.add_argument("--csv",      default=None,
                        help="File CSV ground truth. Mặc định tìm labels.csv trong folder")
    parser.add_argument("--conf",     type=float, default=0.25,
                        help="Ngưỡng confidence YOLO (mặc định: 0.25)")
    parser.add_argument("--iou",      type=float, default=0.45,
                        help="Ngưỡng IoU NMS YOLO (mặc định: 0.45)")
    parser.add_argument("--top3",     action="store_true",
                        help="Hiển thị top-3 dự đoán cho mỗi ảnh")
    parser.add_argument("--save-dir", default=None,
                        help="Thư mục lưu ảnh kết quả có bounding box. Mặc định không lưu.")
    parser.add_argument("--report",   default=None,
                        help="Lưu báo cáo ra file CSV. Vd: --report report.csv")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", default=True,
                        help="Tắt chuẩn hoá [0,1]")
    parser.add_argument("--max",      type=int, default=None,
                        help="Giới hạn số ảnh xử lý (để test nhanh)")
    return parser.parse_args()


def get_input_size(model):
    try:
        shape = model.input.shape
    except AttributeError:
        shape = model.input_shape
    h = int(shape[1]) if shape[1] is not None else 30
    w = int(shape[2]) if shape[2] is not None else 30
    return h, w


def load_ground_truth(csv_path):
    """Trả về dict: filename → class_id (int)."""
    gt = {}
    if not csv_path or not os.path.isfile(csv_path):
        return gt
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("file") or row.get("Path") or "").strip()
            fname = os.path.basename(fname)
            cid   = row.get("ClassId") or row.get("class_id") or ""
            if fname and cid.strip().lstrip("-").isdigit():
                gt[fname] = int(cid)
    return gt


def pick_color(conf):
    if conf >= 0.70: return COLOR_HIGH
    if conf >= 0.45: return COLOR_MEDIUM
    return COLOR_LOW


def draw_box(img_bgr, x1, y1, x2, y2, cls_id, conf_cls, yolo_conf, label):
    color = pick_color(conf_cls)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    text = f"[{cls_id}] {label[:20]} {conf_cls:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img_bgr, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
    cv2.putText(img_bgr, text, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_bgr, f"Y:{yolo_conf:.2f}", (x1 + 2, y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)


def classify_crop(crop_rgb, clf_model, h, w, normalize):
    """Resize, normalize và predict một crop. Trả về (scores array)."""
    crop = cv2.resize(crop_rgb, (w, h)).astype(np.float32)
    if normalize:
        crop /= 255.0
    return clf_model.predict(np.expand_dims(crop, 0), verbose=0)[0]


def process_image(img_path, yolo, clf_model, h, w, normalize, conf_thr, iou_thr, top3):
    """
    Xử lý 1 ảnh: YOLO detect → crop → classify.

    Trả về list of dict với mỗi detection:
        {box, cls_id, confidence, label, yolo_conf, top3_list}
    và ảnh BGR đã vẽ bounding box.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ih, iw  = img_bgr.shape[:2]

    # ── YOLO detect ──────────────────────────────────────────
    yolo_results = yolo(img_rgb, conf=conf_thr, iou=iou_thr, verbose=False)
    boxes_all = yolo_results[0].boxes

    sign_boxes = [b for b in boxes_all if int(b.cls) in COCO_SIGN_CLASSES]
    if not sign_boxes:
        sign_boxes = list(boxes_all)   # fallback: tất cả object

    detections = []
    result_img = img_bgr.copy()

    if not sign_boxes:
        # Fallback hoàn toàn: classify cả ảnh
        scores  = classify_crop(img_rgb, clf_model, h, w, normalize)
        top_n   = np.argsort(scores)[::-1][:3 if top3 else 1]
        cls_id  = int(top_n[0])
        detections.append({
            "box":       None,
            "cls_id":    cls_id,
            "confidence": float(scores[cls_id]),
            "label":     SIGNS.get(cls_id, "Unknown"),
            "yolo_conf": 0.0,
            "fallback":  True,
            "top3":      [(int(c), float(scores[c])) for c in top_n],
        })
        # Ghi nhãn vào góc ảnh
        label = SIGNS.get(cls_id, "Unknown")
        cv2.putText(result_img, f"[fallback] [{cls_id}] {label[:30]} {scores[cls_id]:.0%}",
                    (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1, cv2.LINE_AA)
        return detections, result_img

    for box in sign_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        yolo_conf = float(box.conf[0])

        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        scores = classify_crop(crop, clf_model, h, w, normalize)
        top_n  = np.argsort(scores)[::-1][:3 if top3 else 1]
        cls_id = int(top_n[0])
        conf   = float(scores[cls_id])
        label  = SIGNS.get(cls_id, "Unknown")

        detections.append({
            "box":        (x1, y1, x2, y2),
            "cls_id":     cls_id,
            "confidence": conf,
            "label":      label,
            "yolo_conf":  yolo_conf,
            "fallback":   False,
            "top3":       [(int(c), float(scores[c])) for c in top_n],
        })
        draw_box(result_img, x1, y1, x2, y2, cls_id, conf, yolo_conf, label)

    return detections, result_img


# ============================================================
# MAIN
# ============================================================

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"[Lỗi] Không tìm thấy folder: '{args.folder}'")
    if not os.path.isfile(args.model):
        sys.exit(f"[Lỗi] Không tìm thấy model: '{args.model}'")

    # ── Load YOLO ────────────────────────────────────────────
    print("Đang tải YOLOv8n …")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[Lỗi] Chưa cài ultralytics. Chạy: pip install ultralytics")
    yolo = YOLO("yolov8n.pt")

    # ── Load Keras model ─────────────────────────────────────
    print(f"Đang tải Keras model: {args.model} …")
    try:
        clf_model = tf.keras.models.load_model(args.model)
    except Exception as e:
        sys.exit(f"[Lỗi] Không load được model: {e}")
    h, w = get_input_size(clf_model)
    print(f"  Input size: {h}×{w}  |  Normalize: {args.normalize}\n")

    # ── Ground truth ─────────────────────────────────────────
    csv_path = args.csv or os.path.join(args.folder, "labels.csv")
    gt = load_ground_truth(csv_path)
    has_gt = bool(gt)
    if has_gt:
        print(f"Ground truth: {len(gt)} nhãn từ '{csv_path}'")
    else:
        print("Ground truth: không có (chạy không so sánh accuracy)")

    # ── Chuẩn bị save dir ────────────────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── Lấy danh sách ảnh ────────────────────────────────────
    all_files = sorted(
        f for f in os.listdir(args.folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
    if args.max:
        all_files = all_files[:args.max]

    if not all_files:
        sys.exit("[Lỗi] Không tìm thấy ảnh nào trong folder.")

    print(f"Tổng ảnh cần xử lý: {len(all_files)}\n")

    # ── Tiêu đề bảng ─────────────────────────────────────────
    W = 115
    if has_gt:
        print("═" * W)
        print(f"  {'File':<20} │ {'Vị trí detect':<22} │ {'Dự đoán':<28} │ {'Conf':>5} │ {'Ground Truth':<28} │ {'KQ':>4}")
        print("─" * W)
    else:
        print("═" * W)
        print(f"  {'File':<20} │ {'Vị trí detect':<22} │ {'Dự đoán':<35} │ {'Conf':>5} │ {'YOLO':>5}")
        print("─" * W)

    # ── Thống kê ─────────────────────────────────────────────
    total_imgs  = 0
    total_det   = 0   # ảnh có YOLO detect được ≥1 box
    correct     = 0
    evaluated   = 0
    fallback_n  = 0
    report_rows = []
    t0 = time.time()

    # ── Xử lý từng ảnh ───────────────────────────────────────
    for fname in all_files:
        fpath = os.path.join(args.folder, fname)
        total_imgs += 1

        detections, result_img = process_image(
            fpath, yolo, clf_model, h, w,
            args.normalize, args.conf, args.iou, args.top3
        )

        if detections is None:
            print(f"  {'[ĐỌC LỖI]':<20} │ {fname}")
            continue

        # Lưu ảnh kết quả
        if args.save_dir and result_img is not None:
            cv2.imwrite(os.path.join(args.save_dir, fname), result_img)

        gt_id    = gt.get(fname)
        gt_label = SIGNS.get(gt_id, "N/A") if gt_id is not None else "N/A"

        # Chọn detection chính: box confidence cao nhất (hoặc duy nhất)
        best_det = max(detections, key=lambda d: d["confidence"])
        cls_id   = best_det["cls_id"]
        conf     = best_det["confidence"]
        label    = best_det["label"]
        yconf    = best_det["yolo_conf"]
        box_str  = "fullimg" if best_det["fallback"] else \
                   "({},{})→({},{})".format(*best_det["box"])

        is_fallback = best_det["fallback"]
        if is_fallback:
            fallback_n += 1
        else:
            total_det += 1

        # So sánh với ground truth
        result_str = ""
        if has_gt and gt_id is not None:
            evaluated += 1
            if cls_id == gt_id:
                correct += 1
                result_str = "✓"
            else:
                result_str = "✗"

        # In ra bảng
        fname_s = fname[:19] if len(fname) > 19 else fname
        box_s   = box_str[:21] if len(box_str) > 21 else box_str
        label_s = label[:27] if len(label) > 27 else label
        if has_gt:
            gt_s = gt_label[:27] if len(gt_label) > 27 else gt_label
            print(f"  {fname_s:<20} │ {box_s:<22} │ [{cls_id:2d}] {label_s:<24} │ {conf:>4.0%} │ [{gt_id if gt_id is not None else '-':>2}] {gt_s:<24} │ {result_str:>4}")
        else:
            print(f"  {fname_s:<20} │ {box_s:<22} │ [{cls_id:2d}] {label_s:<31} │ {conf:>4.0%} │ {yconf:>4.2f}")

        # Top-3 thêm
        if args.top3 and len(best_det["top3"]) > 1:
            for rank, (cid, sc) in enumerate(best_det["top3"][1:], 2):
                print(f"  {'':20} │ {'  top-'+str(rank):<22} │     [{cid:2d}] {SIGNS.get(cid,'?')[:26]:<27} │ {sc:>4.0%} │")

        # Chuẩn bị report
        report_rows.append({
            "file":         fname,
            "yolo_box":     box_str,
            "pred_class":   cls_id,
            "pred_label":   label,
            "confidence":   f"{conf:.4f}",
            "yolo_conf":    f"{yconf:.4f}",
            "gt_class":     gt_id if gt_id is not None else "",
            "gt_label":     gt_label,
            "correct":      result_str,
            "fallback":     is_fallback,
        })

    elapsed = time.time() - t0

    # ── Tóm tắt ──────────────────────────────────────────────
    print("═" * W)
    print(f"\n{'─'*50}")
    print(f"  Tổng ảnh đã xử lý    : {total_imgs}")
    print(f"  YOLO detect được box  : {total_det}  (fallback toàn ảnh: {fallback_n})")
    if has_gt and evaluated > 0:
        acc = correct / evaluated * 100
        print(f"  So khớp ground truth  : {correct} / {evaluated}  →  Accuracy: {acc:.2f}%")
    print(f"  Thời gian xử lý       : {elapsed:.1f}s  ({elapsed/max(total_imgs,1):.2f}s/ảnh)")
    print(f"{'─'*50}\n")

    # ── Lưu report CSV ───────────────────────────────────────
    if args.report and report_rows:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"[Report] Đã lưu → {args.report}")

    if args.save_dir:
        print(f"[Images] Ảnh kết quả → {args.save_dir}/")


if __name__ == "__main__":
    main()