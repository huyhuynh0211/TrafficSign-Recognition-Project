# ============================================================
#  test_folder_yolo.py  (v2 — fix segfault)
#  Pipeline: YOLOv8n detect (subprocess) → crop → Keras classify
#
#  Fix segfault: YOLO và TensorFlow chạy trong 2 process riêng.
#  Yêu cầu: yolo_worker.py phải nằm cùng thư mục với file này.
#
#  Cài đặt:  pip install ultralytics tensorflow opencv-python
#
#  Cách dùng:
#    python test_folder_yolo.py --model best_model.keras --folder Test/
#    python test_folder_yolo.py --model best_model.keras --folder Test_3/ --csv Test_3/labels.csv
#    python test_folder_yolo.py --model best_model.keras --folder Test/ --save-dir out/ --report r.csv
#    python test_folder_yolo.py --model best_model.keras --folder Test/ --max 20 --conf 0.15
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import csv
import json
import subprocess
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# GTSRB — 43 nhãn tiếng Việt
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

IMG_EXTS    = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
DEFAULT_SIZE = 30

COLOR_HIGH   = (0, 200, 0)
COLOR_MEDIUM = (0, 165, 255)
COLOR_LOW    = (0, 0, 220)

# ============================================================
# HELPERS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8n detect (subprocess) → Keras classify — test cả folder"
    )
    parser.add_argument("--model",    required=True)
    parser.add_argument("--folder",   required=True)
    parser.add_argument("--csv",      default=None)
    parser.add_argument("--conf",     type=float, default=0.25)
    parser.add_argument("--iou",      type=float, default=0.45)
    parser.add_argument("--top3",     action="store_true")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--report",   default=None)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", default=True)
    parser.add_argument("--max",      type=int, default=None, help="Giới hạn số ảnh (debug)")
    return parser.parse_args()


def get_input_size(model):
    try:
        shape = model.input.shape
    except AttributeError:
        shape = model.input_shape
    h = int(shape[1]) if shape[1] is not None else DEFAULT_SIZE
    w = int(shape[2]) if shape[2] is not None else DEFAULT_SIZE
    return h, w


def load_ground_truth(csv_path):
    gt = {}
    if not csv_path or not os.path.isfile(csv_path):
        return gt
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("file") or row.get("Path") or "").strip()
            fname = os.path.basename(fname)
            cid   = (row.get("ClassId") or row.get("class_id") or "").strip()
            if fname and cid.lstrip("-").isdigit():
                gt[fname] = int(cid)
    return gt


def run_yolo_worker(image_path, conf, iou, worker_path):
    """Gọi yolo_worker.py trong subprocess riêng, trả về list of dict."""
    cmd = [sys.executable, worker_path,
           "--image", image_path,
           "--conf",  str(conf),
           "--iou",   str(iou)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None, "timeout"

    if result.returncode != 0:
        return None, result.stderr.strip()

    stdout = result.stdout.strip()
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") or line.startswith("{"):
            try:
                return json.loads(line), None
            except json.JSONDecodeError:
                pass
    return None, f"No JSON in stdout: {stdout[:200]}"


def pick_color(conf):
    if conf >= 0.70: return COLOR_HIGH
    if conf >= 0.45: return COLOR_MEDIUM
    return COLOR_LOW


def draw_box(img_bgr, x1, y1, x2, y2, cls_id, conf_cls, yolo_conf, label, fallback):
    color = pick_color(conf_cls)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    prefix = "[fb] " if fallback else ""
    text = f"{prefix}[{cls_id}] {label[:18]} {conf_cls:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(img_bgr, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
    cv2.putText(img_bgr, text, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    if not fallback:
        cv2.putText(img_bgr, f"Y:{yolo_conf:.2f}", (x1 + 2, y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"[Lỗi] Không tìm thấy folder: '{args.folder}'")
    if not os.path.isfile(args.model):
        sys.exit(f"[Lỗi] Không tìm thấy model: '{args.model}'")

    worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_worker.py")
    if not os.path.isfile(worker_path):
        sys.exit(
            f"[Lỗi] Không tìm thấy yolo_worker.py tại: {worker_path}\n"
            "Hãy đặt yolo_worker.py cùng thư mục với file này."
        )

    # ── Ground truth ─────────────────────────────────────────
    csv_path = args.csv or os.path.join(args.folder, "labels.csv")
    gt = load_ground_truth(csv_path)
    has_gt = bool(gt)

    # ── Load Keras model ─────────────────────────────────────
    print(f"[Model] Đang tải: {args.model} …")
    try:
        clf_model = tf.keras.models.load_model(args.model)
    except Exception as e:
        sys.exit(f"[Lỗi] Không load được model: {e}")
    h, w = get_input_size(clf_model)
    print(f"[Model] Input: {h}×{w}  |  Normalize: {args.normalize}")
    if has_gt:
        print(f"[GT]    {len(gt)} nhãn từ '{csv_path}'")
    else:
        print("[GT]    Không có ground truth")

    # ── Save dir ──────────────────────────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── Danh sách ảnh ────────────────────────────────────────
    all_files = sorted(
        f for f in os.listdir(args.folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )
    if args.max:
        all_files = all_files[:args.max]
    if not all_files:
        sys.exit("[Lỗi] Không tìm thấy ảnh trong folder.")

    print(f"\nTổng ảnh: {len(all_files)}\n")

    # ── Header bảng ──────────────────────────────────────────
    W = 110
    if has_gt:
        print("═" * W)
        print(f"  {'File':<18} │ {'Vùng detect':<20} │ {'Dự đoán':<26} │ {'Conf':>4} │ {'Ground Truth':<26} │ {'KQ':>3}")
        print("─" * W)
    else:
        print("═" * W)
        print(f"  {'File':<18} │ {'Vùng detect':<20} │ {'Dự đoán':<33} │ {'Conf':>4} │ {'YOLO':>4}")
        print("─" * W)

    # ── Thống kê ─────────────────────────────────────────────
    total_imgs  = 0
    yolo_ok     = 0
    fallback_n  = 0
    correct     = 0
    evaluated   = 0
    report_rows = []
    t0 = time.time()

    # ── Xử lý từng ảnh ───────────────────────────────────────
    for fname in all_files:
        fpath = os.path.join(args.folder, fname)
        total_imgs += 1

        # YOLO (subprocess)
        detections, err = run_yolo_worker(fpath, args.conf, args.iou, worker_path)
        if detections is None:
            print(f"  {'[YOLO ERR]':<18} │ {fname}  ({err})")
            continue

        is_fallback = detections[0]["fallback"] if detections else True
        if is_fallback:
            fallback_n += 1
        else:
            yolo_ok += 1

        # Đọc ảnh để crop và vẽ
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            print(f"  {'[READ ERR]':<18} │ {fname}")
            continue
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result_img = img_bgr.copy()

        # Classify từng detection
        best_cls, best_conf, best_label, best_det = None, -1.0, "", None
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            yolo_conf = det["yolo_conf"]

            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_r = cv2.resize(crop, (w, h)).astype(np.float32)
            if args.normalize:
                crop_r /= 255.0
            pred   = clf_model.predict(np.expand_dims(crop_r, 0), verbose=0)[0]
            top_n  = np.argsort(pred)[::-1][:3 if args.top3 else 1]
            cls_id = int(top_n[0])
            conf   = float(pred[cls_id])
            label  = SIGNS.get(cls_id, "Unknown")

            draw_box(result_img, x1, y1, x2, y2, cls_id, conf, yolo_conf, label, det["fallback"])

            if conf > best_conf:
                best_cls, best_conf, best_label, best_det = cls_id, conf, label, det

        if best_det is None:
            continue

        # Ground truth
        gt_id    = gt.get(fname)
        gt_label = SIGNS.get(gt_id, "N/A") if gt_id is not None else "N/A"
        result_str = ""
        if has_gt and gt_id is not None:
            evaluated += 1
            result_str = "✓" if best_cls == gt_id else "✗"
            if best_cls == gt_id:
                correct += 1

        # In bảng
        fn_s = fname[:17] if len(fname) > 17 else fname
        bd   = best_det
        box_s = "toàn ảnh" if bd["fallback"] else f"({bd['x1']},{bd['y1']})→({bd['x2']},{bd['y2']})"
        box_s = box_s[:19] if len(box_s) > 19 else box_s
        lbl_s = best_label[:25] if len(best_label) > 25 else best_label
        if has_gt:
            gt_s = gt_label[:25] if len(gt_label) > 25 else gt_label
            print(f"  {fn_s:<18} │ {box_s:<20} │ [{best_cls:2d}] {lbl_s:<22} │ {best_conf:>3.0%} │ [{gt_id if gt_id is not None else '-':>2}] {gt_s:<22} │ {result_str:>3}")
        else:
            print(f"  {fn_s:<18} │ {box_s:<20} │ [{best_cls:2d}] {lbl_s:<29} │ {best_conf:>3.0%} │ {bd['yolo_conf']:>4.2f}")

        if args.top3:
            # In thêm top-3 của detection tốt nhất
            x1,y1,x2,y2 = bd["x1"],bd["y1"],bd["x2"],bd["y2"]
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size > 0:
                crop_r = cv2.resize(crop, (w,h)).astype(np.float32)
                if args.normalize: crop_r /= 255.0
                pred  = clf_model.predict(np.expand_dims(crop_r,0), verbose=0)[0]
                top3  = np.argsort(pred)[::-1][:3]
                for rank, cid in enumerate(top3[1:], 2):
                    print(f"  {'':18} │ {'  #'+str(rank):<20} │ [{cid:2d}] {SIGNS.get(cid,'?')[:25]:<29} │ {pred[cid]:>3.0%} │")

        # Lưu ảnh kết quả
        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, fname), result_img)

        # Report
        report_rows.append({
            "file":       fname,
            "yolo_box":   f"({bd['x1']},{bd['y1']},{bd['x2']},{bd['y2']})",
            "pred_class": best_cls,
            "pred_label": best_label,
            "confidence": f"{best_conf:.4f}",
            "yolo_conf":  f"{bd['yolo_conf']:.4f}",
            "fallback":   bd["fallback"],
            "gt_class":   gt_id if gt_id is not None else "",
            "gt_label":   gt_label,
            "correct":    result_str,
        })

    elapsed = time.time() - t0

    # ── Tóm tắt ──────────────────────────────────────────────
    print("═" * W)
    print(f"\n{'─'*48}")
    print(f"  Tổng ảnh đã xử lý     : {total_imgs}")
    print(f"  YOLO detect được box   : {yolo_ok}")
    print(f"  Fallback (toàn ảnh)    : {fallback_n}")
    if has_gt and evaluated > 0:
        print(f"  Đúng / Tổng so sánh    : {correct} / {evaluated}  →  Accuracy: {correct/evaluated*100:.2f}%")
    print(f"  Thời gian              : {elapsed:.1f}s  ({elapsed/max(total_imgs,1):.2f}s/ảnh)")
    print(f"{'─'*48}\n")

    if args.report and report_rows:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"[Report] → {args.report}")

    if args.save_dir:
        print(f"[Images] → {args.save_dir}/")


if __name__ == "__main__":
    main()