import sys, json, os
os.environ['YOLO_VERBOSE'] = 'False'


# ──────────────────────────────────────────────────────────────────
#  NMS helper — loại bỏ box trùng lặp (IoU-based)
# ──────────────────────────────────────────────────────────────────
def _iou(a, b):
    """Tính IoU giữa 2 boxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(boxes, iou_thresh=0.45):
    """
    Non-Maximum Suppression đơn giản theo confidence (yolo_conf).
    boxes: list of dict với keys x1,y1,x2,y2,yolo_conf,...
    Trả về list đã lọc.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b["yolo_conf"], reverse=True)
    keep = []
    suppressed = [False] * len(boxes)
    for i, bi in enumerate(boxes):
        if suppressed[i]:
            continue
        keep.append(bi)
        rect_i = (bi["x1"], bi["y1"], bi["x2"], bi["y2"])
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            rect_j = (boxes[j]["x1"], boxes[j]["y1"], boxes[j]["x2"], boxes[j]["y2"])
            if _iou(rect_i, rect_j) > iou_thresh:
                suppressed[j] = True
    return keep


# ──────────────────────────────────────────────────────────────────
#  HSV color-based fallback detector
#  Chỉ dùng khi YOLO hoàn toàn không tìm được box nào.
# ──────────────────────────────────────────────────────────────────
def hsv_detect_signs(img_bgr):
    import cv2, numpy as np
    ih, iw = img_bgr.shape[:2]
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_red1   = cv2.inRange(img_hsv, np.array([0,   100,  60]), np.array([10,  255, 255]))
    mask_red2   = cv2.inRange(img_hsv, np.array([160, 100,  60]), np.array([180, 255, 255]))
    mask_blue   = cv2.inRange(img_hsv, np.array([100, 100,  60]), np.array([130, 255, 255]))
    mask_yellow = cv2.inRange(img_hsv, np.array([15,  100, 100]), np.array([35,  255, 255]))

    mask_all = cv2.bitwise_or(
        cv2.bitwise_or(mask_red1, mask_red2),
        cv2.bitwise_or(mask_blue, mask_yellow)
    )

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Giới hạn kích thước hợp lý: biển báo chiếm 0.08%–15% diện tích ảnh
    MIN_AREA = (iw * ih) * 0.0008
    MAX_AREA = (iw * ih) * 0.15     # ← giảm từ 0.50 xuống 0.15, loại vùng quá lớn
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / max(h, 1)
        if ratio < 0.3 or ratio > 3.5:   # tỷ lệ hợp lý hơn cho biển báo
            continue
        pad = int(max(w, h) * 0.10)
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)
        boxes.append((x1, y1, x2, y2, area))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return [(b[0], b[1], b[2], b[3]) for b in boxes[:5]]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--yolo-model", default=None)
    args = parser.parse_args()

    import cv2, numpy as np
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(json.dumps({"error": f"Cannot read: {args.image}"}))
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ih, iw  = img_bgr.shape[:2]

    # ── 1. Custom YOLO ───────────────────────────────────────
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    custom_model = args.yolo_model or os.path.join(script_dir, "traffic_sign_detector.pt")
    print(f"[DEBUG] custom_model path: {custom_model}", file=sys.stderr)
    print(f"[DEBUG] file exists: {os.path.isfile(custom_model)}", file=sys.stderr)

    yolo_boxes = []
    if os.path.isfile(custom_model):
        try:
            from ultralytics import YOLO
            yolo      = YOLO(custom_model)
            results   = yolo(img_rgb, conf=args.conf, iou=args.iou, verbose=False)
            boxes_all = results[0].boxes
            print(f"[DEBUG] YOLO detect: {len(boxes_all)} boxes", file=sys.stderr)
            for b in boxes_all:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue
                yolo_boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "yolo_conf": round(float(b.conf[0]), 4),
                    "method": "custom_yolo",
                    "fallback": False,
                })
        except Exception as e:
            print(f"[DEBUG] Custom YOLO exception: {e}", file=sys.stderr)

    # ── 2. Nếu YOLO tìm được → dùng ngay, NMS nội bộ ───────
    if yolo_boxes:
        out = _nms(yolo_boxes, iou_thresh=args.iou)
        print(f"[DEBUG] Sau NMS: {len(out)} YOLO boxes", file=sys.stderr)
        print(json.dumps(out))
        return

    # ── 3. YOLO không tìm được → thử lại với conf thấp hơn ─
    #    (Dành cho ảnh khó: điều kiện ánh sáng kém, biển báo nhỏ)
    LOW_CONF = 0.15
    if os.path.isfile(custom_model):
        try:
            from ultralytics import YOLO
            print(f"[DEBUG] YOLO retry với conf={LOW_CONF}", file=sys.stderr)
            yolo    = YOLO(custom_model)
            results = yolo(img_rgb, conf=LOW_CONF, iou=args.iou, verbose=False)
            for b in results[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue
                yolo_boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "yolo_conf": round(float(b.conf[0]), 4),
                    "method": "custom_yolo_lowconf",
                    "fallback": False,
                })
            print(f"[DEBUG] YOLO retry: {len(yolo_boxes)} boxes", file=sys.stderr)
        except Exception as e:
            print(f"[DEBUG] YOLO retry exception: {e}", file=sys.stderr)

    if yolo_boxes:
        out = _nms(yolo_boxes, iou_thresh=args.iou)
        print(f"[DEBUG] Sau NMS (low-conf retry): {len(out)} boxes", file=sys.stderr)
        print(json.dumps(out))
        return

    # ── 4. HSV fallback — chỉ khi YOLO thực sự trắng tay ───
    print("[DEBUG] Không có YOLO box → dùng HSV", file=sys.stderr)
    hsv_results = hsv_detect_signs(img_bgr)
    if hsv_results:
        out = []
        for (x1, y1, x2, y2) in hsv_results:
            out.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "yolo_conf": 0.0,
                "method": "hsv_color",
                "fallback": False,
            })
        print(f"[DEBUG] HSV detect: {len(out)} boxes", file=sys.stderr)
        print(json.dumps(out))
        return

    # ── 5. Fallback toàn ảnh ────────────────────────────────
    print("[DEBUG] HSV cũng trắng tay → fallback toàn ảnh", file=sys.stderr)
    out = [{
        "x1": 0, "y1": 0, "x2": iw, "y2": ih,
        "yolo_conf": 0.0,
        "method": "fallback_full_image",
        "fallback": True,
    }]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
