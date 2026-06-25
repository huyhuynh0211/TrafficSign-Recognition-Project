# ============================================================
#  prepare_gtsdb.py
#  Convert GTSDB → YOLO format với 1 class duy nhất: "traffic-sign"
#
#  Cách dùng:
#    1. Tải GTSDB tại: https://benchmark.ini.rub.de/gtsdb_dataset.html
#       File: FullIJCNN2013.zip (~600MB)
#    2. Giải nén vào thư mục GTSDB/ (cùng chỗ với file này)
#    3. python prepare_gtsdb.py
#    4. Kết quả: thư mục gtsdb_yolo/ sẵn sàng để train
# ============================================================

import os, csv, random, shutil, sys
import cv2

GTSDB_DIR  = "gtsdb"           # thư mục giải nén GTSDB vào đây
OUTPUT_DIR = "gtsdb_yolo"      # output YOLO dataset
TRAIN_RATIO = 0.85

def main():
    gt_file = os.path.join(GTSDB_DIR, "gt.txt")
    if not os.path.isdir(GTSDB_DIR) or not os.path.isfile(gt_file):
        print(f"[Lỗi] Không tìm thấy GTSDB tại '{GTSDB_DIR}/'")
        print()
        print("  Bước 1: Vào https://benchmark.ini.rub.de/gtsdb_dataset.html")
        print("  Bước 2: Tải file FullIJCNN2013.zip")
        print("  Bước 3: Giải nén → đặt vào thư mục GTSDB/")
        print("  Bước 4: Chạy lại script này")
        sys.exit(1)

    print(f"[OK] Tìm thấy GTSDB tại '{GTSDB_DIR}/'")

    # ── Đọc gt.txt ──────────────────────────────────────────
    # Format: filename.ppm;x1;y1;x2;y2;classId
    annotations = {}
    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 6:
                continue
            fname = parts[0].strip()
            try:
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            except ValueError:
                continue
            # Bỏ qua class id (parts[5]) — gộp tất cả thành 1 class
            if fname not in annotations:
                annotations[fname] = []
            annotations[fname].append((x1, y1, x2, y2))

    print(f"[OK] Đọc được {sum(len(v) for v in annotations.values())} bbox "
          f"từ {len(annotations)} ảnh")

    # ── Tạo cấu trúc thư mục YOLO ───────────────────────────
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # ── Shuffle và chia train/val ────────────────────────────
    all_fnames = list(annotations.keys())
    random.seed(42)
    random.shuffle(all_fnames)
    split_idx   = int(len(all_fnames) * TRAIN_RATIO)
    split_map   = {}
    for fn in all_fnames[:split_idx]:
        split_map[fn] = "train"
    for fn in all_fnames[split_idx:]:
        split_map[fn] = "val"

    # ── Convert và lưu ──────────────────────────────────────
    ok = 0
    skip = 0
    for fname, boxes in annotations.items():
        img_path = os.path.join(GTSDB_DIR, fname)
        if not os.path.isfile(img_path):
            skip += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skip += 1
            continue

        ih, iw = img.shape[:2]
        split   = split_map[fname]

        # Lưu ảnh dưới dạng .jpg
        base_name = os.path.splitext(fname)[0]
        out_img   = os.path.join(OUTPUT_DIR, "images", split, base_name + ".jpg")
        cv2.imwrite(out_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Lưu label YOLO (class cx cy w h — normalized, class luôn = 0)
        out_lbl = os.path.join(OUTPUT_DIR, "labels", split, base_name + ".txt")
        with open(out_lbl, "w") as f:
            for (x1, y1, x2, y2) in boxes:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                cx = ((x1 + x2) / 2) / iw
                cy = ((y1 + y2) / 2) / ih
                bw = (x2 - x1) / iw
                bh = (y2 - y1) / ih
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        ok += 1

    # ── Tạo dataset.yaml ─────────────────────────────────────
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    abs_path  = os.path.abspath(OUTPUT_DIR)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['traffic-sign']\n")

    n_train = sum(1 for v in split_map.values() if v == "train")
    n_val   = len(split_map) - n_train
    print(f"[OK] Đã convert: {ok} ảnh  (train={n_train}, val={n_val}), skip={skip}")
    print(f"[OK] Dataset YOLO → '{OUTPUT_DIR}/'")
    print(f"[OK] dataset.yaml → '{yaml_path}'")
    print()
    print("Bước tiếp theo:")
    print("  python train_detector.py")

if __name__ == "__main__":
    main()