
import os, csv, random, shutil, sys
import cv2

GTSDB_DIR  = "gtsdb"
OUTPUT_DIR = "gtsdb_yolo"

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.10


def main():
    gt_file = os.path.join(GTSDB_DIR, "gt.txt")

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

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # ── Shuffle và chia train/val/test ───────────────────────
    all_fnames = list(annotations.keys())
    random.seed(42)
    random.shuffle(all_fnames)

    n_total  = len(all_fnames)
    n_train  = int(n_total * TRAIN_RATIO)
    n_val    = int(n_total * VAL_RATIO)

    split_map = {}
    for fn in all_fnames[:n_train]:
        split_map[fn] = "train"
    for fn in all_fnames[n_train:n_train + n_val]:
        split_map[fn] = "val"
    for fn in all_fnames[n_train + n_val:]:
        split_map[fn] = "test"

    # ── Convert và lưu ──────────────────────────────────────
    ok   = 0
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

        safe_base = os.path.splitext(fname)[0].replace(os.sep, "_").replace("/", "_")

        out_img = os.path.join(OUTPUT_DIR, "images", split, safe_base + ".jpg")
        cv2.imwrite(out_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        out_lbl = os.path.join(OUTPUT_DIR, "labels", split, safe_base + ".txt")
        with open(out_lbl, "w") as f:
            for (x1, y1, x2, y2) in boxes:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                cx = ((x1 + x2) / 2) / iw
                cy = ((y1 + y2) / 2) / ih
                bw = (x2 - x1) / iw
                bh = (y2 - y1) / ih
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        ok += 1

    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    abs_path  = os.path.abspath(OUTPUT_DIR)
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write("test:  images/test\n")
        f.write("nc: 1\n")
        f.write("names: ['traffic-sign']\n")

    n_train_out = sum(1 for v in split_map.values() if v == "train")
    n_val_out   = sum(1 for v in split_map.values() if v == "val")
    n_test_out  = sum(1 for v in split_map.values() if v == "test")

    print(f"[OK] Đã convert: {ok} ảnh  "
          f"(train={n_train_out}, val={n_val_out}, test={n_test_out}), skip={skip}")
    print(f"[OK] Dataset YOLO → '{OUTPUT_DIR}/'")
    print(f"[OK] dataset.yaml → '{yaml_path}'")


if __name__ == "__main__":
    main()