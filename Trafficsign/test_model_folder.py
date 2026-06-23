#Turn off error of the new version
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
######################

import argparse
import sys
import cv2
import numpy as np
import tensorflow as tf
import csv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

tf.get_logger().setLevel('ERROR')

SIGNS = {
    0: "Giới hạn tốc độ 20 km/h",
    1: "Giới hạn tốc độ 30 km/h",
    2: "Giới hạn tốc độ 50 km/h",
    3: "Giới hạn tốc độ 60 km/h",
    4: "Giới hạn tốc độ 70 km/h",
    5: "Giới hạn tốc độ 80 km/h",
    6: "Hết giới hạn tốc độ 80 km/h",
    7: "Giới hạn tốc độ 100 km/h",
    8: "Giới hạn tốc độ 120 km/h",
    9: "Cấm vượt",
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

DEFAULT_SIZE = 30  # fallback if model has dynamic input shape


def parse_args():
    parser = argparse.ArgumentParser(description="Test model on a folder of traffic sign images")
    parser.add_argument("model", help="Path to .h5 or .keras model file")
    parser.add_argument("folder", help="Path to folder containing test images")
    parser.add_argument("--csv", help="Optional: Path to CSV file with truth. Default is to look for labels.csv in the folder.", default=None)
    return parser.parse_args()


def get_input_size(model, fallback=DEFAULT_SIZE):
    """Return (height, width) from model input shape, with fallback for dynamic dims."""
    try:
        shape = model.input.shape
    except AttributeError:
        shape = model.input_shape

    h = shape[1] if shape[1] is not None else fallback
    w = shape[2] if shape[2] is not None else fallback

    return int(h), int(w)


def load_ground_truth(csv_path):
    """Reads CSV and returns a dict mapping filename to integer ClassId."""
    ground_truth = {}
    if not os.path.exists(csv_path):
        return ground_truth
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Try to get filename, sometimes it's under 'file' or 'Path'
            filename = row.get('file', row.get('Path', '')).split('/')[-1].split('\\')[-1]
            class_id = row.get('ClassId', row.get('ClassId'))
            if filename and class_id is not None:
                ground_truth[filename] = int(class_id)
    return ground_truth


def main():
    args = parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Error: '{args.folder}' is not a valid directory.")

    # Determine CSV path
    csv_path = args.csv if args.csv else os.path.join(args.folder, "labels.csv")
    ground_truth = load_ground_truth(csv_path)

    # Load model
    try:
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        sys.exit(f"Error loading model '{args.model}': {e}")

    h, w = get_input_size(model)

    print(f"Loaded model     : {args.model}")
    print(f"Model Input size : {h}x{w}")
    print(f"Test Folder      : {args.folder}")
    if ground_truth:
        print(f"Truth CSV : Found with {len(ground_truth)} labels.")
    else:
        print(f"Truth CSV : None (labels.csv not found)")
    print("=" * 105)

    filenames = []
    images = []

    # Read all images from the folder
    for filename in sorted(os.listdir(args.folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".ppm", ".bmp")):
            continue
        
        filepath = os.path.join(args.folder, filename)
        img = cv2.imread(filepath)
        
        if img is None:
            print(f"Warning: Could not read image '{filename}'. Skipping.")
            continue

        # Preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        img = img.astype(np.float32) / 255.0  # Normalize
        
        images.append(img)
        filenames.append(filename)

    if not images:
        print("No valid images found in the directory.")
        return

    # Convert to batch numpy array
    images_batch = np.array(images)

    # Predict all at once
    predictions = model.predict(images_batch, verbose=0)

    # Display results
    correct_count = 0
    total_evaluated = 0

    if ground_truth:
        print(f"{'Filename':<20} | {'Predicted':<30} | {'Truth':<30} | {'Result':<6} | {'Conf':>6}")
        print("-" * 105)
    else:
        print(f"{'Filename':<25} | {'Predicted Class':<40} | {'Conf':>10}")
        print("-" * 80)
    
    for i in range(len(filenames)):
        fname_full = filenames[i]
        scores = predictions[i]
        class_id = np.argmax(scores)
        confidence = scores[class_id] * 100
        
        pred_label = SIGNS.get(class_id, "Unknown")
        
        if ground_truth:
            gt_id = ground_truth.get(fname_full)
            if gt_id is not None:
                total_evaluated += 1
                gt_label = SIGNS.get(gt_id, "Unknown")
                is_correct = (class_id == gt_id)
                if is_correct:
                    correct_count += 1
                    result_str = "PASS"
                else:
                    result_str = "FAIL"
                
                # Truncate strings for neat columns
                f_str = fname_full if len(fname_full) <= 20 else fname_full[:17] + "..."
                p_str = pred_label if len(pred_label) <= 30 else pred_label[:27] + "..."
                g_str = gt_label if len(gt_label) <= 30 else gt_label[:27] + "..."
                
                print(f"{f_str:<20} | {p_str:<30} | {g_str:<30} | {result_str:<6} | {confidence:>5.1f}%")
            else:
                # File exists but no ground truth
                f_str = fname_full if len(fname_full) <= 20 else fname_full[:17] + "..."
                p_str = pred_label if len(pred_label) <= 30 else pred_label[:27] + "..."
                print(f"{f_str:<20} | {p_str:<30} | {'N/A':<30} | {'N/A':<6} | {confidence:>5.1f}%")
        else:
            f_str = fname_full if len(fname_full) <= 25 else fname_full[:22] + "..."
            p_str = pred_label if len(pred_label) <= 40 else pred_label[:37] + "..."
            print(f"{f_str:<25} | {p_str:<40} | {confidence:>9.2f}%")

    if ground_truth:
        print("-" * 105)
    else:
        print("-" * 80)

    print(f"Successfully processed {len(filenames)} images.")
    if ground_truth and total_evaluated > 0:
        accuracy = (correct_count / total_evaluated) * 100
        print(f"Correct Predictions  : {correct_count} / {total_evaluated}")
        print(f"Overall Accuracy     : {accuracy:.2f}%")

if __name__ == "__main__":
    main()
