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

tf.get_logger().setLevel('ERROR') #Turn off error of the new version

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

DEFAULT_SIZE = 30
DEFAULT_NORMALIZE = True


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic sign classifier")
    parser.add_argument("model", help="Path to .h5 or .keras model file")
    parser.add_argument("image", help="Path to image file")
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="Divide pixel values by 255 (default: True)",
    )
    normalize_group.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Do NOT divide by 255 (use if model was trained on raw [0,255] pixels)",
    )
    return parser.parse_args()


def get_input_size(model, fallback=DEFAULT_SIZE):
    """Return (height, width) from model input shape, with fallback for dynamic dims."""
    try:
        shape = model.input.shape
    except AttributeError:
        shape = model.input_shape

    if len(shape) < 4:
        sys.exit(f"Unexpected model input rank {len(shape)}, expected 4 (batch, H, W, C).")

    h = shape[1] if shape[1] is not None else fallback
    w = shape[2] if shape[2] is not None else fallback

    if shape[1] is None or shape[2] is None:
        print(
            f"Warning: model has dynamic input shape {tuple(shape)}. "
            f"Defaulting to {fallback}x{fallback}.",
            file=sys.stderr,
        )

    return int(h), int(w)


def main():
    args = parse_args()

    try:
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        sys.exit(f"Error loading model '{args.model}': {e}")

    h, w = get_input_size(model)

    normalize = args.normalize

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Error: Could not read image '{args.image}'.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32)
    if normalize:
        img /= 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    scores = pred[0]

    print(f"Model     : {args.model}")
    print(f"Input size: {h}x{w}  |  Normalize: {normalize}")
    print(f"Image     : {args.image}")
    print()

    top3 = np.argsort(scores)[::-1][:3]
    for rank, cls in enumerate(top3, start=1):
        confidence = scores[cls]
        bar = "#" * int(confidence * 25)
        label = SIGNS.get(int(cls), "Unknown")
        print(f"  #{rank}  Class {cls:2d} — {label}")
        print(f"         Rate: {confidence:.2%}  {bar}")
        print()


if __name__ == "__main__":
    main()