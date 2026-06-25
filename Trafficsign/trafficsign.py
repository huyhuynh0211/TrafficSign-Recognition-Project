# Suppress TensorFlow verbose logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # An toan khi khong co GUI
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight as cw_utils

tf.get_logger().setLevel('ERROR')  # Suppress TF logger output

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.2


def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python trafficsign.py data_directory [model.h5]")

    images, labels = load_data(sys.argv[1])

    # class weight
    labels_int = np.array(labels)
    class_weights_arr = cw_utils.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_int),
        y=labels_int
    )
    class_weight_dict = {i: class_weights_arr[i] for i in range(NUM_CATEGORIES)}

    labels_cat = tf.keras.utils.to_categorical(labels, NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), labels_cat, test_size=TEST_SIZE, random_state=42
    )

    model = get_model()

    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        class_weight=class_weight_dict
    )

    model.evaluate(x_test, y_test, verbose=2)


    plot_history(history, title="Baseline CNN")

    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Each sub-directory is named 0..NUM_CATEGORIES-1 and contains image files.
    Returns (images, labels) where images are float32 RGB arrays normalised to [0, 1].
    """
    images = []
    labels = []
    skipped = 0

    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_path):
            print(f"Warning: directory not found: {category_path}", file=sys.stderr)
            continue

        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: could not read {img_path}, skipping.", file=sys.stderr)
                skipped += 1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(category)

    if skipped:
        print(f"Skipped {skipped} unreadable file(s).", file=sys.stderr)

    return images, labels


def get_model():
    """
    Returns a compiled CNN.
    Input shape: (IMG_WIDTH, IMG_HEIGHT, 3) — normalised float32 RGB.
    Output: NUM_CATEGORIES units with softmax.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # --- Convolutional Block 1 (5x5 filters for general features) ---
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu"),
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # --- Convolutional Block 2 (3x3 filters for specific features) ---
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Fully-connected head
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer — one unit per category
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_history(history, title="Model"):
    """
    Plot training accuracy and loss per epoch.
    Saves a PNG to the results/ directory.
    """
    os.makedirs("results", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # --- Accuracy ---
    ax1.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
    ax1.set_title(f"{title} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # --- Loss ---
    ax2.plot(history.history["loss"], label="Train Loss", color="orange", marker="o")
    ax2.set_title(f"{title} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    safe_title = title.replace(" ", "_")
    save_path = os.path.join("results", f"{safe_title}_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()