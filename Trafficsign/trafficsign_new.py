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
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.get_logger().setLevel('ERROR')  # Suppress TF logger output

EPOCHS = 30
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.2
BATCH_SIZE = 32


def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python trafficsign_new.py data_directory [model.h5]")

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("=" * 60)
    print("LOADING DATA...")
    images, labels = load_data(sys.argv[1])
    labels_int = np.array(labels)
    print(f"Loaded {len(images)} images across {NUM_CATEGORIES} classes.")

    # ============================================================
    # 2. ANALYZE CLASS DISTRIBUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    class_counts = {i: int(np.sum(labels_int == i)) for i in range(NUM_CATEGORIES)}
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)
    print(f"  Min class {min_class}: {class_counts[min_class]} images")
    print(f"  Max class {max_class}: {class_counts[max_class]} images")
    print(f"  Imbalance ratio: {class_counts[max_class] / class_counts[min_class]:.1f}:1")

    # ============================================================
    # 3. COMPUTE CLASS WEIGHTS
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPUTING CLASS WEIGHTS FOR IMBALANCED DATA")
    class_weights_array = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_int),
        y=labels_int
    )
    class_weight_dict = {i: class_weights_array[i] for i in range(NUM_CATEGORIES)}
    print(f"  Weight range: {min(class_weights_array):.3f} - {max(class_weights_array):.3f}")

    # ============================================================
    # 4. PREPARE DATA
    # ============================================================
    print("\n" + "=" * 60)
    print("PREPARING DATA")

    # Images are already normalised in load_data() — do NOT divide by 255 again
    x_data = np.array(images, dtype=np.float32)
    labels_cat = tf.keras.utils.to_categorical(labels_int, NUM_CATEGORIES)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, labels_cat, test_size=TEST_SIZE, random_state=42
    )
    # Derive integer class indices for classification_report
    y_test_int = np.argmax(y_test, axis=1)

    print(f"  Train: {x_train.shape[0]} | Test: {x_test.shape[0]}")
    print(f"  Input shape: {x_train.shape[1:]}")

    # ============================================================
    # 5. TRAIN MODELS
    # ============================================================
    # Store (model_object, history, test_acc) tuples keyed by a plain string
    results = {}

    def _train(name, model):
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {name}")
        model.summary()
        # Define data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            horizontal_flip=False, # Traffic signs are not horizontally symmetric
            fill_mode="nearest"
        )

        # Tối ưu hóa thời gian train: Tự động dừng nếu test_acc không tăng sau 5 epochs
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(x_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"  Test Accuracy: {acc * 100:.2f}%")
        results[name] = {"model": model, "history": history, "test_acc": acc}

    _train("Baseline CNN + Class Weights",      build_baseline_cnn())
    _train("CNN + L2 Reg + Class Weights",      build_cnn_with_l2())
    _train("Deeper CNN (3 Blocks + BN)",         build_deeper_cnn())
    _train("Transfer Learning (MobileNetV2)",    build_transfer_learning_model())

    # ============================================================
    # 6. COMPARE RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORTS")
    for name, data in results.items():
        print(f"\n--- {name} ---")
        y_pred = np.argmax(data["model"].predict(x_test, verbose=0), axis=1)
        print(classification_report(y_test_int, y_pred, digits=3))

    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<42} {'Test Acc':>10}")
    print("-" * 54)
    for name, data in results.items():
        print(f"{name:<42} {data['test_acc']*100:>9.2f}%")
    print("-" * 54)

    best_name = max(results, key=lambda k: results[k]["test_acc"])
    print(f"\nBEST MODEL: {best_name} — {results[best_name]['test_acc']*100:.2f}%")

    # ============================================================
    # 7. PLOT BIEU DO
    # ============================================================
    plot_all_histories(results)
    plot_comparison(results)

    # ============================================================
    # 8. SAVE BEST MODEL
    # ============================================================
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        results[best_name]["model"].save(filename)
        print(f"\nBest model ({best_name}) saved to {filename}.")


# ================================================================
# DATA LOADING
# ================================================================
def load_data(data_dir):
    """
    Load images from `data_dir/<category>/` for categories 0..NUM_CATEGORIES-1.
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
            # Only process image files; skip CSVs and other non-image files
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".ppm", ".bmp")):
                continue

            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)

            # Skip corrupt or unreadable files instead of crashing
            if img is None:
                print(f"Warning: could not read {img_path}, skipping.", file=sys.stderr)
                skipped += 1
                continue

            # Convert BGR (OpenCV default) to RGB to match inference pipeline
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Normalise pixels to [0, 1] — done exactly once here
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(category)

    if skipped:
        print(f"Skipped {skipped} unreadable file(s).", file=sys.stderr)

    return images, labels


# ================================================================
# MODEL A: BASELINE CNN
# ================================================================
def build_baseline_cnn():
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

        # Flatten
        tf.keras.layers.Flatten(),

        # Fully-connected head
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer — one unit per category
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ================================================================
# MODEL B: CNN WITH L2 REGULARIZATION
# ================================================================
def build_cnn_with_l2():
    from tensorflow.keras import regularizers
    l2 = regularizers.l2(0.001)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # --- Convolutional Block 1 (5x5 filters for general features) ---
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu", kernel_regularizer=l2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # --- Convolutional Block 2 (3x3 filters for specific features) ---
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Flatten
        tf.keras.layers.Flatten(),

        # Fully-connected head
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ================================================================
# MODEL C: DEEPER CNN (3 Double-Conv Blocks + BatchNorm + per-block Dropout)
# Each block has 2 Conv layers (VGG-style) + BN + Dropout — unlike Baseline CNN
# which has only 1 Conv per block and no BN.
# Spatial flow with 30x30 input: 30 -> 15 -> 7 -> 3 (3x3x128 = 1152 features)
# ================================================================
def build_deeper_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # --- Block 1: 30x30 -> 15x15 (double conv) ---
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # --- Block 2: 15x15 -> 7x7 (double conv) ---
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # --- Block 3: 7x7 -> 3x3 (double conv) ---
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Flatten: 3x3x128 = 1152 spatial features
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ================================================================
# MODEL D: TRANSFER LEARNING WITH MobileNetV2
# ================================================================
def build_transfer_learning_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    # Unfreeze the last 20 layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    x = tf.keras.layers.Resizing(96, 96)(inputs)
    # MobileNetV2 expects input in [-1, 1]; preprocess_input rescales from [0,255]
    # Our images are already in [0,1], so multiply back to [0,255] before preprocessing
    x = tf.keras.layers.Lambda(lambda t: mv2_preprocess(t * 255.0))(x)
    # training=False keeps frozen BatchNorm layers in inference mode,
    # preventing them from updating statistics and destabilising training
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ================================================================
# VISUALIZATION
# ================================================================
def plot_all_histories(results):
    """
    Plot training accuracy and loss per epoch for every model.
    Saves to results/all_models_curves.png
    """
    os.makedirs("results", exist_ok=True)
    names = list(results.keys())
    n = len(names)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    fig.suptitle("Training Curves - All Models", fontsize=14, fontweight="bold")

    for col, name in enumerate(names):
        hist = results[name]["history"].history
        epochs_range = range(1, len(hist["accuracy"]) + 1)

        # Row 1: Accuracy
        axes[0, col].plot(epochs_range, hist["accuracy"], marker="o", label="Train Acc")
        axes[0, col].set_title(f"{name}\nAccuracy")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].set_ylabel("Accuracy")
        axes[0, col].legend()
        axes[0, col].grid(True)

        # Row 2: Loss
        axes[1, col].plot(epochs_range, hist["loss"], marker="o",
                          color="orange", label="Train Loss")
        axes[1, col].set_title(f"{name}\nLoss")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].set_ylabel("Loss")
        axes[1, col].legend()
        axes[1, col].grid(True)

    plt.tight_layout()
    save_path = os.path.join("results", "all_models_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"All-model training curves saved to {save_path}")


def plot_comparison(results):
    """
    Bar chart comparing test accuracy across all trained models.
    Saves to results/comparison.png
    """
    os.makedirs("results", exist_ok=True)
    names = list(results.keys())
    accs  = [results[n]["test_acc"] * 100 for n in names]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, accs, color=colors[:len(names)], edgecolor="white", linewidth=1.2)
    plt.ylim(max(0, min(accs) - 5), 100)
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Comparison — GTSRB Test Accuracy", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right")

    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=10
        )

    plt.tight_layout()
    save_path = os.path.join("results", "comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to {save_path}")


if __name__ == "__main__":
    main()