# Suppress TensorFlow verbose logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import sys
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.get_logger().setLevel('ERROR')

EPOCHS = 30
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.2
BATCH_SIZE = 32


def main():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python trafficsign_improved.py data_directory [model.h5]")

    print("=" * 60)
    print("LOADING DATA...")
    images, labels = load_data(sys.argv[1])
    labels_int = np.array(labels)
    print(f"Loaded {len(images)} images across {NUM_CATEGORIES} classes.")

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    class_counts = {i: int(np.sum(labels_int == i)) for i in range(NUM_CATEGORIES)}
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)
    print(f"  Min class {min_class}: {class_counts[min_class]} images")
    print(f"  Max class {max_class}: {class_counts[max_class]} images")
    print(f"  Imbalance ratio: {class_counts[max_class] / class_counts[min_class]:.1f}:1")

    print("\n" + "=" * 60)
    print("COMPUTING CLASS WEIGHTS FOR IMBALANCED DATA")
    class_weights_array = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_int),
        y=labels_int
    )
    class_weight_dict = {i: class_weights_array[i] for i in range(NUM_CATEGORIES)}
    print(f"  Weight range: {min(class_weights_array):.3f} - {max(class_weights_array):.3f}")

    print("\n" + "=" * 60)
    print("PREPARING DATA")

    x_data = np.array(images, dtype=np.float32)
    labels_cat = tf.keras.utils.to_categorical(labels_int, NUM_CATEGORIES)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, labels_cat,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=labels_int
    )
    y_test_int = np.argmax(y_test, axis=1)

    print(f"  Train: {x_train.shape[0]} | Test: {x_test.shape[0]}")
    print(f"  Input shape: {x_train.shape[1:]}")
    results = {}

    def _train(name, model, use_cw=True):
        """Train a model with augmentation, EarlyStopping, and ReduceLROnPlateau."""
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {name}")
        model.summary()

        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            horizontal_flip=False,    # traffic signs không đối xứng ngang
            fill_mode="nearest"
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )

        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weight_dict if use_cw else None, 
            verbose=1
        )

        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"  Test Accuracy: {acc * 100:.2f}%")
        results[name] = {"model": model, "history": history, "test_acc": acc}

    _train("Baseline CNN + Class Weights",    build_baseline_cnn())
    _train("CNN + L2 Reg + Class Weights",    build_cnn_with_l2())
    _train("Deeper CNN (3 Blocks + BN)",      build_deeper_cnn())
    _train("Transfer Learning (MobileNetV2)", build_transfer_learning_model())

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORTS")
    for name, data in results.items():
        print(f"\n--- {name} ---")
        y_pred = np.argmax(data["model"].predict(x_test, verbose=0), axis=1)
        print(classification_report(y_test_int, y_pred, digits=3))

    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE (sorted by accuracy)")
    print("=" * 60)
    print(f"{'Model':<42} {'Test Acc':>10}")
    print("-" * 54)
    for name, data in sorted(results.items(), key=lambda kv: kv[1]["test_acc"], reverse=True):
        print(f"{name:<42} {data['test_acc']*100:>9.2f}%")
    print("-" * 54)

    best_name = max(results, key=lambda k: results[k]["test_acc"])
    print(f"\nBEST MODEL: {best_name} — {results[best_name]['test_acc']*100:.2f}%")

    plot_all_histories(results)
    plot_comparison(results)
    y_pred_best = np.argmax(results[best_name]["model"].predict(x_test, verbose=0), axis=1)
    plot_confusion_matrix(y_test_int, y_pred_best, best_name)  # [NEW]

    # 8. save best model
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        results[best_name]["model"].save(filename)
        print(f"\nBest model ({best_name}) saved to {filename}.")


# load data
def load_data(data_dir):
    """
    Load images từ `data_dir/<category>/` cho categories 0..NUM_CATEGORIES-1.
    Trả về (images, labels) với images là float32 RGB arrays, chuẩn hoá về [0, 1].
    """
    images, labels = [], []
    skipped = 0

    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_path):
            print(f"Warning: directory not found: {category_path}", file=sys.stderr)
            continue

        for filename in os.listdir(category_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".ppm", ".bmp")):
                continue

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


# model A
def build_baseline_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # Block 1: 5x5 filters for general features
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu"),
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 2: 3x3 filters for specific features
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# model B
def build_cnn_with_l2():
    from tensorflow.keras import regularizers
    l2 = regularizers.l2(0.001)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        tf.keras.layers.Conv2D(32, (5, 5), activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Conv2D(32, (5, 5), activation="relu", kernel_regularizer=l2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# model C
def build_deeper_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        # Block 1: 30x30 -> 15x15
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 2: 15x15 -> 7x7
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Block 3: 7x7 -> 3x3
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# model D
def build_transfer_learning_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    x = tf.keras.layers.Resizing(96, 96)(inputs)

   x = tf . keras . layers . Lambda ( lambda t : mv2_preprocess ( t * 255.0) ) (x)

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


# plot block
def plot_all_histories(results):
    """
    [FIX #2] Plot cả Training VÀ Validation curves — bản gốc chỉ có train!
    Việc thấy val_accuracy/loss là quan trọng để phát hiện overfitting.
    Saves to results/all_models_curves.png
    """
    os.makedirs("results", exist_ok=True)
    names = list(results.keys())
    n = len(names)

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    fig.suptitle("Training vs Validation Curves — All Models", fontsize=14, fontweight="bold")

    for col, name in enumerate(names):
        hist = results[name]["history"].history
        epochs_range = range(1, len(hist["accuracy"]) + 1)

        # Row 1: Accuracy — train + val
        axes[0, col].plot(epochs_range, hist["accuracy"],
                          marker="o", label="Train Acc")
        axes[0, col].plot(epochs_range, hist["val_accuracy"],
                          marker="s", linestyle="--", label="Val Acc")   
        axes[0, col].set_title(f"{name}\nAccuracy")
        axes[0, col].set_xlabel("Epoch")
        axes[0, col].set_ylabel("Accuracy")
        axes[0, col].legend()
        axes[0, col].grid(True)

        # Row 2: Loss — train + val
        axes[1, col].plot(epochs_range, hist["loss"],
                          marker="o", color="orange", label="Train Loss")
        axes[1, col].plot(epochs_range, hist["val_loss"],
                          marker="s", linestyle="--", color="red", label="Val Loss")
        axes[1, col].set_title(f"{name}\nLoss")
        axes[1, col].set_xlabel("Epoch")
        axes[1, col].set_ylabel("Loss")
        axes[1, col].legend()
        axes[1, col].grid(True)

    plt.tight_layout()
    save_path = os.path.join("results", "all_models_curves_2.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training/Validation curves saved to {save_path}")


def plot_comparison(results):
    """
    Bar chart so sánh test accuracy, sorted từ cao xuống thấp.
    Saves to results/comparison.png
    """
    os.makedirs("results", exist_ok=True)

    sorted_items = sorted(results.items(), key=lambda kv: kv[1]["test_acc"], reverse=True)
    names  = [k for k, _ in sorted_items]
    accs   = [v["test_acc"] * 100 for _, v in sorted_items]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, accs, color=colors[:len(names)], edgecolor="white", linewidth=1.2)
    plt.ylim(max(0, min(accs) - 5), 100)
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Comparison — GTSRB Test Accuracy (sorted)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right")

    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=10
        )

    plt.tight_layout()
    save_path = os.path.join("results", "comparison_2.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    [NEW] Heatmap confusion matrix cho model tốt nhất.
    Giúp xác định class nào hay bị nhầm lẫn nhất.
    Saves to results/confusion_matrix.png
    """
    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)

    ticks = np.arange(NUM_CATEGORIES)
    plt.xticks(ticks, ticks, fontsize=7, rotation=90)
    plt.yticks(ticks, ticks, fontsize=7)

    plt.tight_layout()
    save_path = os.path.join("results", "confusion_matrix_2.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    main()