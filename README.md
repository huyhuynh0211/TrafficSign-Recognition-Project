# Traffic Sign Recognition using CNN on GTSRB

This project implements a **German traffic sign recognition system** using **Convolutional Neural Networks (CNNs)** with **TensorFlow/Keras** on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The goal of the project is to classify traffic sign images into **43 different categories**, such as speed limits, yield, stop, priority road, no passing, road work, and other common German traffic signs.

---

## 1. Introduction

Traffic sign recognition is an important computer vision task in:

- Autonomous vehicles
- Advanced Driver Assistance Systems (ADAS)
- Intelligent transportation systems
- Real-time road scene understanding

This project builds a complete deep learning pipeline including:

1. Loading and preprocessing image data.
2. Resizing images to a fixed input size.
3. Normalizing pixel values to the range `[0, 1]`.
4. Handling class imbalance using `class weights`.
5. Training CNN-based models.
6. Comparing multiple model architectures.
7. Evaluating models using accuracy, precision, recall, and F1-score.
8. Performing inference on a single image or a folder of images.

Requirements:
pip install opencv-python numpy scikit-learn matplotlib pandas
pip install tensorflow-cpu==2.16.2

• Python: 3.12.10
• TensorFlow: 2.16.2 (CPU)
• Numpy Version: 1.26.4
---

## 2. Dataset

The project uses the **GTSRB - German Traffic Sign Recognition Benchmark** dataset.

Main information:

| Property | Value |
|---|---|
| Number of classes | 43 |
| Image format | PNG / PPM |
| Image type | RGB color images |
| Original image size | Variable |
| Processed image size | `30 x 30 x 3` |
| Standard test set | 12,630 images |

The training dataset is usually organized as:

```text
gtsrb/
├── 0/
├── 1/
├── 2/
├── ...
└── 42/
```

Each subfolder represents one traffic sign class.

Examples:

```text
0  -> Speed limit 20 km/h
1  -> Speed limit 30 km/h
2  -> Speed limit 50 km/h
...
42 -> End of no passing by vehicles over 3.5 metric tons
```

---

## 3. Project Structure

A recommended project structure is:

```text
traffic-sign-recognition/
├── trafficsign.py
├── trafficsign_new.py
├── test_models.py
├── test_model_folder.py
├── gtsrb/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 42/
├── Test/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── labels.csv
├── results/
├── best_model.h5
└── README.md
```

Description of important files:

| File | Description |
|---|---|
| `trafficsign.py` | Trains the baseline CNN model |
| `trafficsign_new.py` | Trains and compares multiple models |
| `test_models.py` | Predicts one single image |
| `test_model_folder.py` | Predicts all images in a folder |
| `labels.csv` | Ground-truth labels used for testing |
| `best_model.h5` | Saved trained model |
| `results/` | Stores plots, reports, and experiment results |

---

## 4. Environment Setup

Recommended Python version:

```text
Python 3.10 or newer
```

Install required libraries:

```bash
pip install opencv-python numpy scikit-learn matplotlib pandas
pip install tensorflow-cpu==2.16.2
```

If your machine has a properly configured GPU environment, you may install the GPU-supported TensorFlow version instead of `tensorflow-cpu`.

---

## 5. How to Run

### 5.1. Train the baseline CNN model

```bash
python trafficsign.py gtsrb model.keras
```

This script will:

- Load images from the `gtsrb/` folder.
- Resize all images to `30 x 30`.
- Normalize pixel values to `[0, 1]`.
- Compute class weights.
- Train a CNN model.
- Print accuracy and loss.
- Save training curves into the `results/` folder.

---

### 5.2. Train and compare multiple models

```bash
python trafficsign_new.py gtsrb
```

The advanced script compares the following models:

| Model | Description |
|---|---|
| Model A | Baseline CNN + Class Weights |
| Model B | CNN + L2 Regularization + Class Weights |
| Model C | Deeper CNN + Batch Normalization |
| Model D | MobileNetV2 Transfer Learning |

---

### 5.3. Train and save the best model

```bash
python trafficsign_new.py gtsrb best_model.keras
```

After training, the best model will be saved as:

```text
best_model.keras
```

---

### 5.4. Predict a single image

```bash
python test_models.py best_model.keras image.png
```

Example:

```bash
python test_models.py best_model.keras Test/00000.png
```

The output usually includes:

- Top-1 predicted class
- Top-3 predicted classes
- Confidence scores
- Traffic sign name

---

### 5.5. Predict all images in a folder

```bash
python test_model_folder.py best_model.keras test_images/ --csv labels.csv
```

Example:

```bash
python test_model_folder.py best_model.keras Test_2/ --csv gtsrb_test_labels_08184_to_08210.csv
```

The script will:

- Read all images in the folder.
- Predict each image.
- Compare predictions with the ground-truth labels from the CSV file.
- Print `PASS` or `FAIL` for each image.
- Calculate the overall accuracy.

---

## 6. Data Preprocessing

Each image is processed using the following steps:

1. Read the image with OpenCV.
2. Convert from BGR to RGB.
3. Resize to `30 x 30`.
4. Convert the image to `float32`.
5. Normalize pixel values:

```python
img = img.astype(np.float32) / 255.0
```

Normalization helps the model train more stably and converge faster.

---

## 7. Handling Class Imbalance

The GTSRB dataset is imbalanced. Some classes contain many images, while others contain far fewer samples.

This project uses `class_weight` to assign higher weights to minority classes:

```python
from sklearn.utils import class_weight

class_weights_array = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_int),
    y=labels_int
)

class_weight_dict = {
    i: class_weights_array[i]
    for i in range(NUM_CATEGORIES)
}
```

Formula:

```text
w_c = N_total / (N_classes * N_c)
```

Where:

- `N_total` is the total number of images.
- `N_classes` is the number of classes.
- `N_c` is the number of images in class `c`.

A class with fewer images receives a larger weight during training.

---

## 8. Data Augmentation

In the advanced training script, data augmentation is applied using `ImageDataGenerator`:

```python
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode="nearest"
)
```

Important note:

```python
horizontal_flip=False
```

Traffic signs should not be flipped horizontally because flipping may change the meaning of a sign, for example turning a right-turn sign into a left-turn sign.

---

## 9. Baseline CNN Architecture

The baseline CNN architecture is:

```text
Input: 30 x 30 x 3

Conv2D 32 filters, 5x5, ReLU
Conv2D 32 filters, 5x5, ReLU
MaxPooling2D 2x2
Dropout 0.25

Conv2D 64 filters, 3x3, ReLU
Conv2D 64 filters, 3x3, ReLU
MaxPooling2D 2x2
Dropout 0.25

Flatten
Dense 256, ReLU
Dropout 0.5
Dense 43, Softmax
```

The final output is a probability vector of length 43, corresponding to the 43 traffic sign classes.

---

## 10. Experimental Results

Main experimental results:

| Model | Test Accuracy |
|---|---:|
| `trafficsign.py` - Baseline CNN | 99.20% |
| Model A - Baseline CNN + Class Weights | 99.19% |
| Model B - CNN + L2 Regularization | 99.14% |
| Model C - Deeper CNN + BatchNorm | 99.96% |
| Model D - MobileNetV2 Transfer Learning | 98.48% |

Inference results on the full GTSRB standard test set of 12,630 images:

| Model | Correct Images | Total Images | Accuracy |
|---|---:|---:|---:|
| `trafficsign.py` - Baseline CNN | 12,146 | 12,630 | 96.17% |
| Model C - Deeper CNN + BatchNorm | 12,392 | 12,630 | 98.12% |

Observations:

- Model C achieved the highest validation accuracy in the comparison experiment.
- Model A generalized better on the full standard GTSRB test set.
- Class weights helped improve learning for minority classes.
- Data augmentation improved robustness when testing on real-world images.

---

## 11. Test Label CSV Format

When testing a folder of images, the CSV file should have the following format:

```csv
file,Path,ClassId,class_name,label_en,label_vi
00000.png,Test/00000.png,16,vehicles_over_3_5_tons_prohibited,Vehicles over 3.5 metric tons prohibited,Cam xe tren 3.5 tan
00001.png,Test/00001.png,1,speed_limit_30,Speed limit 30 km/h,Gioi han toc do 30 km/h
```

Column description:

| Column | Meaning |
|---|---|
| `file` | Image filename |
| `Path` | Image path in the dataset |
| `ClassId` | Ground-truth class ID from 0 to 42 |
| `class_name` | Code-friendly class name |
| `label_en` | English traffic sign label |
| `label_vi` | Vietnamese traffic sign label |

---

## 12. GTSRB Class List

| ClassId | English Label |
|---:|---|
| 0 | Speed limit 20 km/h |
| 1 | Speed limit 30 km/h |
| 2 | Speed limit 50 km/h |
| 3 | Speed limit 60 km/h |
| 4 | Speed limit 70 km/h |
| 5 | Speed limit 80 km/h |
| 6 | End of speed limit 80 km/h |
| 7 | Speed limit 100 km/h |
| 8 | Speed limit 120 km/h |
| 9 | No passing |
| 10 | No passing for vehicles over 3.5 metric tons |
| 11 | Right-of-way at the next intersection |
| 12 | Priority road |
| 13 | Yield |
| 14 | Stop |
| 15 | No vehicles |
| 16 | Vehicles over 3.5 metric tons prohibited |
| 17 | No entry |
| 18 | General caution |
| 19 | Dangerous curve to the left |
| 20 | Dangerous curve to the right |
| 21 | Double curve |
| 22 | Bumpy road |
| 23 | Slippery road |
| 24 | Road narrows on the right |
| 25 | Road work |
| 26 | Traffic signals |
| 27 | Pedestrians |
| 28 | Children crossing |
| 29 | Bicycles crossing |
| 30 | Beware of ice/snow |
| 31 | Wild animals crossing |
| 32 | End of all speed and passing limits |
| 33 | Turn right ahead |
| 34 | Turn left ahead |
| 35 | Ahead only |
| 36 | Go straight or right |
| 37 | Go straight or left |
| 38 | Keep right |
| 39 | Keep left |
| 40 | Roundabout mandatory |
| 41 | End of no passing |
| 42 | End of no passing by vehicles over 3.5 metric tons |

---

## 13. GitHub Upload Notes

Avoid uploading large files such as:

```text
*.h5
*.keras
gtsrb/
Train/
Test/
```

If you want to exclude datasets and large model files, create a `.gitignore` file:

```gitignore
gtsrb/
Train/
Test/
*.h5
*.keras
*.zip
__pycache__/
results/
```

If you must upload large models or datasets, use **Git LFS**.

---

## 15. References

- German Traffic Sign Recognition Benchmark, GTSRB
- TensorFlow / Keras Documentation
- OpenCV Documentation
- Scikit-learn Documentation

