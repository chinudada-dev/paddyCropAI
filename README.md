# Crop Image Classifier

A lightweight image classification pipeline that identifies crop types from photos using a single-layer neural network built with TensorFlow 1.x and scikit-image.

---

## Overview

This script trains a fully connected neural network to classify images into one of **4 crop categories**. Images are loaded from a directory structure, resized to 28×28 pixels, and fed into a softmax classifier optimised with Adam.

---

## Directory Structure

```
Downloads/
├── crop_images_train/
│   ├── 3/          # Folder name = class label
│   │   ├── img1.jpg
│   │   └── ...
│   ├── 22/
│   ├── 36/
│   └── 100/
└── crop_images_test/
    ├── 3/
    ├── 22/
    ├── 36/
    └── 100/
```

Each subdirectory name is used directly as the integer class label.

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.6 – 3.7 |
| TensorFlow | 1.x |
| scikit-image | ≥ 0.14 |
| NumPy | ≥ 1.15 |
| Matplotlib | ≥ 2.0 |

Install dependencies:

```bash
pip install tensorflow==1.15 scikit-image numpy matplotlib
```

> **Note:** This script uses TF1-style APIs (`tf.placeholder`, `tf.contrib`, `tf.Session`). It is **not** compatible with TensorFlow 2.x without enabling legacy mode (`tf.compat.v1`).

---

## Usage

1. Place your training and test images under the directory structure shown above.
2. Update the root paths at the top of the script if needed:

```python
ROOT_PATH_1 = "/home/akapoor/Downloads/"   # train root
ROOT_PATH_2 = "/home/akapoor/Downloads/"   # test root
```

3. Run the script:

```bash
python crop_classifier.py
```

---

## Pipeline

```
Load JPGs  →  Resize to 28×28  →  Flatten  →  Fully Connected (ReLU)  →  Softmax (4 classes)
```

| Step | Detail |
|---|---|
| Input | RGB `.jpg` images of arbitrary size |
| Preprocessing | Resize to 28×28 via `skimage.transform.resize` |
| Model | Single fully connected layer → 4 output logits |
| Loss | Sparse softmax cross-entropy |
| Optimiser | Adam (`lr = 0.001`) |
| Training epochs | 201 |

---

## Output

- **Training:** Loss and accuracy logged every 10 epochs.
- **Sample predictions:** A 5×2 grid showing 10 randomly sampled test images with ground truth vs. predicted labels (green = correct, red = incorrect).
- **Final accuracy:** Printed as a float after evaluating the full test set.

---

## Known Limitations

- The model is a single dense layer with no convolution — accuracy will be limited for complex visual patterns.
- Test set evaluation uses `rgb2gray` conversion but training does not, causing a channel mismatch if not handled carefully.
- No data augmentation or regularisation is applied.
- Tied to TensorFlow 1.x legacy APIs.

---

## Potential Improvements

- Migrate to TensorFlow 2 / Keras for maintainability.
- Add convolutional layers (CNN) for better spatial feature extraction.
- Apply consistent grayscale or RGB preprocessing across train and test sets.
- Introduce train/validation split and early stopping.
- Use `tf.data` pipelines for scalable data loading.
