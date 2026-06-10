# ✋ Alphabetical Hand Gesture Recognition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prakas-Dutta/Hand-gesture-to-alphabet/blob/main/Alphabetical_Hand_gesture_recognition_using_deep_learning.ipynb)

A classical machine learning project that recognizes hand gestures corresponding to all 26 English alphabets (A–Z) from grayscale images. Instead of relying on deep learning or pre-trained models, this project builds and compares multiple classifiers — including a custom KMeans-prototype model built from scratch using NumPy.

---

## 📌 Project Highlights

- Recognizes all **26 alphabetical hand gestures** (A–Z)
- Implements **3 different classifiers** and compares their performance
- Includes a **fully custom classifier** built without any sklearn estimator
- Uses **Incremental PCA** for memory-efficient dimensionality reduction on large image datasets
- Full pipeline: raw images → grayscale → CSV → PCA-reduced features → model training → evaluation

---

## 🧠 Models Implemented

| Model | Type | Notes |
|---|---|---|
| **SVM (RBF kernel)** | Sklearn | `C=100`, `gamma=5`, One-vs-Rest |
| **Random Forest** | Sklearn | `n_estimators=100`, `max_leaf_nodes=30` |
| **Custom Prototype Classifier** | Built from scratch | KMeans-based class prototypes + Euclidean distance voting |

### Custom Classifier Design
The custom model builds class prototypes using KMeans clustering (k=8 per class) over the PCA-reduced feature space. Prediction is done by finding the nearest prototype via Euclidean distance — no sklearn estimator involved.

---

## 🔄 Pipeline Overview

```
Raw Images (alphaset/)
        ↓
Grayscale Conversion (OpenCV)
        ↓
Flatten pixels → images.csv
        ↓
StandardScaler + Incremental PCA (300 components)
        ↓
images_reduced.csv
        ↓
Train/Test Split (80/20)
        ↓
SVM  |  Random Forest  |  Custom KMeans Classifier
        ↓
Accuracy Evaluation + Learning Curves
```

---

## 📁 Repository Structure

```
Hand-gesture-to-alphabet/
│
├── Alphabetical_Hand_gesture_recognition_with_custom_model.ipynb   # Main notebook
├── svm_classifier.pkl          # Saved SVM model
├── random_forest_classifier.pkl  # Saved Random Forest model
└── README.md
```

> **Note:** The dataset (`alphaset/`) and generated CSVs (`images.csv`, `images_reduced.csv`) are not included in the repo due to size. See the Dataset section below.

---

## 📦 Dataset

The dataset consists of labeled hand gesture images for each alphabet (A–Z), organized in class-wise folders under `alphaset/`.

- Each image is converted to **grayscale** and **flattened** into a pixel vector
- Labels are stored as **ASCII codes** (65–90 for A–Z)
- Dataset is loaded from **Google Drive** in the notebook

You can use any standard ASL (American Sign Language) alphabet dataset or collect your own images.

---

## ⚙️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Prakas-Dutta/Hand-gesture-to-alphabet.git
cd Hand-gesture-to-alphabet
```

### 2. Install dependencies

```bash
pip install scikit-learn numpy pandas matplotlib opencv-python
```

### 3. Run on Google Colab (Recommended)

Click the **Open in Colab** badge at the top, mount your Google Drive, place the `alphaset/` folder at the root of your Drive, and run all cells sequentially.

---

## 🔧 Key Techniques

- **Incremental PCA** — processes the dataset in chunks to avoid memory overflow; reduces image features to 300 principal components
- **StandardScaler** — applied before PCA for zero mean and unit variance
- **Learning Curves** — plotted for both Random Forest and the custom classifier to analyze bias-variance behaviour
- **Pickle serialization** — SVM and Random Forest models are saved for reuse

---

## 📊 Results

Learning curves are generated for the Random Forest and the custom KMeans-prototype classifier to visualize training vs. validation accuracy across different training set sizes.

> Accuracy results depend on the dataset used. Run the notebook to reproduce.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** NumPy, Pandas, OpenCV, Scikit-learn, Matplotlib
- **Environment:** Google Colab + Google Drive

---

## 👤 Author

**Prakas Dutta**  
MCA, University of Kalyani | GATE CSE 2026  
[GitHub](https://github.com/Prakas-Dutta)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
