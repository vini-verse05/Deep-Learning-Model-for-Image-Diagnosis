# 🧠 Deep Learning Model Using Image Diagnosis

An end-to-end AI-powered medical diagnosis system for brain tumor detection from MRI scans — accurate, explainable, and secure.

---

## 📌 Overview

This project is a **Deep Learning-based web application** that detects brain tumors from MRI images using a Convolutional Neural Network (CNN) built on **ResNet50 via transfer learning**. The system provides binary classification (Healthy / Diseased), visual explainability through Grad-CAM heatmaps, and AES-256 encryption for secure medical data handling.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🧠 Brain Tumor Detection | Classifies MRI images as Healthy or Diseased (tumor present) |
| 🔬 Transfer Learning | ResNet50-based CNN with custom classification layers |
| 🌡️ Grad-CAM Visualization | Heatmap overlay showing regions influencing the model's prediction |
| 🔐 AES-256 Encryption | All uploaded MRI images are encrypted before storage |
| 🔑 Secure Login System | User authentication to restrict access to medical data |
| 🌐 Web-Based UI | Flask-powered interface for image upload, results, and heatmap display |

---

## 🖥️ Demo Flow

```
User uploads MRI → Image encrypted (AES-256) → Preprocessed (224×224, normalized)
    → ResNet50 CNN → Sigmoid prediction → Confidence score + Grad-CAM heatmap
```

---

## 🏗️ Project Structure

```
Deep_Learning_Model_For_Image_Diagnosis/
│
├── backend/
│   └── app.py                      # Flask backend — routes, upload, predict, display
│
├── model/
│   ├── cnn_model.py                # Model architecture: ResNet50 + custom dense layers
│   ├── train_model.py              # Training pipeline with callbacks (EarlyStopping, etc.)
│   └── brain_tumor_model.h5        # Saved trained model weights
│
├── utils/
│   └── preprocessing.py            # Image preprocessing and augmented data generators
│
├── security/
│   └── aes_encryption.py           # AES-256 image encryption/decryption
│
├── explainability/
│   └── gradcam.py                  # Grad-CAM heatmap generation
│
├── templates/
│   ├── login.html                  # Secure login page
│   └── index.html                  # Main UI — upload, prediction, heatmap display
│
├── static/
│   └── heatmaps/                   # Generated Grad-CAM heatmap images
│
├── dataset/                        # MRI training and test images
│
├── evaluate_metrics.py             # Accuracy, sensitivity, specificity, F1, AUC-ROC
└── requirements.txt
```

---

## 🧬 Model Architecture

- **Base Model:** ResNet50 (pretrained on ImageNet, top layers removed)
- **Custom Layers:**
  - GlobalAveragePooling2D
  - Dense (256, ReLU) + Dropout (0.5)
  - Dense (1, **Sigmoid**) — binary classification output
- **Task:** Binary classification → `0 = Healthy`, `1 = Diseased`
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam

---

## 🔄 Preprocessing Pipeline

- Input images resized to **224 × 224 pixels**
- Pixel values normalized to **[0, 1]**
- **Data Augmentation:** Horizontal/vertical flip, rotation, zoom, shear
- **Class Balancing:** Weighted class balancing to handle imbalanced datasets
- Generators built using `ImageDataGenerator` from Keras

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Sensitivity (Recall)** | True positive rate — correctly identified diseased cases |
| **Specificity** | True negative rate — correctly identified healthy cases |
| **F1-Score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Area Under the ROC Curve — overall discriminative ability |

All metrics computed in `evaluate_metrics.py`.

---

## 🌡️ Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) generates a **heatmap** overlaid on the original MRI image to highlight the regions that most influenced the model's prediction.

- Target layer: Last convolutional layer of ResNet50
- Output: Color-coded heatmap (red = high activation, blue = low)
- Displayed alongside prediction and confidence score in the UI

---

## 🔐 Security

- **AES-256 encryption** applied to all uploaded MRI images before storage
- Images are decrypted on-the-fly only during inference
- **Secure login system** restricts access to authenticated users only
- Encryption handled in `aes_encryption.py` using the `PyCryptodome` library

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8+
- pip

### Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 📦 Requirements

```
tensorflow>=2.10
keras
flask
opencv-python
numpy
scikit-learn
matplotlib
pycryptodome
Pillow
```

> See `requirements.txt` for full list with pinned versions.

---

## 🧪 Training the Model

```bash
python train_model.py
```

- Callbacks used: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`
- Trained model saved to `models/brain_tumor_model.h5`

---

## 📈 Evaluating the Model

```bash
python evaluate_metrics.py
```

Outputs accuracy, sensitivity, specificity, F1-score, and AUC-ROC on the test set.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Backend | Flask (Python) |
| Frontend | HTML, CSS, JavaScript |
| Encryption | PyCryptodome (AES-256) |
| Explainability | Grad-CAM |

---

## 📸 UI Overview

The web interface allows users to:
1. Log in securely
2. Upload an MRI image
3. View the **prediction** (Healthy / Diseased)
4. View the **confidence score**
5. View the **Grad-CAM heatmap** highlighting tumor-relevant regions

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🧠 Acknowledgements

- [ResNet50 — He et al., 2015](https://arxiv.org/abs/1512.03385)
- [Grad-CAM — Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
- [Kaggle Brain MRI Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

---

> **Disclaimer:** This tool is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis.
