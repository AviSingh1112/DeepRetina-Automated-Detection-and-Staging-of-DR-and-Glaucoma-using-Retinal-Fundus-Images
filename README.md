Got it — here’s a clean, **single-block** `README.md` ready for direct upload to your GitHub repo.
This version correctly mentions **DR detection (not staging)** and includes dataset links, model usage, and authors.

---

```markdown
# 🧠 DeepRetina: Automated Detection of Diabetic Retinopathy and Glaucoma using Retinal Fundus Images

DeepRetina is a deep learning-based system for **automated detection of Diabetic Retinopathy (DR)** and **Glaucoma** using retinal fundus images.  
The project integrates **EfficientNet-B3** and **EfficientNet-B4** architectures enhanced with **Squeeze-and-Excitation (SE) attention**, advanced preprocessing, and robust augmentation techniques to achieve high diagnostic accuracy and clinical reliability.

---

## 📘 Overview

Retinal diseases such as **Diabetic Retinopathy (DR)** and **Glaucoma** are leading causes of irreversible blindness worldwide. This project aims to build an end-to-end AI pipeline capable of automatically detecting these diseases from fundus images with high precision.

Key highlights:
- **EfficientNet-B4 backbone** with Squeeze-and-Excitation (SE) Attention.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) preprocessing.
- **Advanced Augmentation:** Flip, rotation, brightness/contrast adjustments, MixUp, CutMix.
- **Optimized Training:** AdamW optimizer, cosine learning rate decay, label smoothing.
- **Test-Time Augmentation (TTA)** for stable inference.
- **~97% accuracy** on the cleaned dataset (4-class classification: Cataract, DR, Glaucoma, Normal).

---

## 👩‍💻 Authors

- **Arohi Agrawal (T009)**  
- **Divyendra Singh (T022)**  
- **Kashish Wadhwani (T037)**  

**Under the supervision of:**  
Dr. **Raj Gaurav Mishra**, Assistant Professor  
MUKESH PATEL SCHOOL OF TECHNOLOGY MANAGEMENT & ENGINEERING (MPSTME), NMIMS University, Mumbai  

---

## 📂 Repository Structure

```

DeepRetina/
│
├── models/
│   ├── efficientnet_b3_model.h5
│   ├── efficientnet_b4_with_duplicates.h5
│   └── efficientnet_b4_cleaned.h5
│
├── demo/
│   └── deepretina_demo.py
│
├── preprocessing/
│   ├── clahe_preprocess.py
│   ├── data_augmentation.py
│   └── duplicate_filtering.py
│
├── training/
│   ├── train_b3.py
│   ├── train_b4.py
│   └── train_dual_head_placeholder.py  # for future DR + Glaucoma specialization
│
├── requirements.txt
└── README.md

````

---

## 📊 Datasets Used

| Dataset | Purpose | Link |
|----------|----------|------|
| Eye Diseases Classification | Diabetic Retinopathy and Glaucoma classification | https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification |

All datasets were cleaned using MD5 and Perceptual Hashing (pHash) to remove duplicates and ensure data integrity.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/DeepRetina.git
cd DeepRetina
````

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Demo

To test the trained model interactively:

```bash
cd demo
python deepretina_demo.py
```

The demo displays a grid of sample fundus images.

* Enter an image index (0–11) to test prediction on that image.
* The model outputs:

  * **True label**
  * **Predicted label**
  * **Confidence score**
  * **Probability distribution**
* A green “✔ Correct!” message appears for correct predictions.

---

## 🧠 Model Details

| Model                                 | Description                                   | Accuracy   |
| ------------------------------------- | --------------------------------------------- | ---------- |
| **EfficientNet-B3**                   | Baseline pretrained model                     | ~95%       |
| **EfficientNet-B4 (with duplicates)** | Early experimental version                    | ~96%       |
| **EfficientNet-B4 (cleaned)**         | Final model trained on duplicate-free dataset | **96.96%** |

**Architecture Summary:**

* CLAHE preprocessing
* MixUp + CutMix + Label Smoothing
* SE Attention Block integrated in backbone
* AdamW optimizer with cosine LR schedule
* Test-Time Augmentation (TTA) at inference

---

## 🧩 Results Summary

**Test Accuracy:** 96.96%
**Macro-Average AUROC:** 0.97

| Class    | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Cataract | 0.974     | 0.993  | 0.983    |
| DR       | 1.000     | 1.000  | 1.000    |
| Glaucoma | 0.951     | 0.932  | 0.941    |
| Normal   | 0.950     | 0.950  | 0.950    |

---

## 🧪 How to Retrain

To train from scratch:

```bash
cd training
python train_b4.py
```

This will:

* Load preprocessed images from dataset folders
* Apply CLAHE, augmentations, and MixUp/CutMix
* Train EfficientNet-B4 with attention and log performance metrics
* Save best weights under `/models/`

---

## 📈 Future Work

* Implement **dual-head EfficientNet-B4** for independent DR and Glaucoma detection
* Add **Explainable AI (Grad-CAM / SHAP)** for clinical interpretability
* Expand dataset diversity with external validation
* Deploy via **TensorFlow Lite / ONNX** for real-time teleophthalmology

---

## 📜 License

This project is intended for **academic and research purposes only** under the
*B.Tech Computer Engineering Capstone Project (2025–2026)* at **NMIMS University**.

---

## 🏫 Citation

If you use this project in your research or publication, please cite:

> Arohi Agrawal, Divyendra Singh, Kashish Wadhwani,
> *"DeepRetina: Automated Detection of Diabetic Retinopathy and Glaucoma using Retinal Fundus Images"*,
> Capstone Project Report, MPSTME, NMIMS University, 2025.

---

```

---

Would you like me to also generate a minimal `requirements.txt` (TensorFlow, OpenCV, Albumentations, etc.) to include in your repo?
```
