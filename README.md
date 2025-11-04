````markdown
 DeepRetina: Automated Detection of Diabetic Retinopathy and Glaucoma using Retinal Fundus Images

DeepRetina is a deep learning-based system for automated detection of Diabetic Retinopathy (DR) and Glaucoma using retinal fundus images. The project uses EfficientNet-B3 and EfficientNet-B4 backbones enhanced with Squeeze-and-Excitation (SE) attention, CLAHE preprocessing, and robust augmentation to achieve high diagnostic performance. This repository contains trained models, preprocessing and training scripts, and a demo for interactive inference. Note: this version performs DR detection (not DR staging).

---

 Overview

Retinal diseases such as Diabetic Retinopathy (DR) and Glaucoma are leading causes of irreversible blindness. DeepRetina provides an end-to-end pipeline for automatic screening from fundus images with the following key components:

- EfficientNet-B4 backbone with SE attention  
- CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing  
- Augmentations: flips, rotations, brightness/contrast, MixUp, CutMix  
- Optimized training: AdamW optimizer, cosine LR decay, label smoothing  
- Test-Time Augmentation (TTA) at inference  
- Achieved ~96.96% test accuracy on the cleaned dataset (4-class: Cataract, DR, Glaucoma, Normal)

---

 Authors

- Arohi Agrawal (T009)  
- Divyendra Singh (T022)  
- Kashish Wadhwani (T037)  

Supervised by: Dr. Raj Gaurav Mishra, Assistant Professor, MPSTME, NMIMS University

---

 Repository Structure

- DeepRetina/  
  - models/  
    - efficientnet_b3_model.h5  
    - efficientnet_b4_with_duplicates.h5  
    - efficientnet_b4_cleaned.h5  
  - demo/  
    - deepretina_demo.py  
  - preprocessing/  
    - clahe_preprocess.py  
    - data_augmentation.py  
    - duplicate_filtering.py  
  - training/  
    - train_b3.py  
    - train_b4.py  
    - train_dual_head_placeholder.py  
  - requirements.txt  
  - README.md

---

 Datasets Used

- **Eye Diseases Classification** — Diabetic Retinopathy and Glaucoma classification  
  https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

All datasets were cleaned using MD5 and Perceptual Hashing (pHash) to remove duplicates and ensure data integrity.

---

 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/DeepRetina.git
   cd DeepRetina
````

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   # Linux / macOS
   source venv/bin/activate
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # Windows (cmd)
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

 Running the Demo

To run the interactive demo:

```bash
cd demo
python deepretina_demo.py
```

The demo displays a grid of sample fundus images. Enter an image index (0–11) to run inference. The demo prints the true label, predicted label, confidence score, and class probabilities.

---

## Model Details

| Model                             | Description                                   | Accuracy |
| --------------------------------- | --------------------------------------------- | -------- |
| EfficientNet-B3                   | Baseline pretrained model                     | ~95%     |
| EfficientNet-B4 (with duplicates) | Early experimental version                    | ~96%     |
| EfficientNet-B4 (cleaned)         | Final model trained on duplicate-free dataset | 96.96%   |

Architecture summary:

* CLAHE preprocessing
* MixUp, CutMix, and label smoothing for regularization
* SE attention block integrated into backbone
* AdamW optimizer with cosine LR schedule
* Test-Time Augmentation (TTA) during inference

---

## Results Summary

* Test Accuracy: **96.96%**
* Macro-Average AUROC: **0.97**

| Class    | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| Cataract | 0.974     | 0.993  | 0.983    |
| DR       | 1.000     | 1.000  | 1.000    |
| Glaucoma | 0.951     | 0.932  | 0.941    |
| Normal   | 0.950     | 0.950  | 0.950    |

---

## How to Retrain

To train from scratch:

```bash
cd training
python train_b4.py
```

Training script steps:

* Loads preprocessed images from dataset folders
* Applies CLAHE and augmentations (MixUp/CutMix)
* Trains EfficientNet-B4 with SE attention
* Logs metrics and saves best weights to `/models/`

---

## Future Work

* Implement a dual-head EfficientNet-B4 for independent DR and Glaucoma outputs
* Add explainability (Grad-CAM, SHAP) for clinical interpretability
* Expand dataset diversity and perform external validation
* Convert models for deployment (TensorFlow Lite / ONNX) for edge/telemedicine use

---

## License

For academic and research use only. Developed as part of the B.Tech Computer Engineering Capstone Project (2025–2026) at NMIMS University.

---

## Citation

Arohi Agrawal, Divyendra Singh, Kashish Wadhwani. "DeepRetina: Automated Detection of Diabetic Retinopathy and Glaucoma using Retinal Fundus Images." Capstone Project Report, MPSTME, NMIMS University, 2025.

