A Comparative Evaluation of CNN and VisionTransformer Architectures for Brain Tumor Detection in MRI Scans

This repository contains all code, models, and materials for the research project titled **"A Comparative Evaluation of CNN and Vision Transformer Architectures for Brain Tumor Detection in MRI Scans
"**. The study evaluates the performance of convolutional neural networks (CNNs) and Vision Transformers (ViTs) in classifying various brain MRI scans for the detection of tumors. 

---

## Project Summary

This study evaluates the performance of convolutional neural networks (CNNs) and Vision Transformers (ViTs) in classifying various brain MRI scans for the detection of tumors. Model series such as EfficientNet, ConvNeXt, ViT, and SwinTransformer were trained on a publicly
available multiclass brain tumor dataset. To support experimentation and reproducibility, a custom GUI-based deep learning software was developed, enabling users to train models, configure parameters, apply data augmentation, monitor performance metrics, and generate diagnostic reports. A comparative analysis was conducted using accuracy, precision, recall, F1-score, AUROC, and confusion matrices. Results showed that ViT-B/16, EfficientNetB0, and ViT-B/32
achieved test accuracies of over 98 %. ViT-B/16 performed the best overall, demonstrating that larger model capacity and reduced patch size enhance feature extraction in brain MRI classification. EfficientNetB0 delivered strong performance despite its reduced complexity, demonstrating the strength of CNNs and the potential of transformer-based architectures for detecting brain tumors.


---

## Large Files / Git LFS
This repository uses Git Large File Storage (LFS) to manage large files such as trained model weights. 

**IMPORTANT: ** Before cloning, make sure Git LFS is installed, otherwise large files will not download properly. 

---


## Dataset

The [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) was used, containing T1-weighted contrast-enhanced images with labeled tumor types.

More details in `dataset_info.txt`.

> **Note:** The dataset is not included in this repo. The dataset can either be downloaded through the link above and placed in a folder named ```data``` or the training interface (```gui.py```) can automatically download the dataset in the local filespace. 

---
## Repository Layout

```
brain-mri-classifier
├── models/
│  ├── ConvNeXt (Base)
│    ├── NDA (No Data Augmentation)
│      └── convnext_nda.pth
│    └── WDA (With Data Augmentation)
│      └── convnext_wda.pth
│  ├── EfficientNet
│    ├── EfficientNetB0
│      ├── NDA (No Data Augmentation)
│        └── efnetb0_nda.pth
│      └── WDA (With Data Augmentation)
│        └── efnetb0_wda.pth
│    ├── EfficientNetB5
│    ├── EfficientNetB7
│  ├── ViT
│    ├── ViT-B/16
│    ├── ViT-B/32
│  ├── SwinTransformer (Base)
├── paper/
│    ├── paper.tex
│    ├── references.bib
│    ├── paper.pdf
├── reports /
│  ├── ConvNeXt (Base)
│    ├── NDA (No Data Augmentation)
│      └── convnext_report_nda.pdf
│    └── WDA (With Data Augmentation)
│      └── convnext_report_wda.pdf
│  ├── EfficientNet
│    ├── EfficientNetB0
│    ├── EfficientNetB5
│    ├── EfficientNetB7
├── dataset_info.txt
├── gui.py
├── requirements.txt

```


---

## Installation

### Clone the repository
```bash
git clone https://github.com/yourusername/vit-cnn-brain-mri.git
cd vit-cnn-brain-mri
