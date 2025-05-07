# 🛰️ Skyview Aerial Landscape Classification

![Aerial Landscape Classification](https://img.shields.io/badge/Computer%20Vision-Aerial%20Classification-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-blue)
![Dataset](https://img.shields.io/badge/Dataset-Skyview-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.4%25-success)

A state-of-the-art deep learning model for classifying aerial landscapes using PyTorch and transfer learning.

## 📊 Overview

This repository contains a ResNet18-based model that classifies aerial images into 15 different landscape categories with **98.4% validation accuracy**. The model was trained on the Skyview Multi-Landscape Aerial Imagery Dataset, which contains 12,000 high-quality aerial images across diverse landscape types.

## 🗂️ Dataset

The Skyview dataset includes 15 landscape categories with 800 images per category (12,000 total):

- 🌾 Agriculture
- ✈️ Airport
- 🏖️ Beach
- 🏙️ City
- 🏜️ Desert
- 🌲 Forest
- 🌿 Grassland
- 🛣️ Highway
- 🌊 Lake
- ⛰️ Mountain
- 🅿️ Parking
- 🚢 Port
- 🚂 Railway
- 🏘️ Residential
- 🏞️ River

Each image has a resolution of 256×256 pixels and was sourced from the AID and NWPU-Resisc45 datasets.

![image](https://github.com/user-attachments/assets/3b92181c-2eba-4a0a-b205-3622e0cb604b)


## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| ✅ Validation Accuracy | 98.4% |
| 📉 Training Loss | 0.0769 |
| 📉 Validation Loss | 0.0523 |
| 🧠 Model Architecture | ResNet18 (Transfer Learning) |
| ⏱️ Training Time | ~45 minutes (20 epochs) |

![image](https://github.com/user-attachments/assets/94fd0ee5-b11f-4ce8-bb01-32a1e9d84f52)

## 🔧 Model Architecture

The model uses a pre-trained ResNet18 backbone with a custom classifier head fine-tuned for the Skyview dataset:

- 🖼️ Input: 224×224 RGB images
- 🏗️ Backbone: ResNet18 (pretrained on ImageNet)
- 🧩 Final layer: Fully connected (512 → 15 classes)
- ⚙️ Optimizer: Adam (lr=0.001)
- 📊 Loss function: CrossEntropyLoss

## 📋 Requirements

```
numpy>=1.19.2
pandas>=1.1.3
torch>=1.10.0
torchvision>=0.11.1
matplotlib>=3.3.2
Pillow>=8.0.1
tqdm>=4.50.2
```

## 🎯 Results Visualization

![image](https://github.com/user-attachments/assets/7748cdb6-c3f3-4ad0-a727-347251c29005)

## 💡 Key Findings

- 🔄 Transfer learning from ImageNet provides an excellent starting point for aerial imagery classification
- 🛑 Early stopping helps prevent overfitting while maintaining high accuracy
- 🔁 Data augmentation significantly improves model generalization
- 🎉 The model achieves near-perfect accuracy on most categories, with occasional confusion between visually similar categories

## 📝 Citation

If you use this model or implementation in your research, please cite:

```
@misc{aerial-classification,
  author = {Yash Kumar},
  title = {Aerial Landscape Classification},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yashkc2025/aerial_landscape_classification}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The Skyview dataset is derived from the [AID Dataset](https://captain-whu.github.io/AID/) and [NWPU-Resisc45 Dataset](https://paperswithcode.com/dataset/resisc45)
- Implementation based on PyTorch's transfer learning tutorials
