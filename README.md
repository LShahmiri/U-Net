# U-Net with MobileNetV2 Encoder for Image Segmentation

This repository implements an image segmentation pipeline using **U-Net** with a **MobileNetV2 encoder** pretrained on ImageNet.  
The model is designed for binary semantic segmentation tasks (e.g., foreground–background separation such as butterflies vs. background).

---

## 🚀 Features
- **Preprocessing & Data Loading**:
  - Resizes training/test images and masks to `512x512`.
  - Combines multiple instance masks into a single binary mask.
  - Preserves original test image sizes for prediction resizing.

- **Model Architecture**:
  - U-Net decoder built on top of a pretrained **MobileNetV2** encoder.
  - Skip connections for multi-scale feature fusion.
  - Final sigmoid output for binary masks.

- **Training**:
  - Dice coefficient + Binary Cross-Entropy (BCE) combined loss.
  - Training with encoder frozen, followed by fine-tuning (unfreezing all layers).
  - Uses callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard.

- **Evaluation & Prediction**:
  - Train/validation split (90/10).
  - Sanity check plots for images, masks, and predictions.
  - Saves test predictions resized back to their **original dimensions**.

---

## 📂 Project Structure
