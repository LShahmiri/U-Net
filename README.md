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
📊 Dataset Format
Training data (/mask-trin-path/)

Each sample is in its own folder:
├── main.py # Main script (training + prediction pipeline)
├── logs/ # TensorBoard logs
├── model_for_butterfly_unet.h5 # Saved trained model (created after training)
├── mask-output-path-/ # Predicted segmentation masks for test data
├── /mask-trin-path/ # Training dataset (images + masks)
└── /test-path/ # Test dataset (images only)

---

## 🛠 Requirements
Install dependencies via pip:

```bash
pip install tensorflow numpy scikit-image matplotlib tqdm
📊 Dataset Format
Training data (/mask-trin-path/)

Each sample is in its own folder:
train_id/
 ├── images/
 │    └── train_id.png
 └── masks/
      ├── mask1.png
      ├── mask2.png
      └── ...
Test data (/test-path/)

Each test sample folder contains only an image:
test_id/
 └── images/
      └── test_id.png
🏃 Usage

1- Prepare your dataset in the structure described above.

2- Run the script to train and predict:
   python main.py
3- After training:

   -Best model saved as model_for_butterfly_unet.h5.

   -Predictions saved in mask-output-path-/test_id/test_id_pred.png.

📈 Example Outputs

During training and validation, sanity check plots are displayed:

Train Image

Ground Truth Mask

Predicted Mask

At inference, binary masks are saved for each test image.
