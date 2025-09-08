import os
import random
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

# ----------------------------
# Reproducibility
# ----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ----------------------------
# Config
# ----------------------------
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

TRAIN_PATH = '/mask-trin-path/'
TEST_PATH  = '/test-path/'
PRED_TEST_OUT = 'mask-output-path-/'
os.makedirs(PRED_TEST_OUT, exist_ok=True)

# ----------------------------
# Discover IDs
# ----------------------------
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids  = next(os.walk(TEST_PATH))[1]

# ----------------------------
# Load Training Data
# ----------------------------
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)  # store as float 0/1

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = os.path.join(TRAIN_PATH, id_)

    # Image
    img_path = os.path.join(path, 'images', f'{id_}.png')
    img = imread(img_path)[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, anti_aliasing=True)
    X_train[n] = img.astype(np.uint8)

    # Mask (union of all instance masks)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    masks_dir = os.path.join(path, 'masks')
    for mask_file in next(os.walk(masks_dir))[2]:
        m = imread(os.path.join(masks_dir, mask_file), as_gray=True)
        m = resize(m, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, anti_aliasing=False)
        m = np.expand_dims(m, axis=-1)
        mask = np.maximum(mask, m)

    # binarize to {0,1}
    mask = (mask > 0).astype(np.float32)
    Y_train[n] = mask

# ----------------------------
# Load Test Data (keep originals sizes)
# ----------------------------
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []  # (H, W)

print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = os.path.join(TEST_PATH, id_)
    img_path = os.path.join(path, 'images', f'{id_}.png')

    img = imread(img_path)[:, :, :IMG_CHANNELS]
    sizes_test.append((img.shape[0], img.shape[1]))  # original H, W
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True, anti_aliasing=True)
    X_test[n] = img.astype(np.uint8)

print('Done loading!')

# ----------------------------
# Quick sanity check (optional)
# ----------------------------
if len(train_ids) > 0:
    image_x = random.randint(0, len(train_ids)-1)
    plt.figure(); plt.title("Train Image"); plt.imshow(X_train[image_x]); plt.axis('off')
    plt.figure(); plt.title("Train Mask");  plt.imshow(np.squeeze(Y_train[image_x]), cmap='gray'); plt.axis('off')
    plt.show()

# ----------------------------
# Build Model: U-Net with MobileNetV2 encoder (pretrained on ImageNet)
# ----------------------------
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

# Use MobileNetV2 preprocessing: scales to [-1, 1]
preproc = tf.keras.applications.mobilenet_v2.preprocess_input
x = tf.keras.layers.Lambda(preproc, name='mobilenetv2_preproc')(inputs)

base = tf.keras.applications.MobileNetV2(
    input_tensor=x, include_top=False, weights='imagenet'
)

# Skip connections from MobileNetV2
# resolutions (for 512 input):
# block_1_expand_relu(256x256), block_3_expand_relu(128x128),
# block_6_expand_relu(64x64), block_13_expand_relu(32x32)
skip1 = base.get_layer('block_1_expand_relu').output  # 256x256
skip2 = base.get_layer('block_3_expand_relu').output  # 128x128
skip3 = base.get_layer('block_6_expand_relu').output  #  64x64
skip4 = base.get_layer('block_13_expand_relu').output #  32x32
bottleneck = base.get_layer('block_16_project').output # 16x16

def up_block(x, skip, filters, name=None):
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding='same', name=None if not name else name+"_up")(x)
    x = tf.keras.layers.Concatenate(name=None if not name else name+"_concat")([x, skip])
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu', kernel_initializer='he_normal', name=None if not name else name+"_conv1")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu', kernel_initializer='he_normal', name=None if not name else name+"_conv2")(x)
    return x

d1 = up_block(bottleneck, skip4, 256, name="dec1")  # 16->32
d2 = up_block(d1,        skip3, 128, name="dec2")   # 32->64
d3 = up_block(d2,        skip2, 64,  name="dec3")   # 64->128
d4 = up_block(d3,        skip1, 32,  name="dec4")   # 128->256

d5 = tf.keras.layers.Conv2DTranspose(16, 2, strides=2, padding='same', name="dec5_up")(d4)  # 256->512
d5 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="dec5_conv1")(d5)
d5 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="dec5_conv2")(d5)

outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='mask')(d5)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='UNet_MobileNetV2')

# Start with encoder frozen, then unfreeze for fine-tuning
for layer in base.layers:
    layer.trainable = False

# Dice + BCE loss and metrics
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (1.0 - dice_coef(y_true, y_pred))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=bce_dice_loss,
    metrics=['accuracy', dice_coef]
)

model.summary()

# ----------------------------
# Training
# ----------------------------
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_butterfly_unet.h5', monitor='val_loss', verbose=1, save_best_only=True)
early_stop  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')

# Inputs to model should be float in [0,255] but preprocessing layer handles scaling.
# Masks are float {0,1}
results = model.fit(
    X_train.astype(np.float32), Y_train.astype(np.float32),
    validation_split=0.1,
    batch_size=16,
    epochs=25,
    callbacks=[checkpointer, early_stop, reduce_lr, tensorboard],
    shuffle=True
)

# ----------------------------
# Fine-tune: unfreeze encoder (optional but recommended)
# ----------------------------
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=['accuracy', dice_coef]
)

results_ft = model.fit(
    X_train.astype(np.float32), Y_train.astype(np.float32),
    validation_split=0.1,
    batch_size=8,
    epochs=10,
    callbacks=[checkpointer, early_stop, reduce_lr, tensorboard],
    shuffle=True
)

# ----------------------------
# Predictions
# ----------------------------
# Train / Val split index
split_idx = int(X_train.shape[0] * 0.9)

preds_train = model.predict(X_train[:split_idx], verbose=1)
preds_val   = model.predict(X_train[split_idx:], verbose=1)
preds_test  = model.predict(X_test, verbose=1)

# Threshold to binary
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t   = (preds_val   > 0.5).astype(np.uint8)
preds_test_t  = (preds_test  > 0.5).astype(np.uint8)

# ----------------------------
# Sanity checks (optional)
# ----------------------------
if split_idx > 0:
    ix = random.randint(0, preds_train_t.shape[0]-1)
    plt.figure(); plt.title("Train Image"); plt.imshow(X_train[ix]); plt.axis('off')
    plt.figure(); plt.title("GT Mask");     plt.imshow(np.squeeze(Y_train[ix]), cmap='gray'); plt.axis('off')
    plt.figure(); plt.title("Pred Mask");   plt.imshow(np.squeeze(preds_train_t[ix]), cmap='gray'); plt.axis('off')
    plt.show()

if preds_val_t.shape[0] > 0:
    ix = random.randint(0, preds_val_t.shape[0]-1)
    plt.figure(); plt.title("Val Image"); plt.imshow(X_train[split_idx:][ix]); plt.axis('off')
    plt.figure(); plt.title("Val GT");    plt.imshow(np.squeeze(Y_train[split_idx:][ix]), cmap='gray'); plt.axis('off')
    plt.figure(); plt.title("Val Pred");  plt.imshow(np.squeeze(preds_val_t[ix]), cmap='gray'); plt.axis('off')
    plt.show()

# ----------------------------
# Save test predictions resized back to original size
# ----------------------------
print("Saving test predictions to:", PRED_TEST_OUT)
for i, id_ in enumerate(test_ids):
    H, W = sizes_test[i]
    mask_small = np.squeeze(preds_test_t[i]).astype(np.float32)  # (512, 512)
    # Resize back to original with nearest (order=0) to keep binary
    mask_orig = resize(mask_small, (H, W), order=0, anti_aliasing=False, preserve_range=True)
    mask_orig = (mask_orig > 0.5).astype(np.uint8) * 255

    out_dir = os.path.join(PRED_TEST_OUT, id_)
    os.makedirs(out_dir, exist_ok=True)
    imsave(os.path.join(out_dir, f'{id_}_pred.png'), mask_orig.astype(np.uint8))

print("All done!")
