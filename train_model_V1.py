"""
Sign Language Model Training Script
=====================================
Trains an LSTM model on self-recorded landmark sequences.

Supports:
  - Single language (ASL only or BSL only)
  - Combined bilingual training (ASL + BSL)

Usage:
  python train_model.py

Edit the CONFIG section below before running.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR     = r"C:\Diss\CollectedData"
OUTPUT_DIR   = r"C:\Diss\Model"
SIGNS_FILTER = ['before', 'book', 'candy', 'chair', 'clothes']

# Which languages to include — set to ["ASL"], ["BSL"], or ["ASL", "BSL"]
LANGUAGES    = ["ASL"]

SEQUENCE_LEN = 64       # must match what you recorded with
NUM_FEATURES = 126      # 63 per hand × 2 hands
EPOCHS       = 100
BATCH_SIZE   = 16

# ── AUGMENTATION CONFIG ───────────────────────────────────────────────────────

AUGMENT          = True    # turn off if you want to train on raw data only
AUGMENT_FACTOR   = 3       # how many augmented copies per real recording
NOISE_STD        = 0.01    # gaussian noise strength (small = subtle)
SCALE_RANGE      = (0.90, 1.10)   # random hand scale
TIME_WARP_RANGE  = (0.85, 1.15)   # random speed variation

# ── SETUP ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}\n")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_dataset(data_dir, languages):
    """
    Walks CollectedData/LANGUAGE/sign/*.npy and loads all sequences.
    Returns X (samples, seq_len, features), y (samples,), label_map dict.
    """
    sequences = []
    labels_str = []

    for lang in languages:
        lang_dir = os.path.join(data_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"WARNING: No data found for language '{lang}' at {lang_dir}")
            continue

        for sign in sorted(os.listdir(lang_dir)):
            if SIGNS_FILTER and sign not in SIGNS_FILTER:
                continue
            sign_dir = os.path.join(lang_dir, sign)
            if not os.path.isdir(sign_dir):
                continue

            files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
            if len(files) == 0:
                print(f"  WARNING: No recordings found for {lang}/{sign}")
                continue

            for fname in files:
                seq = np.load(os.path.join(sign_dir, fname))

                # Handle sequences recorded with 1-hand (63 features)
                # pad to 126 so everything is the same shape
                if seq.shape[1] == 63:
                    seq = np.concatenate(
                        [seq, np.zeros_like(seq)], axis=1
                    )

                # Resample to fixed length if needed
                if seq.shape[0] != SEQUENCE_LEN:
                    indices = np.linspace(0, len(seq) - 1, SEQUENCE_LEN).astype(int)
                    seq = seq[indices]

                sequences.append(seq.astype(np.float32))
                labels_str.append(sign)   # label is the sign name regardless of language

    if len(sequences) == 0:
        raise RuntimeError("No training data found. Check your DATA_DIR and LANGUAGES config.")

    # Build label map from unique sign names
    unique_signs = sorted(set(labels_str))
    label_map    = {sign: idx for idx, sign in enumerate(unique_signs)}

    X = np.array(sequences, dtype=np.float32)
    y = np.array([label_map[s] for s in labels_str], dtype=np.int64)

    return X, y, label_map


print("Loading data...")
X, y, label_map = load_dataset(DATA_DIR, LANGUAGES)
id_to_sign = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)

print(f"Dataset: {X.shape}  |  Classes: {NUM_CLASSES}")
print(f"Label map: {label_map}\n")

from collections import Counter
for sign, idx in sorted(label_map.items(), key=lambda x: x[1]):
    count = np.sum(y == idx)
    print(f"  [{idx}] {sign:15s}  {count} recordings")

# ── AUGMENTATION ──────────────────────────────────────────────────────────────

def augment_sequence(seq):
    """
    Apply random augmentation to a single sequence (SEQUENCE_LEN, NUM_FEATURES).
    Returns a new augmented sequence without modifying the original.
    """
    aug = seq.copy()

    # 1. Gaussian noise — adds slight jitter to landmark positions
    noise = np.random.normal(0, NOISE_STD, aug.shape).astype(np.float32)
    zero_frames = (aug == 0).all(axis=-1, keepdims=True)
    aug = np.where(zero_frames, 0.0, aug + noise)

    # 2. Random scale — makes the hand slightly bigger or smaller
    scale = np.random.uniform(*SCALE_RANGE)
    aug = np.where(zero_frames, 0.0, aug * scale)

    # 3. Time warp — slightly stretch or compress the sequence
    warp = np.random.uniform(*TIME_WARP_RANGE)
    n = len(aug)
    warped_len = int(n * warp)
    warped_len = max(2, warped_len)
    indices_from = np.linspace(0, n - 1, warped_len)
    indices_to   = np.linspace(0, warped_len - 1, n).astype(int)
    warped = np.array([
        aug[min(int(i), n - 1)] for i in indices_from
    ], dtype=np.float32)
    aug = warped[indices_to]

    return aug


def augment_dataset(X, y, factor):
    """Create `factor` augmented copies of every sample and append to dataset."""
    print(f"\nAugmenting dataset (factor={factor})...")
    X_aug_list = [X]
    y_aug_list = [y]

    for i in range(factor):
        X_new = np.array([augment_sequence(seq) for seq in X], dtype=np.float32)
        X_aug_list.append(X_new)
        y_aug_list.append(y)

    X_out = np.concatenate(X_aug_list, axis=0)
    y_out = np.concatenate(y_aug_list, axis=0)

    # Shuffle so augmented copies aren't all grouped together
    perm  = np.random.permutation(len(X_out))
    print(f"Dataset size after augmentation: {X_out.shape}")
    return X_out[perm], y_out[perm]


if AUGMENT:
    X, y = augment_dataset(X, y, AUGMENT_FACTOR)

# ── NORMALISE ─────────────────────────────────────────────────────────────────

print("\nNormalising features...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Compute stats on training data only — avoid data leakage
flat_train   = X_train.reshape(-1, NUM_FEATURES)
nonzero_mask = flat_train.any(axis=1)
feat_mean    = flat_train[nonzero_mask].mean(axis=0)   # (126,)
feat_std     = flat_train[nonzero_mask].std(axis=0) + 1e-8

def normalise(arr):
    """Normalise, keeping all-zero frames as zero."""
    out         = (arr - feat_mean) / feat_std
    zero_frames = (arr == 0).all(axis=-1, keepdims=True)
    return np.where(zero_frames, 0.0, out)

X_train = normalise(X_train)
X_val   = normalise(X_val)

print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")

# Save normalisation stats
np.save(os.path.join(OUTPUT_DIR, "feat_mean.npy"), feat_mean)
np.save(os.path.join(OUTPUT_DIR, "feat_std.npy"),  feat_std)
print("Saved feat_mean.npy and feat_std.npy")

# ── MODEL ─────────────────────────────────────────────────────────────────────

def build_model(seq_len, num_features, num_classes):
    inp = keras.Input(shape=(seq_len, num_features))
    x   = layers.Masking(mask_value=0.0)(inp)
    x   = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x   = layers.LSTM(64,  dropout=0.3)(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inp, out)

model = build_model(SEQUENCE_LEN, NUM_FEATURES, NUM_CLASSES)
model.summary()

# ── TRAIN ─────────────────────────────────────────────────────────────────────

checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.keras")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
]

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ── PLOT TRAINING CURVES ──────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(history.history['loss'],     label='Train loss')
ax1.plot(history.history['val_loss'], label='Val loss')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['accuracy'],     label='Train acc')
ax2.plot(history.history['val_accuracy'], label='Val acc')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylim(0, 1.05)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=150)
plt.show()
print(f"Training curves saved to {curves_path}")

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────

y_pred     = np.argmax(model.predict(X_val), axis=1)
sign_names = [id_to_sign[i] for i in range(NUM_CLASSES)]
cm         = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=sign_names, yticklabels=sign_names,
            cmap='Blues')
plt.title('Confusion Matrix — Validation Set')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"Confusion matrix saved to {cm_path}")

print('\nPer-class report:')
print(classification_report(y_val, y_pred, target_names=sign_names))

# ── SAVE CONFIG ───────────────────────────────────────────────────────────────

best_val_acc = max(history.history['val_accuracy'])

config = {
    'sequence_len'   : SEQUENCE_LEN,
    'num_features'   : NUM_FEATURES,
    'num_classes'    : NUM_CLASSES,
    'languages'      : LANGUAGES,
    'label_map'      : label_map,
    'model_path'     : 'best_model.keras',
    'feat_mean_path' : 'feat_mean.npy',
    'feat_std_path'  : 'feat_std.npy',
    'best_val_acc'   : round(best_val_acc, 4),
}

config_path = os.path.join(OUTPUT_DIR, "inference_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n── Training complete ──")
print(f"Best val accuracy : {best_val_acc:.1%}")
print(f"Model saved to    : {checkpoint_path}")
print(f"Config saved to   : {config_path}")
print(f"Ready for inference!")
