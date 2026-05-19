"""
Sign Language Model Training Script
=====================================
Trains an LSTM model on self-recorded landmark sequences.

Supports:
  - Single language  (ASL only, BSL only, or MUTUAL only)
  - Combined training across any subset of ASL / BSL / MUTUAL

Usage:
  python train_model.py

Edit the CONFIG section below before running.
"""

import os
import csv
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
OUTPUT_DIR   = r"C:\Diss\Models\Dual_60_Model_Mutual"
SIGNS_FILTER = [
     # original 10
    'before', 'book', 'candy', 'chair', 'clothes',
    'color', 'computer', 'corn',
    # everyday actions
    'sleep', 'walk', 'run', 'sit',
    'help', 'stop', 'go', 'come', 'wait',
    'finish', 'want', 'need', 'like', 'love',
    'hate', 'know', 'think', 'open', 'close',
    # objects and places
    'water', 'house', 'door', 'table',
    # people
    'friend', 'mother', 'father',
    'teacher', 'doctor',
    # descriptors
    'hot', 'good', 'bad', 'fast', 'slow',
    'big', 'small', 'happy', 'sad',
    # extra 5
    'give', 'take', 'look', 'hear', 'smile',
    'me', 'you', 'drink', 'eat','money',
    'car', 'cold', 'phone', 'stand', 'baby'
    ]

LANGUAGES    = ["ASL", "BSL", "MUTUAL"]

SEQUENCE_LEN      = 45
NUM_FEATURES      = 126
EPOCHS            = 100
BATCH_SIZE        = 16
SNAPSHOT_INTERVAL = 10   # save a manual checkpoint every N epochs

# ── AUGMENTATION CONFIG ───────────────────────────────────────────────────────

AUGMENT          = True
AUGMENT_FACTOR   = 3       # how many augmented copies per real recording
NOISE_STD        = 0.01    # gaussian noise strength (small = subtle)
SCALE_RANGE      = (0.90, 1.10)   # random hand scale
TIME_WARP_RANGE  = (0.85, 1.15)   # random speed variation

# ── SETUP ─────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
snapshots_dir = os.path.join(OUTPUT_DIR, "snapshots")
os.makedirs(snapshots_dir, exist_ok=True)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}\n")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_dataset(data_dir, languages):
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

                if seq.shape[1] == 63:
                    seq = np.concatenate([seq, np.zeros_like(seq)], axis=1)

                if seq.shape[0] != SEQUENCE_LEN:
                    indices = np.linspace(0, len(seq) - 1, SEQUENCE_LEN).astype(int)
                    seq = seq[indices]

                sequences.append(seq.astype(np.float32))
                labels_str.append(f"{lang}_{sign}")

    if len(sequences) == 0:
        raise RuntimeError("No training data found. Check your DATA_DIR and LANGUAGES config.")

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
    aug = seq.copy()
    noise = np.random.normal(0, NOISE_STD, aug.shape).astype(np.float32)
    zero_frames = (aug == 0).all(axis=-1, keepdims=True)
    aug = np.where(zero_frames, 0.0, aug + noise)
    scale = np.random.uniform(*SCALE_RANGE)
    aug = np.where(zero_frames, 0.0, aug * scale)
    warp = np.random.uniform(*TIME_WARP_RANGE)
    n = len(aug)
    warped_len = max(2, int(n * warp))
    indices_from = np.linspace(0, n - 1, warped_len)
    indices_to   = np.linspace(0, warped_len - 1, n).astype(int)
    warped = np.array([aug[min(int(i), n - 1)] for i in indices_from], dtype=np.float32)
    aug = warped[indices_to]
    return aug


def augment_dataset(X, y, factor):
    print(f"\nAugmenting dataset (factor={factor})...")
    X_aug_list = [X]
    y_aug_list = [y]
    for i in range(factor):
        X_new = np.array([augment_sequence(seq) for seq in X], dtype=np.float32)
        X_aug_list.append(X_new)
        y_aug_list.append(y)
    X_out = np.concatenate(X_aug_list, axis=0)
    y_out = np.concatenate(y_aug_list, axis=0)
    perm  = np.random.permutation(len(X_out))
    print(f"Dataset size after augmentation: {X_out.shape}")
    return X_out[perm], y_out[perm]


if AUGMENT:
    X, y = augment_dataset(X, y, AUGMENT_FACTOR)

# ── NORMALISE ─────────────────────────────────────────────────────────────────

print("\nNormalising features...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

flat_train   = X_train.reshape(-1, NUM_FEATURES)
nonzero_mask = flat_train.any(axis=1)
feat_mean    = flat_train[nonzero_mask].mean(axis=0)
feat_std     = flat_train[nonzero_mask].std(axis=0) + 1e-8

def normalise(arr):
    out         = (arr - feat_mean) / feat_std
    zero_frames = (arr == 0).all(axis=-1, keepdims=True)
    return np.where(zero_frames, 0.0, out)

X_train = normalise(X_train)
X_val   = normalise(X_val)

print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")

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

# ── MANUAL SNAPSHOT CALLBACK ──────────────────────────────────────────────────

class SnapshotCallback(keras.callbacks.Callback):
    """Saves a full model snapshot every SNAPSHOT_INTERVAL epochs."""
    def __init__(self, save_dir, interval):
        super().__init__()
        self.save_dir = save_dir
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        # epoch is 0-indexed; save after epoch 9, 19, 29 ... → labels 10, 20, 30 ...
        ep = epoch + 1
        if ep % self.interval == 0:
            path = os.path.join(self.save_dir, f"snapshot_epoch_{ep:03d}.h5")
            self.model.save(path)
            val_acc = logs.get("val_accuracy", 0)
            print(f"\n  [Snapshot] Epoch {ep:3d} → saved ({val_acc:.4f} val_acc)")

# ── TRAIN ─────────────────────────────────────────────────────────────────────

checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.h5")

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
    ),
    SnapshotCallback(snapshots_dir, SNAPSHOT_INTERVAL),
]

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ── CHECKPOINT AUDIT TABLE ────────────────────────────────────────────────────

print("\n── Checkpoint Audit ──")
print(f"{'Epoch':>8}  {'Val Acc':>10}  {'Val Loss':>10}  {'Selected':>10}")
print("-" * 46)

snapshot_files = sorted([
    f for f in os.listdir(snapshots_dir) if f.endswith('.h5')
])

audit_rows = []
best_epoch   = None
best_val_acc = -1

for fname in snapshot_files:
    # Extract epoch number from filename e.g. snapshot_epoch_010.h5
    epoch_num = int(fname.replace("snapshot_epoch_", "").replace(".h5", ""))
    # Fetch val_accuracy from training history (0-indexed, so epoch 10 → index 9)
    hist_idx  = epoch_num - 1
    if hist_idx < len(history.history['val_accuracy']):
        val_acc  = history.history['val_accuracy'][hist_idx]
        val_loss = history.history['val_loss'][hist_idx]
    else:
        # Epoch was not reached (early stopping)
        val_acc  = None
        val_loss = None

    if val_acc is not None and val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch_num

    audit_rows.append({
        'epoch'   : epoch_num,
        'val_acc' : round(val_acc, 6)  if val_acc  is not None else 'N/A (not reached)',
        'val_loss': round(val_loss, 6) if val_loss is not None else 'N/A (not reached)',
    })

for row in audit_rows:
    selected = '<<< BEST' if row['epoch'] == best_epoch else ''
    acc_str  = f"{row['val_acc']:.4f}" if isinstance(row['val_acc'], float) else row['val_acc']
    loss_str = f"{row['val_loss']:.4f}" if isinstance(row['val_loss'], float) else row['val_loss']
    print(f"{row['epoch']:>8}  {acc_str:>10}  {loss_str:>10}  {selected}")

# Load and save the best snapshot as the selected model
if best_epoch is not None:
    best_snapshot_path = os.path.join(snapshots_dir, f"snapshot_epoch_{best_epoch:03d}.h5")
    selected_model = keras.models.load_model(best_snapshot_path)
    selected_path  = os.path.join(OUTPUT_DIR, "selected_model.h5")
    selected_model.save(selected_path)
    print(f"\n  Best snapshot: Epoch {best_epoch} (val_acc={best_val_acc:.4f})")
    print(f"  Saved as selected_model.h5")
else:
    print("  No snapshots found — using best_model.h5 from ModelCheckpoint.")
    selected_model = model

# Save audit table as CSV
# The file has two sections:
#   1. A metadata block (key, value) describing the run.
#   2. A blank separator row, then the per-epoch audit data.
import datetime
csv_path = os.path.join(OUTPUT_DIR, "checkpoint_audit.csv")
with open(csv_path, 'w', newline='') as f:
    # ── metadata block ────────────────────────────────────────────────────────
    meta_writer = csv.writer(f)
    meta_writer.writerow(['key', 'value'])
    meta_writer.writerow(['languages',      ', '.join(LANGUAGES)])
    meta_writer.writerow(['num_classes',    NUM_CLASSES])
    meta_writer.writerow(['epochs_trained', len(history.history['val_accuracy'])])
    meta_writer.writerow(['best_epoch',     best_epoch if best_epoch else 'N/A'])
    meta_writer.writerow(['best_val_acc',   f"{best_val_acc:.6f}" if best_val_acc > 0 else 'N/A'])
    meta_writer.writerow(['timestamp',      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    meta_writer.writerow([])   # blank separator row
    # ── per-epoch audit data ──────────────────────────────────────────────────
    writer = csv.DictWriter(f, fieldnames=['epoch', 'val_acc', 'val_loss', 'selected'])
    writer.writeheader()
    for row in audit_rows:
        writer.writerow({**row, 'selected': 'BEST' if row['epoch'] == best_epoch else ''})
print(f"  Audit table saved to checkpoint_audit.csv\n")

# ── PLOT TRAINING CURVES ──────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(
    f"Languages: {' + '.join(LANGUAGES)}  |  Classes: {NUM_CLASSES}  |  "
    f"Best epoch: {best_epoch}  (val_acc={best_val_acc:.4f})",
    fontsize=11, y=1.02,
)

ax1.plot(history.history['loss'],     label='Train loss')
ax1.plot(history.history['val_loss'], label='Val loss')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['accuracy'],     label='Train acc')
ax2.plot(history.history['val_accuracy'], label='Val acc')
# Mark snapshot epochs with vertical lines
for row in audit_rows:
    ep = row['epoch']
    if isinstance(row['val_acc'], float):
        is_best = (row['epoch'] == best_epoch)
        ax2.axvline(ep - 1, color='green' if is_best else 'grey',
                    linestyle='--', linewidth=0.8, alpha=0.6)
        if is_best:
            ax2.annotate(
                f"best\nep {ep}",
                xy=(ep - 1, best_val_acc), xytext=(ep + 0.5, best_val_acc - 0.05),
                fontsize=7, color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=0.8),
            )
ax2.set_title('Accuracy  (▲ = best snapshot)')
ax2.set_xlabel('Epoch')
ax2.set_ylim(0, 1.05)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Training curves saved to {curves_path}")

# ── CONFUSION MATRIX (using selected model) ───────────────────────────────────

y_pred     = np.argmax(selected_model.predict(X_val), axis=1)
sign_names = [id_to_sign[i] for i in range(NUM_CLASSES)]
cm         = confusion_matrix(y_val, y_pred)
tick_labels = [s.split('_', 1)[1] if '_' in s else s for s in sign_names]

n = len(sign_names)
fig_side = max(16, n * 0.28)
fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.92))
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

sns.heatmap(
    cm_norm, ax=ax, annot=False, fmt='', cmap='Blues',
    vmin=0, vmax=1, xticklabels=tick_labels, yticklabels=tick_labels,
    linewidths=0.25, linecolor='#dddddd',
    cbar_kws={'label': 'Recall (row-normalised)', 'shrink': 0.6},
)

tick_fs = max(5, int(180 / n))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fs)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=tick_fs)

if len(LANGUAGES) > 1:
    # Compute cumulative boundary positions for each language group.
    # sign_names is sorted alphabetically so ASL_ < BSL_ < MUTUAL_ naturally.
    cumulative = 0
    lang_order = sorted(set(s.split('_')[0] for s in sign_names))  # preserves alpha sort
    lang_spans = {}   # lang -> (start_idx, end_idx)  in the matrix
    for lang in lang_order:
        count = sum(1 for s in sign_names if s.startswith(f"{lang}_"))
        if count:
            lang_spans[lang] = (cumulative, cumulative + count)
            cumulative += count

    # Draw divider lines between each language block
    boundaries = [span[1] for span in list(lang_spans.values())[:-1]]
    for boundary in boundaries:
        ax.axhline(boundary, color='red', linewidth=1.5, linestyle='--')
        ax.axvline(boundary, color='red', linewidth=1.5, linestyle='--')

    # Label each language block on both axes
    for lang, (start, end) in lang_spans.items():
        mid = (start + end) / 2
        ax.text(mid, -1.5, lang, ha='center', va='bottom',
                fontsize=tick_fs + 3, fontweight='bold', color='red',
                transform=ax.transData)
        ax.text(-1.5, mid, lang, ha='right', va='center',
                fontsize=tick_fs + 3, fontweight='bold', color='red',
                rotation=90, transform=ax.transData)

ax.set_title(f'Confusion Matrix — Validation Set (row-normalised)\nUsing selected model: Epoch {best_epoch}', fontsize=14, pad=14)
ax.set_ylabel('True label',      fontsize=11)
ax.set_xlabel('Predicted label', fontsize=11)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Confusion matrix saved to {cm_path}")

print('\nPer-class report:')
print(classification_report(y_val, y_pred, target_names=sign_names))

# ── PER-LANGUAGE CONFUSION MATRICES ──────────────────────────────────────────

for lang in LANGUAGES:
    lang_indices = sorted([idx for sign, idx in label_map.items() if sign.startswith(f"{lang}_")])
    lang_names   = [id_to_sign[i].replace(f"{lang}_", "") for i in lang_indices]

    mask        = np.isin(y_val, lang_indices)
    y_true_lang = y_val[mask]
    y_pred_lang = y_pred[mask]

    OTHER  = len(lang_indices)
    remap  = {old: new for new, old in enumerate(lang_indices)}
    y_true_lang = np.array([remap[i] for i in y_true_lang])
    y_pred_lang = np.array([remap.get(i, OTHER) for i in y_pred_lang])

    all_labels = list(range(len(lang_indices) + 1))
    cm_lang    = confusion_matrix(y_true_lang, y_pred_lang, labels=all_labels)

    has_other = cm_lang[:, OTHER].sum() > 0
    if not has_other:
        cm_lang   = cm_lang[:, :OTHER]
        col_names = lang_names
    else:
        other_langs = [l for l in LANGUAGES if l != lang]
        col_names  = lang_names + [f'── {"/".join(other_langs)} ──']

    n_lang   = len(lang_indices)
    fig_side = max(12, n_lang * 0.38)
    fig, ax  = plt.subplots(figsize=(fig_side, fig_side * 0.92))
    tick_fs  = max(7, int(200 / n_lang))

    sns.heatmap(
        cm_lang, ax=ax, annot=True, fmt='d',
        xticklabels=col_names, yticklabels=lang_names,
        cmap='Blues', linewidths=0.4, linecolor='#dddddd',
        annot_kws={'size': max(6, int(170 / n_lang))},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_fs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=tick_fs)
    ax.set_title(f'Confusion Matrix — {lang} — Validation Set\nUsing selected model: Epoch {best_epoch}', fontsize=13, pad=12)
    ax.set_ylabel('True label',      fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{lang}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"{lang} confusion matrix saved to {path}")

    print(f'\n{lang} per-class report:')
    in_lang_mask = y_pred_lang < OTHER
    print(classification_report(
        y_true_lang[in_lang_mask], y_pred_lang[in_lang_mask],
        labels=list(range(len(lang_indices))),
        target_names=lang_names, zero_division=0,
    ))
    if has_other:
        cross = int(cm_lang[:, OTHER].sum())
        other_langs = [l for l in LANGUAGES if l != lang]
        print(f"  ⚠  {cross} sample(s) predicted outside {lang} (into {'/'.join(other_langs)} signs)\n")

# ── SAVE CONFIG ───────────────────────────────────────────────────────────────

config = {
    'sequence_len'          : SEQUENCE_LEN,
    'num_features'          : NUM_FEATURES,
    'num_classes'           : NUM_CLASSES,
    'languages'             : LANGUAGES,
    'label_map'             : label_map,
    'model_path'            : 'selected_model.h5',
    'feat_mean_path'        : 'feat_mean.npy',
    'feat_std_path'         : 'feat_std.npy',
    'best_snapshot_epoch'   : best_epoch,
    'best_snapshot_val_acc' : round(best_val_acc, 4),
}

config_path = os.path.join(OUTPUT_DIR, "inference_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n── Training complete ──")
print(f"Selected model    : Epoch {best_epoch} (val_acc={best_val_acc:.4f})")
print(f"Model saved to    : {os.path.join(OUTPUT_DIR, 'selected_model.h5')}")
print(f"Config saved to   : {config_path}")
print(f"Audit CSV saved to: {csv_path}")
print(f"Ready for inference!")
