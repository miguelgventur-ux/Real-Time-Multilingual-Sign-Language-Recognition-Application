"""
Sign Language Model Testing Script

The test data directory (CollectedDataTesting) must follow the following structure for the script to function:
    CollectedDataTesting/
        ASL/
            baby/
                baby_ASL_000.npy
                ...
            bad/
                ...
        BSL/
            baby/
                baby_BSL_000.npy
                ...
            bad/
                ...
        MUTUAL/
            baby/
                baby_MUTUAL_000.npy
                ...
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow import keras

# Configuration
MODEL_DIR = r"C:\Diss\Models\Dual_60_Model_Mutual"
TEST_DIR = r"C:\Diss\CollectedDataTesting"
OUTPUT_DIR = r"C:\Diss\Models\Dual_60_Model_Mutual\TestResults"

# Best snapshot, chosen by the checkpoint audit as the model to test
MODEL_FILE = "selected_model.h5"

SEQUENCE_LEN = 45
NUM_FEATURES = 126

# Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}\n")

#Load the inference configuration
config_path = os.path.join(MODEL_DIR, "inference_config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"inference_config.json not found in {MODEL_DIR}. Run training first.")

with open(config_path) as f:
    config = json.load(f)

label_map = config['label_map']
id_to_sign  = {v: k for k, v in label_map.items()}
NUM_CLASSES = config['num_classes']
LANGUAGES = config['languages']

print(f"Model config loaded: {NUM_CLASSES} classes, languages={LANGUAGES}")
print(f"Best snapshot epoch: {config.get('best_snapshot_epoch', 'N/A')}")
print(f"Best snapshot val_acc: {config.get('best_snapshot_val_acc', 'N/A')}\n")

# Load normalisation statistics
feat_mean = np.load(os.path.join(MODEL_DIR, "feat_mean.npy"))
feat_std = np.load(os.path.join(MODEL_DIR, "feat_std.npy"))

def normalise(arr):
    out = (arr - feat_mean) / feat_std
    zero_frames = (arr == 0).all(axis=-1, keepdims=True)
    return np.where(zero_frames, 0.0, out)

# Load the model
model_path = os.path.join(MODEL_DIR, MODEL_FILE)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model = keras.models.load_model(model_path)
print(f"Model loaded: {model_path}\n")

# Load test data
def load_test_dataset(test_dir, languages, label_map):
    """
    Walks TestingCollectedData/LANGUAGE/sign/*.npy.
    Only loads signs that exist in the training label_map.
    Returns X, y_true, and a list of (language, sign, filename) metadata.
    """
    sequences = []
    labels = []
    metadata  = [] #language, sign, filename) for per-sample reporting

    for lang in languages:
        lang_dir = os.path.join(test_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"WARNING: No test data found for language '{lang}' at {lang_dir}")
            continue

        for sign in sorted(os.listdir(lang_dir)):
            sign_dir = os.path.join(lang_dir, sign)
            if not os.path.isdir(sign_dir):
                continue

            class_key = f"{lang}_{sign}"
            if class_key not in label_map:
                print(f"  SKIP: {class_key} not in training label map")
                continue

            files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
            if len(files) == 0:
                print(f"  WARNING: No test recordings for {lang}/{sign}")
                continue

            for fname in sorted(files):
                seq = np.load(os.path.join(sign_dir, fname))

                if seq.shape[1] == 63:
                    seq = np.concatenate([seq, np.zeros_like(seq)], axis=1)

                if seq.shape[0] != SEQUENCE_LEN:
                    indices = np.linspace(0, len(seq) - 1, SEQUENCE_LEN).astype(int)
                    seq = seq[indices]

                sequences.append(seq.astype(np.float32))
                labels.append(label_map[class_key])
                metadata.append((lang, sign, fname))

    if len(sequences) == 0:
        raise RuntimeError(f"No test data found in {test_dir}. Check directory structure.")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, metadata


print("Loading test data...")
X_test, y_true, metadata = load_test_dataset(TEST_DIR, LANGUAGES, label_map)
print(f"Test set: {X_test.shape[0]} samples across {len(set(y_true))} classes\n")

# Print test sample counts per-class
print("Samples per class in test set:")
for class_idx in sorted(set(y_true)):
    count = int(np.sum(y_true == class_idx))
    print(f"  [{class_idx:3d}] {id_to_sign[class_idx]:20s}  {count} sample(s)")
print()

# Normalise and predict
X_test_norm = normalise(X_test)
y_probs = model.predict(X_test_norm, verbose=1)
y_pred = np.argmax(y_probs, axis=1)
y_conf = np.max(y_probs, axis=1) # confidence of each prediction

# Overall metrics
sign_names = [id_to_sign[i] for i in range(NUM_CLASSES)]

overall_acc = accuracy_score(y_true, y_pred)
overall_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
overall_rec = recall_score(y_true, y_pred,    average='macro', zero_division=0)
overall_f1 = f1_score(y_true, y_pred,        average='macro', zero_division=0)

print("\n── Overall Test Results ──────────────────────────────────────")
print(f"  Accuracy  : {overall_acc:.4f}  ({overall_acc:.1%})")
print(f"  Precision : {overall_prec:.4f}  (macro)")
print(f"  Recall    : {overall_rec:.4f}  (macro)")
print(f"  F1 Score  : {overall_f1:.4f}  (macro)")
print(f"  Samples   : {len(y_true)}")
print("─────────────────────────────────────────────────────────────\n")

# Results table per sample

print("── Per-Sample Results ──")
print(f"{'Lang':>5}  {'Sign':>15}  {'File':>25}  {'True':>20}  {'Pred':>20}  {'Conf':>6}  {'Correct':>8}")
print("-" * 110)

per_sample_rows = []
for i, (lang, sign, fname) in enumerate(metadata):
    true_label = id_to_sign[y_true[i]]
    pred_label = id_to_sign[y_pred[i]]
    conf = y_conf[i]
    correct = y_true[i] == y_pred[i]
    marker = "✓" if correct else "✗"
    print(f"{lang:>5}  {sign:>15}  {fname:>25}  {true_label:>20}  {pred_label:>20}  {conf:>6.4f}  {marker:>8}")
    per_sample_rows.append({
        'language': lang,
        'sign': sign,
        'file': fname,
        'true_label': true_label,
        'pred_label': pred_label,
        'confidence': round(float(conf), 6),
        'correct': correct,
    })

# CSV per sample

sample_csv = os.path.join(OUTPUT_DIR, "test_results_per_sample.csv")
with open(sample_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['language','sign','file','true_label','pred_label','confidence','correct'])
    writer.writeheader()
    writer.writerows(per_sample_rows)
print(f"\nPer-sample results saved to: {sample_csv}")

# Summary CSV

summary_csv = os.path.join(OUTPUT_DIR, "test_results_summary.csv")
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Languages", ', '.join(LANGUAGES)])
    writer.writerow(["Test samples", len(y_true)])
    writer.writerow(["Accuracy", f"{overall_acc:.6f}"])
    writer.writerow(["Precision (macro)", f"{overall_prec:.6f}"])
    writer.writerow(["Recall (macro)", f"{overall_rec:.6f}"])
    writer.writerow(["F1 (macro)", f"{overall_f1:.6f}"])
    writer.writerow([])
    writer.writerow(["Per-language accuracy", ""])
    for lang in LANGUAGES:
        lang_indices = [idx for sign, idx in label_map.items()
                        if sign.startswith(f"{lang}_") and idx in set(y_true)]
        if lang_indices:
            mask = np.isin(y_true, lang_indices)
            lang_acc = accuracy_score(y_true[mask], y_pred[mask])
            lang_n = int(mask.sum())
            writer.writerow([f"  {lang} accuracy", f"{lang_acc:.6f}  ({lang_n} samples)"])
        else:
            writer.writerow([f"  {lang} accuracy", "no test samples"])
    writer.writerow([])
    writer.writerow(["Model file", MODEL_FILE])
    writer.writerow(["Best snapshot epoch", config.get('best_snapshot_epoch', 'N/A')])
    writer.writerow(["Best snapshot val_acc", config.get('best_snapshot_val_acc', 'N/A')])
print(f"Summary saved to: {summary_csv}")

# Classification report
# Only report on classes actually present in the test set
present_classes = sorted(set(y_true))
present_names = [id_to_sign[i] for i in present_classes]

print("\n── Classification Report ──")
print(classification_report(
    y_true, y_pred,
    labels=present_classes,
    target_names=present_names,
    zero_division=0
))

# Full confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=present_classes)
tick_labels = [s.split('_', 1)[1] if '_' in s else s for s in present_names]
n = len(present_classes)
fig_side = max(14, n * 0.32)
tick_fs = max(6, int(180 / n))

fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.92))
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

# Build annotation array: show value for non-zero cells, blank for zeros
annot_array = np.where(cm_norm > 0,np.vectorize(lambda v: f"{v:.2f}")(cm_norm), "")

sns.heatmap(
    cm_norm, ax=ax, annot=annot_array, fmt='', cmap='Blues',
    vmin=0, vmax=1,
    xticklabels=tick_labels, yticklabels=tick_labels,
    linewidths=0.3, linecolor='#dddddd',
    annot_kws={'size': max(5, int(140 / n))},
    cbar_kws={'label': 'Recall (row-normalised)', 'shrink': 0.6},
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fs)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=tick_fs)

if len(LANGUAGES) > 1:
    # Compute language block boundaries dynamically from present_names
    # order (ASL_ < BSL_ < MUTUAL_), so boundaries are naturally contiguous
    lang_order = sorted(set(s.split('_')[0] for s in present_names))
    cumulative = 0
    lang_spans = {}
    for lang_key in lang_order:
        count = sum(1 for s in present_names if s.startswith(f"{lang_key}_"))
        if count:
            lang_spans[lang_key] = (cumulative, cumulative + count)
            cumulative += count

    # Draw a divider line at every language boundary except the last
    boundaries = [span[1] for span in list(lang_spans.values())[:-1]]
    for boundary in boundaries:
        ax.axhline(boundary, color='red', linewidth=1.5, linestyle='--')
        ax.axvline(boundary, color='red', linewidth=1.5, linestyle='--')

    # Label each language block on both axes
    for lang_key, (start, end) in lang_spans.items():
        mid = (start + end) / 2
        ax.text(mid, -1.2, lang_key, ha='center', va='bottom',
                fontsize=tick_fs + 3, fontweight='bold', color='red',
                transform=ax.transData)
        ax.text(-1.2, mid, lang_key, ha='right', va='center',
                fontsize=tick_fs + 3, fontweight='bold', color='red',
                rotation=90, transform=ax.transData)

ax.set_title(f"Confusion Matrix — Test Set  (acc={overall_acc:.1%})", fontsize=14, pad=14)
ax.set_ylabel('True label', fontsize=11)
ax.set_xlabel('Predicted label', fontsize=11)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_test.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Test confusion matrix saved to: {cm_path}")

# Confusion matrices per language

for lang in LANGUAGES:
    lang_indices = sorted([idx for sign, idx in label_map.items()
                           if sign.startswith(f"{lang}_") and idx in present_classes])
    if not lang_indices:
        print(f"  No test samples found for language {lang}, skipping per-language matrix.")
        continue

    lang_names = [id_to_sign[i].replace(f"{lang}_", "") for i in lang_indices]

    mask = np.isin(y_true, lang_indices)
    y_true_lang = y_true[mask]
    y_pred_lang = y_pred[mask]

    OTHER = len(lang_indices)
    remap = {old: new for new, old in enumerate(lang_indices)}
    y_true_lang = np.array([remap[i] for i in y_true_lang])
    y_pred_lang = np.array([remap.get(i, OTHER) for i in y_pred_lang])

    all_labels = list(range(len(lang_indices) + 1))
    cm_lang = confusion_matrix(y_true_lang, y_pred_lang, labels=all_labels)

    has_other = cm_lang[:, OTHER].sum() > 0
    if not has_other:
        cm_lang = cm_lang[:, :OTHER]
        col_names = lang_names
    else:
        other_langs = [l for l in LANGUAGES if l != lang]
        # Short label so it never overflows on small matrices
        col_names   = lang_names + [f'[{"/".join(other_langs)}]']

    lang_acc = accuracy_score(y_true_lang, np.clip(y_pred_lang, 0, OTHER - 1))

    n_lang    = len(lang_indices)
    n_cols    = len(col_names)

    # scale by class count but enforce sensible min/max
    cell_size = 0.55          # inches per cell
    fig_h     = max(4, n_lang * cell_size + 2.5)
    fig_w     = max(5, n_cols * cell_size + 3.0)
    fig, ax   = plt.subplots(figsize=(fig_w, fig_h))

    tick_fs   = min(12, max(7, int(200 / max(n_lang, 8))))
    annot_fs  = min(12, max(6, int(170 / max(n_lang, 8))))

    sns.heatmap(
        cm_lang, ax=ax, annot=True, fmt='d',
        xticklabels=col_names, yticklabels=lang_names,
        cmap='Blues', linewidths=0.4, linecolor='#dddddd',
        annot_kws={'size': annot_fs},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_fs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fs)
    ax.set_title(f"Confusion Matrix — {lang} — Test Set  (acc={lang_acc:.1%})", fontsize=13, pad=12)
    ax.set_ylabel('True label', fontsize=11)
    ax.set_xlabel('Predicted label', fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{lang}_test.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()

    if has_other:
        cross = int(cm_lang[:, OTHER].sum())
        other_langs = [l for l in LANGUAGES if l != lang]
        print(f"  ⚠  {cross} sample(s) predicted outside {lang} (into {'/'.join(other_langs)} signs)\n")
    print(f"{lang} test confusion matrix saved to: {path}")

# Confidence distribution plot
fig, ax = plt.subplots(figsize=(10, 4))
correct_conf   = y_conf[y_true == y_pred]
incorrect_conf = y_conf[y_true != y_pred]

ax.hist(correct_conf, bins=20, alpha=0.7, color='steelblue', label=f'Correct ({len(correct_conf)})')
ax.hist(incorrect_conf, bins=20, alpha=0.7, color='salmon', label=f'Incorrect ({len(incorrect_conf)})')
ax.set_xlabel('Prediction Confidence (max softmax)')
ax.set_ylabel('Count')
ax.set_title('Confidence Distribution — Correct vs Incorrect Predictions')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
conf_path = os.path.join(OUTPUT_DIR, "confidence_distribution.png")
plt.savefig(conf_path, dpi=150)
plt.show()
print(f"Confidence distribution saved to: {conf_path}")

# Final summary
correct = int(np.sum(y_true == y_pred))
incorrect = len(y_true) - correct

print("\n── Test Complete ─────────────────────────────────────────────")
print(f"  Total samples: {len(y_true)}")
print(f"  Correct: {correct}  ({overall_acc:.1%})")
print(f"  Incorrect: {incorrect}")
print(f"  Avg confidence (correct): {correct_conf.mean():.4f}"   if len(correct_conf)   > 0 else "  No correct predictions")
print(f"  Avg confidence (incorrect): {incorrect_conf.mean():.4f}" if len(incorrect_conf) > 0 else "  No incorrect predictions")

print("\n  Per-language breakdown:")
for lang in LANGUAGES:
    lang_indices = [idx for sign, idx in label_map.items()
                    if sign.startswith(f"{lang}_") and idx in set(y_true)]
    if lang_indices:
        mask = np.isin(y_true, lang_indices)
        lang_acc = accuracy_score(y_true[mask], y_pred[mask])
        lang_n = int(mask.sum())
        lang_ok = int(np.sum(y_true[mask] == y_pred[mask]))
        print(f"    {lang:6s}: {lang_ok}/{lang_n} correct  ({lang_acc:.1%})")
    else:
        print(f"    {lang:6s}: no test samples")

print(f"\n  Output files in: {OUTPUT_DIR}")
print(f"    confusion_matrix_test.png")
for lang in LANGUAGES:
    print(f"    confusion_matrix_{lang}_test.png")
print(f"    confidence_distribution.png")
print(f"    test_results_per_sample.csv")
print(f"    test_results_summary.csv")
print("─────────────────────────────────────────────────────────────")
