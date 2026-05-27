# Real-Time Multilingual Sign Language Recognition

A real-time desktop application for recognising **American Sign Language (ASL)**, **British Sign Language (BSL)**, and a novel category of **cross-linguistically equivalent MUTUAL signs** within a single model, running entirely on consumer-grade hardware without a GPU or network dependency.

> Developed as a Software Engineering dissertation project at York St John University (2025–2026).

### Results Summary

| Condition | Overall Accuracy | Macro F1 | Total Errors |
|---|---|---|---|
| Aug-ON (primary) | 93.77% | 93.52% | 34 / 546 |
| Aug-OFF (baseline) | 86.45% | 84.94% | 74 / 546 |

| Language | Aug-ON Accuracy |
|---|---|
| ASL | 92.40% |
| BSL | 94.69% |
| MUTUAL | 96.08% |

---

## Repository Structure

```
├── collect_data.py                           # Data collection application
├── train_multilingual_model.py               # Model training script
├── test_model.py                             # Evaluation script
├── realtime_multilingual_inference.py        # Live inference desktop application
├── requirements.txt                          # Python dependencies
│
└── Final Training and Testing Results/
    ├── TestResults_AugON/
    │   ├── test_results_per_sample.csv       # Per-sample predictions and confidence scores
    │   ├── test_results_summary.csv          # Aggregate metrics
    │   ├── confidence_distribution.png
    │   ├── confusion_matrix_test.png         # Full 109 class confusion matrix
    │   ├── confusion_matrix_ASL_test.png
    │   ├── confusion_matrix_BSL_test.png
    │   └── confusion_matrix_MUTUAL_test.png
    │
    ├── TestResults_AugOFF/
    │   └── (same structure as above)
    │
    ├── TrainingResults_AugON/
    │   ├── training_curves.png
    │   ├── checkpoint_audit.csv              # Validation accuracy/loss at each checkpoint
    │   ├── training_confusion_matrix.png
    │   ├── confusion_matrix_ASL.png
    │   ├── confusion_matrix_BSL.png
    │   └── confusion_matrix_MUTUAL.png
    │
    └── TrainingResults_AugOFF/
        └── (same structure as above)
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Webcam (built-in or external)
- No GPU required

### Installation

```bash
git clone https://github.com/miguelgventur-ux/[YOUR_REPO_NAME.git](https://github.com/miguelgventur-ux/Real-Time-Multilingual-Sign-Language-Recognition-Application.git)
cd Real-Time-Multilingual-Sign-Language-Recognition-Application
pip install -r requirements.txt
```

### Usage

**1. Collect data**
```bash
python collect_data.py
```
Records isolated sign sequences via webcam using MediaPipe Hands. Each sequence captures 45 frames (~1.5 seconds). Signs are saved as 126-dimensional landmark arrays.

**2. Train the model**
```bash
python train_multilingual_model.py
```
Trains the stacked LSTM classifier across 109 sign classes. Supports both augmented (Aug-ON) and baseline (Aug-OFF) conditions. Model checkpoints are saved every 5 epochs; the best validation accuracy snapshot is selected post-hoc.

**3. Evaluate the model**
```bash
python test_model.py
```
Evaluates the trained model on the held-out test set. Outputs overall accuracy, macro-averaged precision/recall/F1, per-language breakdowns, confusion matrices, and confidence distribution plots.

**4. Run real-time inference**
```bash
python realtime_multilingual_inference.py
```
Launches the live desktop application. Point your webcam at your hands — the predicted sign and its language of origin (ASL / BSL / MUTUAL) are displayed on screen with a colour-coded language badge.

---

## Model Architecture

A stacked LSTM network implemented in TensorFlow/Keras:

| Layer | Configuration |
|---|---|
| Input | (45, 126) |
| Masking | mask_value = 0.0 |
| LSTM 1 | 128 units, return_sequences=True, dropout=0.3 |
| LSTM 2 | 64 units, dropout=0.3 |
| Dense | 64 units, ReLU |
| Dropout | rate = 0.3 |
| Output | 110 units, Softmax |

---

## The MUTUAL Class

A core contribution of this project is the identification and modelling of signs that are phonologically equivalent across ASL and BSL — referred to as the **MUTUAL** class. Rather than maintaining separate per-language entries for these signs, training samples from both communities are pooled under a single language-independent label.

The 10 MUTUAL signs are: `me`, `you`, `drink`, `eat`, `baby`, `phone`, `car`, `cold`, `money`, `stand`.

This design achieved **96.08% test accuracy** on the MUTUAL subset — the highest of the three language groups — and was invariant between augmented and non-augmented conditions, validating the pooling strategy.

---

## Hardware

All data collection, training, and inference was performed on a consumer-grade laptop:

- **CPU:** Intel Core i5-1035G7 (1.50 GHz)
- **RAM:** 8 GB
- **GPU:** Intel Iris Plus (integrated, no discrete acceleration)

This was a deliberate design constraint to validate the accessibility objective of the project.

---

## Dependencies

Key libraries (see `requirements.txt` for full list):

- `mediapipe` — hand landmark extraction
- `tensorflow` / `keras` — model training and inference
- `opencv-python` — webcam capture and UI rendering
- `numpy`, `scikit-learn`, `matplotlib` — data processing and evaluation

---

## License

This project was developed for academic purposes at York St John University. Please contact the author before reusing any component.
