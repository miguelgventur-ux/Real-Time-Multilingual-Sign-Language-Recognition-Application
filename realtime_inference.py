"""
Real-Time Sign Language Recognition — Bilingual Webcam Inference
Run from terminal: python realtime_inference.py
Press L to toggle between ASL and BSL.
Press Q to quit.
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import collections
import time
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
ASL_MODEL_DIR  = r"C:\Diss\ModelASL20"
BSL_MODEL_DIR  = r"C:\Diss\ModelBSL10"
CONF_THRESHOLD = 0.4
SMOOTH_WINDOW  = 5
SETTLE_DELAY   = 0.5
TEMPERATURE    = 2   # soften confidence — raise if still overconfident

# ── COLOURS (BGR) ─────────────────────────────────────────────────────────────
GREEN  = (0, 255, 100)
RED    = (0, 60, 255)
YELLOW = (0, 220, 255)
WHITE  = (255, 255, 255)
DARK   = (20, 20, 20)
BLUE   = (255, 100, 0)
GREY   = (120, 120, 120)
DIM    = (60, 60, 60)

# ── MODEL LOADER ──────────────────────────────────────────────────────────────
from tensorflow.keras import layers

def build_model(seq_len, num_features, num_classes):
    inp = tf.keras.Input(shape=(seq_len, num_features))
    x   = layers.Masking(mask_value=0.0)(inp)
    x   = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x   = layers.LSTM(64, dropout=0.3)(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out)


def load_language_model(model_dir, label):
    print(f"Loading {label} model from {model_dir}...")
    with open(os.path.join(model_dir, "inference_config.json")) as f:
        cfg = json.load(f)

    seq_len      = cfg["sequence_len"]
    num_features = cfg["num_features"]
    lmap         = cfg["label_map"]
    id_to_sign   = {v: k for k, v in lmap.items()}
    feat_mean    = np.load(os.path.join(model_dir, "feat_mean.npy"))
    feat_std     = np.load(os.path.join(model_dir, "feat_std.npy"))

    m = build_model(seq_len, num_features, len(lmap))
    m.load_weights(os.path.join(model_dir, "best_model.h5"))
    print(f"  {label} loaded. Classes: {list(lmap.keys())}")

    return {
        "model"      : m,
        "seq_len"    : seq_len,
        "num_features": num_features,
        "label_map"  : lmap,
        "id_to_sign" : id_to_sign,
        "feat_mean"  : feat_mean,
        "feat_std"   : feat_std,
        "label"      : label,
    }


models = {
    "ASL": load_language_model(ASL_MODEL_DIR, "ASL"),
    "BSL": load_language_model(BSL_MODEL_DIR, "BSL"),
}

# ── MEDIAPIPE SETUP ───────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def extract_landmarks(frame_rgb):
    result = hands.process(frame_rgb)
    hand1  = np.zeros(63, dtype=np.float32)
    hand2  = np.zeros(63, dtype=np.float32)
    all_landmarks = []

    if result.multi_hand_landmarks:
        detected = sorted(
            result.multi_hand_landmarks,
            key=lambda h: h.landmark[0].x
        )
        all_landmarks = detected
        if len(detected) >= 1:
            hand1 = np.array(
                [[lm.x, lm.y, lm.z] for lm in detected[0].landmark],
                dtype=np.float32
            ).flatten()
        if len(detected) >= 2:
            hand2 = np.array(
                [[lm.x, lm.y, lm.z] for lm in detected[1].landmark],
                dtype=np.float32
            ).flatten()

    return np.concatenate([hand1, hand2]), all_landmarks


def normalise(seq, feat_mean, feat_std):
    out         = (seq - feat_mean) / feat_std
    zero_frames = (seq == 0).all(axis=-1, keepdims=True)
    return np.where(zero_frames, 0.0, out)


def predict_with(frame_buffer, lang_data):
    seq   = np.array(frame_buffer, dtype=np.float32)
    seq   = normalise(seq, lang_data["feat_mean"], lang_data["feat_std"])
    inp   = seq[np.newaxis, ...]
    probs = lang_data["model"].predict(inp, verbose=0)[0]
    idx   = np.argmax(probs)
    return lang_data["id_to_sign"][idx], float(probs[idx])


# ── DRAWING ───────────────────────────────────────────────────────────────────
def draw_top_bar(frame, hand_count, current_sign, current_conf,
                 show_prediction, active_lang):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 100), DARK, -1)

    # Sign prediction
    if show_prediction:
        sign_colour = GREEN if current_sign != "???" else YELLOW
        cv2.putText(frame, current_sign, (15, 68),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, sign_colour, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "---", (15, 68),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, DIM, 2, cv2.LINE_AA)

    # Confidence
    if show_prediction:
        cv2.putText(frame, f"{current_conf:.0%} confidence", (w - 260, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, GREY, 1, cv2.LINE_AA)

    # Hand count
    if hand_count == 2:
        hand_text, hand_colour = "2 hands", GREEN
    elif hand_count == 1:
        hand_text, hand_colour = "1 hand",  YELLOW
    else:
        hand_text, hand_colour = "no hands", RED
    cv2.putText(frame, hand_text, (w - 260, 72),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, hand_colour, 1, cv2.LINE_AA)

    # Accent line
    cv2.rectangle(frame, (0, 100), (w, 103), (50, 50, 50), -1)


def draw_language_toggle(frame, active_lang):
    """Draw ASL / BSL toggle buttons in the bottom bar."""
    h, w = frame.shape[:2]
    btn_w, btn_h = 90, 28
    margin       = 10
    y_top        = h - btn_h - 4
    centre_x     = w // 2

    for i, lang in enumerate(["ASL", "BSL"]):
        x_left = centre_x - btn_w - margin // 2 + i * (btn_w + margin)
        is_active = lang == active_lang

        bg_colour   = GREEN if is_active else DIM
        text_colour = DARK  if is_active else GREY

        cv2.rectangle(frame, (x_left, y_top),
                      (x_left + btn_w, y_top + btn_h), bg_colour, -1)
        cv2.rectangle(frame, (x_left, y_top),
                      (x_left + btn_w, y_top + btn_h), (80, 80, 80), 1)

        (tw, th), _ = cv2.getTextSize(lang, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        tx = x_left + (btn_w - tw) // 2
        ty = y_top  + (btn_h + th) // 2 - 2
        cv2.putText(frame, lang, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, text_colour, 1, cv2.LINE_AA)


def draw_bottom_bar(frame, buffer_fill, active_lang):
    h, w = frame.shape[:2]

    # Buffer progress bar
    bar_w = int((w - 40) * buffer_fill)
    cv2.rectangle(frame, (20, h - 38), (w - 20, h - 32), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h - 38), (20 + bar_w, h - 32), GREEN, -1)

    # Bottom controls bar
    cv2.rectangle(frame, (0, h - 30), (w, h), DARK, -1)
    cv2.putText(frame, "L: toggle language   Q: quit",
                (10, h - 10), cv2.FONT_HERSHEY_DUPLEX,
                0.45, (160, 160, 160), 1, cv2.LINE_AA)

    draw_language_toggle(frame, active_lang)


def draw_processing_dot(frame):
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (frame.shape[1] - 30, 148), 8, GREEN, -1)


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW_NAME = "Sign Language Recognition"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

active_lang     = "ASL"
frame_buffer    = collections.deque(maxlen=models[active_lang]["seq_len"])
recent_preds    = collections.deque(maxlen=SMOOTH_WINDOW)
current_sign    = "---"
current_conf    = 0.0
hand_appeared_at = None

print("Webcam running. Press L to toggle language. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lang_data = models[active_lang]

    landmarks, hand_landmarks_list = extract_landmarks(rgb)
    frame_buffer.append(landmarks)

    # Settle timer
    if len(hand_landmarks_list) > 0:
        if hand_appeared_at is None:
            hand_appeared_at = time.time()
    else:
        hand_appeared_at = None

    hand_settled = (
        hand_appeared_at is not None and
        (time.time() - hand_appeared_at) >= SETTLE_DELAY
    )

    # Draw hand skeletons
    skeleton_colours = [
        ((0, 255, 100), (255, 255, 255)),
        ((255, 100, 0), (200, 255, 255)),
    ]
    for i, hand_lm in enumerate(hand_landmarks_list):
        dot_col, line_col = skeleton_colours[i % 2]
        mp_draw.draw_landmarks(
            frame, hand_lm, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=dot_col,  thickness=2, circle_radius=4),
            mp_draw.DrawingSpec(color=line_col, thickness=2)
        )

    # Predict using active model only
    if len(frame_buffer) == lang_data["seq_len"]:
        sign, conf = predict_with(frame_buffer, lang_data)
        recent_preds.append(sign)

        counter = collections.Counter(recent_preds)
        top_sign, top_count = counter.most_common(1)[0]

        # Require majority agreement in the smoothing window before showing
        if top_count >= 4 and conf >= CONF_THRESHOLD:
            current_sign = top_sign.upper()
            current_conf = conf
        elif conf < CONF_THRESHOLD:
            current_sign = "???"
            current_conf = conf

    # ── DRAW UI ───────────────────────────────────────────────────────────────
    draw_top_bar(frame, len(hand_landmarks_list), current_sign,
                 current_conf, hand_settled, active_lang)
    draw_bottom_bar(frame, len(frame_buffer) / lang_data["seq_len"], active_lang)
    draw_processing_dot(frame)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('l'):
        # Toggle language and reset state
        active_lang      = "BSL" if active_lang == "ASL" else "ASL"
        frame_buffer     = collections.deque(maxlen=models[active_lang]["seq_len"])
        recent_preds.clear()
        current_sign     = "---"
        current_conf     = 0.0
        hand_appeared_at = None
        print(f"Switched to {active_lang}")

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done.")
