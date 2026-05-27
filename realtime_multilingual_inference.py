"""
Real-Time Sign Language Recognition
Detects ASL, BSL, and MUTUAL signs from a single trained model:
  - ASL predictions shown in blue
  - BSL predictions shown in yellow
  - MUTUAL predictions shown in purple
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import collections
import time
import os

# Configuration
MODEL_DIR = r"C:\Diss\Models\Dual_60_Model"
CONF_THRESHOLD = 0.2
SMOOTH_WINDOW = 5
SETTLE_DELAY = 0.5    # seconds before showing prediction after hand appears

# Colours for convinienve
GREEN = (0, 255, 100)
RED = (0, 60, 255)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
DARK = (20, 20, 20)
BLUE = (255, 100, 0)
GREY = (120, 120, 120)
DIM = (60, 60, 60)
PURPLE = (220, 60, 220)

LANG_COLOUR = {
    "ASL"    : BLUE,
    "BSL"    : YELLOW,
    "MUTUAL" : PURPLE,
}

# Load the model
print("Loading model...")

with open(os.path.join(MODEL_DIR, "inference_config.json")) as f:
    config = json.load(f)

SEQUENCE_LEN = config["sequence_len"]
NUM_FEATURES = config["num_features"]
label_map = config["label_map"]
id_to_sign = {int(v): k for k, v in label_map.items()}

feat_mean = np.load(os.path.join(MODEL_DIR, "feat_mean.npy"))
feat_std  = np.load(os.path.join(MODEL_DIR, "feat_std.npy"))

from tensorflow.keras import layers

def build_model(seq_len, num_features, num_classes):
    inp = tf.keras.Input(shape=(seq_len, num_features))
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
    x = layers.LSTM(64, dropout=0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(label_map), activation='softmax')(x)
    return tf.keras.Model(inp, out)

model = build_model(SEQUENCE_LEN, NUM_FEATURES, len(label_map))
model.load_weights(os.path.join(MODEL_DIR, "best_model.h5"))
print(f"Model loaded. {len(label_map)} classes.\n")

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# The helpers
def extract_landmarks(frame_rgb):
    result = hands.process(frame_rgb)
    hand1 = np.zeros(63, dtype=np.float32)
    hand2 = np.zeros(63, dtype=np.float32)
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


def normalise(seq):
    out = (seq - feat_mean) / feat_std
    zero_frames = (seq == 0).all(axis=-1, keepdims=True)
    return np.where(zero_frames, 0.0, out)


def predict(frame_buffer):
    #Return the top prediction.(e.g..'book', 0.87, 'BSL')
    seq = np.array(frame_buffer, dtype=np.float32)
    seq = normalise(seq)
    inp = seq[np.newaxis, ...]
    probs = model.predict(inp, verbose=0)[0]

    best_idx = int(np.argmax(probs))
    best_conf = float(probs[best_idx])
    label = id_to_sign[best_idx]

    if "_" in label:
        detected_lang, sign = label.split("_", 1)
    else:
        detected_lang, sign = "", label
    return sign, best_conf, detected_lang

# Drawing
def draw_top_bar(frame, hand_count, current_sign, current_conf,
                 show_prediction, detected_lang):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 100), DARK, -1)

    lang_colour = LANG_COLOUR.get(detected_lang.upper(), GREY)

    if show_prediction:
        sign_colour = lang_colour if current_sign != "???" else YELLOW
        cv2.putText(frame, current_sign, (15, 62),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, sign_colour, 2, cv2.LINE_AA)

        if detected_lang:
            badge_text = f"  {detected_lang}  "
            (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(frame, (13, 72), (15 + bw, 75 + bh), lang_colour, -1)
            cv2.putText(frame, badge_text, (15, 90),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, DARK, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "---", (15, 62),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, DIM, 2, cv2.LINE_AA)

    if show_prediction:
        cv2.putText(frame, f"{current_conf:.0%} confidence", (w - 260, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, GREY, 1, cv2.LINE_AA)

    if hand_count == 2:
        hand_text, hand_colour = "2 hands", GREEN
    elif hand_count == 1:
        hand_text, hand_colour = "1 hand",  YELLOW
    else:
        hand_text, hand_colour = "no hands", RED
    cv2.putText(frame, hand_text, (w - 260, 72),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, hand_colour, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 100), (w, 103), (50, 50, 50), -1)




def draw_bottom_bar(frame, buffer_fill):
    h, w = frame.shape[:2]

    bar_w = int((w - 40) * buffer_fill)
    cv2.rectangle(frame, (20, h - 38), (w - 20, h - 32), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h - 38), (20 + bar_w, h - 32), GREEN, -1)

    cv2.rectangle(frame, (0, h - 30), (w, h), DARK, -1)
    cv2.putText(frame, "Q: quit",
                (10, h - 10), cv2.FONT_HERSHEY_DUPLEX,
                0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # Language legend on the bottom right of the UI
    legend_items = [("ASL", BLUE), ("BSL", YELLOW), ("MUTUAL", PURPLE)]
    x_cursor = w - 10
    for label, colour in reversed(legend_items):
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
        x_cursor -= lw + 14
        cv2.putText(frame, label, (x_cursor, h - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, colour, 1, cv2.LINE_AA)



def draw_processing_dot(frame):
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (frame.shape[1] - 30, 148), 8, GREEN, -1)


# Main loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW_NAME = "Sign Language Recognition"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

frame_buffer = collections.deque(maxlen=SEQUENCE_LEN)
recent_preds = collections.deque(maxlen=SMOOTH_WINDOW)
current_sign = "---"
current_conf = 0.0
detected_lang = ""
hand_appeared_at = None

print("Webcam running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    # Predict once buffer is full
    if len(frame_buffer) == SEQUENCE_LEN:
        sign, conf, detected_lang = predict(frame_buffer)
        recent_preds.append(sign)

        counter = collections.Counter(recent_preds)
        top_sign, top_count = counter.most_common(1)[0]

        if top_count >= 4 and conf >= CONF_THRESHOLD:
            current_sign = top_sign.upper()
            current_conf = conf
        elif conf < CONF_THRESHOLD:
            current_sign = "???"
            current_conf = conf

    # Drawi UI
    draw_top_bar(frame, len(hand_landmarks_list), current_sign,
                 current_conf, hand_settled, detected_lang)
    draw_bottom_bar(frame, len(frame_buffer) / SEQUENCE_LEN)
    draw_processing_dot(frame)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done.")
