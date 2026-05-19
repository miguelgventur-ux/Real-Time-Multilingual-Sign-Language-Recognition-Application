"Sign Language Data Collection Script"

import cv2
import numpy as np
import mediapipe as mp
import os
import time

# Signs that differ between ASL and BSL — record separately for each language
SIGNS = [
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
    'give', 'take', 'look', 'hear', 'smile'
]

# Signs whose production is identical (or near-identical) in both ASL and BSL.
MUTUAL_SIGNS = [
    'me', 'you', 'drink', 'eat','money',
    'car', 'cold', 'phone', 'stand', 'baby'
]

LANGUAGE = "MUTUAL"  # change to ASL, BSL, or MUTUAL

# Derive the active sign list from the chosen language
if LANGUAGE == "MUTUAL":
    ACTIVE_SIGNS = MUTUAL_SIGNS
else:
    # Exclude mutual signs so they are not re-recorded under a language directory;
    # remove the next line if you want to record every sign regardless.
    ACTIVE_SIGNS = [s for s in SIGNS if s not in MUTUAL_SIGNS]

OUTPUT_DIR = r"C:\Diss\CollectedDataTesting"
RECORDINGS_PER_SIGN = 5
SEQUENCE_LEN = 45
COUNTDOWN_SECONDS = 2          # recording countdown
MIN_HAND_THRESHOLD = 0.60      # minimum fraction of frames with hands visible

# ── SETUP ─────────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

GREEN  = (0, 255, 100)
RED    = (0, 60, 255)
YELLOW = (0, 220, 255)
WHITE  = (255, 255, 255)
DARK   = (20, 20, 20)
BLUE   = (255, 100, 0)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def extract_two_hand_landmarks(frame_rgb):
    """Returns (126,) array, sorted by wrist x so ordering is consistent.
    Missing hand slots are filled with zeros."""
    result       = hands.process(frame_rgb)
    hand1        = np.zeros(63, dtype=np.float32)
    hand2        = np.zeros(63, dtype=np.float32)
    raw_landmarks = []

    if result.multi_hand_landmarks:
        detected      = sorted(result.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
        raw_landmarks = detected

        if len(detected) >= 1:
            hand1 = np.array(
                [[lm.x, lm.y, lm.z] for lm in detected[0].landmark], dtype=np.float32
            ).flatten()
        if len(detected) >= 2:
            hand2 = np.array(
                [[lm.x, lm.y, lm.z] for lm in detected[1].landmark], dtype=np.float32
            ).flatten()

    return np.concatenate([hand1, hand2]), len(raw_landmarks), raw_landmarks


def quality_check(sequence):
    """Checks that at least one hand was visible for most of the recording.
    Returns (passed, hand_coverage, message)."""
    first_hand   = sequence[:, :63]
    has_any_hand = np.any(first_hand != 0, axis=1).mean()

    if has_any_hand < MIN_HAND_THRESHOLD:
        return False, float(has_any_hand), (
            f"Hand only visible in {has_any_hand:.0%} of frames "
            f"(need {MIN_HAND_THRESHOLD:.0%}) — try again"
        )
    return True, float(has_any_hand), f"Good — hand visible in {has_any_hand:.0%} of frames"


def normalise_length(seq, target_len):
    n = len(seq)
    if n == 0:
        return np.zeros((target_len, 126), dtype=np.float32)
    indices = np.linspace(0, n - 1, target_len).astype(int)
    return seq[indices]


# ── DRAWING ───────────────────────────────────────────────────────────────────

def draw_top_bar(frame, sign, sign_idx, total_signs, saved_count, state, hand_count):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 100), DARK, -1)

    label = f"{LANGUAGE}  |  Sign: {sign.upper()}  [{sign_idx + 1}/{total_signs}]"
    cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

    cv2.putText(frame, f"Saved: {saved_count}/{RECORDINGS_PER_SIGN}",
                (w - 240, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, GREEN, 2, cv2.LINE_AA)

    state_colour = RED if state == "RECORDING" else (YELLOW if state == "COUNTDOWN" else GREEN)
    cv2.putText(frame, state, (15, 72), cv2.FONT_HERSHEY_DUPLEX, 0.75, state_colour, 2, cv2.LINE_AA)

    if hand_count == 2:
        hand_text, hand_colour = "2 hands", GREEN
    elif hand_count == 1:
        hand_text, hand_colour = "1 hand",  YELLOW
    else:
        hand_text, hand_colour = "no hands", RED
    cv2.putText(frame, hand_text, (w - 240, 72), cv2.FONT_HERSHEY_DUPLEX, 0.7, hand_colour, 2, cv2.LINE_AA)


def draw_message_bar(frame, message):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 100), (w, 132), (40, 40, 40), -1)
    cv2.putText(frame, message, (15, 123), cv2.FONT_HERSHEY_DUPLEX, 0.55, WHITE, 1, cv2.LINE_AA)


def draw_controls(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), DARK, -1)
    controls = "SPACE: start recording   S: save   D: discard   N: next sign   Q: quit"
    cv2.putText(frame, controls, (10, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


def draw_progress_bar(frame, saved_count):
    h, w = frame.shape[:2]
    progress = min(saved_count / RECORDINGS_PER_SIGN, 1.0)
    bar_w    = int((w - 40) * progress)
    cv2.rectangle(frame, (20, h - 35), (w - 20,      h - 32), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h - 35), (20 + bar_w,  h - 32), GREEN,        -1)


def draw_recording_dot(frame):
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (frame.shape[1] - 30, 148), 10, RED, -1)


def draw_countdown(frame, seconds_remaining):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.rectangle(frame, (0, 0), (w, 3), YELLOW, -1)

    text = f"{seconds_remaining:.1f}"
    font_scale, thickness = 5, 6
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    cx = (w - tw) // 2
    cy = (h + th) // 2 - 30
    cv2.putText(frame, text, (cx + 3, cy + 3), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0),  thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (cx,     cy),     cv2.FONT_HERSHEY_DUPLEX, font_scale, BLUE,        thickness,     cv2.LINE_AA)


def draw_review_overlay(frame, passed, message):
    h, w = frame.shape[:2]
    colour = GREEN if passed else RED
    cv2.rectangle(frame, (0, 132), (w, 165), (30, 30, 30), -1)
    cv2.putText(frame, message, (15, 156), cv2.FONT_HERSHEY_DUPLEX, 0.6, colour, 1, cv2.LINE_AA)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WINDOW_NAME = "Sign Language Data Collection"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print(f"\nData collection started  |  Language: {LANGUAGE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Signs ({len(ACTIVE_SIGNS)}): {ACTIVE_SIGNS}\n")

    sign_idx       = 0
    state          = "READY"
    frame_buffer   = []
    countdown_start = None
    review_seq     = None
    review_result  = None
    message        = "Press SPACE to start countdown"
    skip_warned    = None

    while sign_idx < len(ACTIVE_SIGNS):
        sign     = ACTIVE_SIGNS[sign_idx]
        sign_dir = os.path.join(OUTPUT_DIR, LANGUAGE, sign)
        os.makedirs(sign_dir, exist_ok=True)
        existing    = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
        saved_count = len(existing)

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, hand_count, raw_lms = extract_two_hand_landmarks(rgb)

        skeleton_colours = [
            ((0, 255, 100), (255, 255, 255)),   # first hand:  green
            ((255, 100, 0), (200, 255, 255)),    # second hand: blue
        ]
        for i, lm in enumerate(raw_lms):
            dot_col, line_col = skeleton_colours[i % 2]
            mp_draw.draw_landmarks(
                frame, lm, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=dot_col,  thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=line_col, thickness=2),
            )

        # ── STATE LOGIC ───────────────────────────────────────────────────────

        if state == "COUNTDOWN":
            elapsed   = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - elapsed

            if remaining > 0:
                draw_countdown(frame, remaining)
                message = "Get your hands ready..."
            else:
                state        = "RECORDING"
                frame_buffer = []
                message      = f"Recording... 0/{SEQUENCE_LEN} frames"

        elif state == "RECORDING":
            frame_buffer.append(landmarks.copy())
            draw_recording_dot(frame)
            message = f"Recording... {len(frame_buffer)}/{SEQUENCE_LEN} frames"

            if len(frame_buffer) >= SEQUENCE_LEN:
                state      = "REVIEW"
                review_seq = np.array(frame_buffer, dtype=np.float32)
                passed, coverage, msg = quality_check(review_seq)
                review_result = (passed, coverage, msg)
                if passed:
                    message = f"{msg}  —  press S to save or D to discard"
                else:
                    message = f"{msg}  —  press D to discard and try again"

        elif state == "REVIEW" and review_seq is not None:
            passed, coverage, msg = review_result
            draw_review_overlay(frame, passed, msg)

        # ── DRAW UI ───────────────────────────────────────────────────────────

        draw_top_bar(frame, sign, sign_idx, len(ACTIVE_SIGNS), saved_count, state, hand_count)
        draw_message_bar(frame, message)
        draw_controls(frame)
        draw_progress_bar(frame, saved_count)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        # ── KEY HANDLERS ──────────────────────────────────────────────────────

        if key == ord('q'):
            print("\nQuitting.")
            break

        elif key == ord(' '):
            if state == "READY":
                state           = "COUNTDOWN"
                countdown_start = time.time()
                message         = "Get your hands ready..."

        elif key == ord('s'):
            if state == "REVIEW" and review_seq is not None:
                passed, coverage, msg = review_result
                if not passed:
                    message = f"Cannot save — {msg}. Press D to discard and retry."
                else:
                    seq_fixed = normalise_length(review_seq, SEQUENCE_LEN)
                    existing  = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
                    filename  = f"{sign}_{LANGUAGE}_{len(existing):03d}.npy"
                    save_path = os.path.join(sign_dir, filename)
                    np.save(save_path, seq_fixed)
                    saved_count += 1
                    print(f"  Saved: {filename}  ({saved_count}/{RECORDINGS_PER_SIGN})")

                    state        = "READY"
                    review_seq   = None
                    frame_buffer = []

                    if saved_count >= RECORDINGS_PER_SIGN:
                        message = f"Done with {sign}! Press N for next sign."
                    else:
                        message = (
                            f"Saved! {RECORDINGS_PER_SIGN - saved_count} to go. "
                            f"Press SPACE to record again."
                        )

        elif key == ord('d'):
            if state in ("REVIEW", "RECORDING", "COUNTDOWN"):
                state        = "READY"
                frame_buffer = []
                review_seq   = None
                message      = "Discarded. Press SPACE to try again."

        elif key == ord('n'):
            if saved_count < RECORDINGS_PER_SIGN:
                if skip_warned == sign:
                    sign_idx   += 1
                    skip_warned = None
                    state        = "READY"
                    frame_buffer = []
                    review_seq   = None
                    message      = "Press SPACE to start countdown"
                else:
                    skip_warned = sign
                    message = (
                        f"Only {saved_count}/{RECORDINGS_PER_SIGN} saved. "
                        f"Press N again to skip anyway."
                    )
            else:
                sign_idx   += 1
                skip_warned = None
                state        = "READY"
                frame_buffer = []
                review_seq   = None
                message      = "Press SPACE to start countdown"

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\n── Collection complete ──")
    for sign in ACTIVE_SIGNS:
        sign_dir = os.path.join(OUTPUT_DIR, LANGUAGE, sign)
        if os.path.exists(sign_dir):
            count = len([f for f in os.listdir(sign_dir) if f.endswith('.npy')])
            print(f"  {sign:12s}: {count} recordings")
    print(f"\nData saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
