import os
import time
import urllib.request
from collections import deque, Counter
import warnings

import cv2
import joblib
import mediapipe as mp
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ─────────────────────────────
# Paths
# ─────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

ALPHABET_MODEL_PATH = os.path.join("models", "alphabet", "fsl_alphabet_model.joblib")
MOTION_MODEL_PATH = os.path.join("models", "alphabet", "fsl_alphabet_motion_model.joblib")

NUMBER_MODEL_PATH = os.path.join("models", "numbers", "fsl_number_model.joblib")
PHRASE_MODEL_PATH = os.path.join("models", "phrases", "fsl_phrase_model.joblib")

# ─────────────────────────────
# Camera / Model Settings
# ─────────────────────────────
CAMERA_INDEX = 0

SEQUENCE_LENGTH = 30
PHRASE_MAX_HANDS = 2

STATIC_CONFIDENCE_THRESHOLD = 0.60
MOTION_CONFIDENCE_THRESHOLD = 0.75
MOTION_MOVEMENT_THRESHOLD = 0.15

NUMBER_CONFIDENCE_THRESHOLD = 0.60

PHRASE_CONFIDENCE_THRESHOLD = 0.70
PHRASE_MOVEMENT_THRESHOLD = 0.00

MOTION_HOLD_SECONDS = 2.0
MENU_HOLD_SECONDS = 1.2

MODE_MENU = "MENU"
MODE_ALPHABET = "ALPHABET"
MODE_NUMBERS = "NUMBERS"
MODE_PHRASES = "PHRASES"

PHRASE_EXPECTED_FEATURES = SEQUENCE_LENGTH * PHRASE_MAX_HANDS * 21 * 3

# ─────────────────────────────
# Download MediaPipe model if needed
# ─────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ─────────────────────────────
# Load Classifiers
# ─────────────────────────────
if not os.path.exists(ALPHABET_MODEL_PATH):
    print("No trained alphabet model found.")
    print("Run collect_alphabet_data.py first, then train_alphabet_model.py.")
    exit()

alphabet_classifier = joblib.load(ALPHABET_MODEL_PATH)
print("Alphabet model loaded.")

motion_classifier = None

if os.path.exists(MOTION_MODEL_PATH):
    motion_classifier = joblib.load(MOTION_MODEL_PATH)
    print("Alphabet motion model loaded. J and Z detection enabled.")
else:
    print("No alphabet motion model found. J and Z movement detection disabled.")

number_classifier = None

if os.path.exists(NUMBER_MODEL_PATH):
    number_classifier = joblib.load(NUMBER_MODEL_PATH)
    print("Number model loaded. Number detection enabled.")
else:
    print("No number model found. Option 2 will open but detection is disabled.")

phrase_classifier = None
phrase_model_compatible = False

if os.path.exists(PHRASE_MODEL_PATH):
    phrase_classifier = joblib.load(PHRASE_MODEL_PATH)
    print("Phrase model loaded.")

    try:
        phrase_feature_count = phrase_classifier.n_features_in_
    except Exception:
        try:
            phrase_feature_count = phrase_classifier.named_steps["scaler"].n_features_in_
        except Exception:
            phrase_feature_count = None

    if phrase_feature_count is None:
        phrase_model_compatible = True
        print("Phrase model feature count could not be checked, but it will be used.")
    elif phrase_feature_count == PHRASE_EXPECTED_FEATURES:
        phrase_model_compatible = True
        print("Phrase model is compatible with 2-hand phrase detection.")
    else:
        phrase_model_compatible = False
        print("")
        print("WARNING: Phrase model is not compatible with this 2-hand main.py.")
        print(f"Expected features: {PHRASE_EXPECTED_FEATURES}")
        print(f"Model features: {phrase_feature_count}")
        print("This usually means the phrase model was trained using the old 1-hand phrase collector.")
        print("Recollect phrase data using the 2-hand collect_phrase_data.py, then run train_phrase_model.py.")
        print("")
else:
    print("No phrase model found. Option 3 will open but detection is disabled.")

# ─────────────────────────────
# MediaPipe Setup
# ─────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────
# Prediction Buffers
# ─────────────────────────────
alphabet_history = deque(maxlen=10)
number_history = deque(maxlen=10)
phrase_history = deque(maxlen=5)

motion_buffer = deque(maxlen=SEQUENCE_LENGTH)

# ─────────────────────────────
# Landmark Helpers
# ─────────────────────────────
def normalize_landmarks(hand_landmarks):
    wrist = hand_landmarks[0]

    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    features = []

    for lm in hand_landmarks:
        features.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale
        ])

    return features


def raw_landmarks(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]


def get_hand_center_x(hand_points):
    return sum(point[0] for point in hand_points) / len(hand_points)


def get_two_hand_frame(all_hand_landmarks):
    hands = []

    for hand_landmarks in all_hand_landmarks:
        hands.append(raw_landmarks(hand_landmarks))

    hands.sort(key=get_hand_center_x)

    while len(hands) < PHRASE_MAX_HANDS:
        hands.append(None)

    return hands[:PHRASE_MAX_HANDS]


def sequence_to_features(sequence):
    first_frame = sequence[0]
    wrist0 = first_frame[0]

    xs = [p[0] for p in first_frame]
    ys = [p[1] for p in first_frame]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    features = []

    for frame in sequence:
        for x, y, z in frame:
            features.extend([
                (x - wrist0[0]) / scale,
                (y - wrist0[1]) / scale,
                (z - wrist0[2]) / scale
            ])

    return features


def sequence_to_phrase_features(sequence):
    first_frame = sequence[0]

    visible_points = []

    for hand in first_frame:
        if hand is not None:
            visible_points.extend(hand)

    if not visible_points:
        return [0.0] * PHRASE_EXPECTED_FEATURES

    anchor = visible_points[0]

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    features = []

    for frame in sequence:
        for hand in frame:
            if hand is None:
                for _ in range(21):
                    features.extend([0.0, 0.0, 0.0])
            else:
                for x, y, z in hand:
                    features.extend([
                        (x - anchor[0]) / scale,
                        (y - anchor[1]) / scale,
                        (z - anchor[2]) / scale
                    ])

    return features


def calculate_movement(sequence):
    if len(sequence) < 2:
        return 0

    first_frame = sequence[0]

    xs = [p[0] for p in first_frame]
    ys = [p[1] for p in first_frame]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    important_points = [0, 4, 8, 12, 16, 20]
    max_dist = 0

    for frame in sequence:
        for point_id in important_points:
            dx = (frame[point_id][0] - first_frame[point_id][0]) / scale
            dy = (frame[point_id][1] - first_frame[point_id][1]) / scale
            dist = (dx * dx + dy * dy) ** 0.5
            max_dist = max(max_dist, dist)

    return max_dist


def calculate_phrase_movement(sequence):
    if len(sequence) < 2:
        return 0

    first_frame = sequence[0]

    visible_points = []

    for hand in first_frame:
        if hand is not None:
            visible_points.extend(hand)

    if not visible_points:
        return 0

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    important_points = [0, 4, 8, 12, 16, 20]
    max_dist = 0

    for frame in sequence:
        for hand_index, hand in enumerate(frame):
            first_hand = first_frame[hand_index]

            if hand is None or first_hand is None:
                continue

            for point_id in important_points:
                dx = (hand[point_id][0] - first_hand[point_id][0]) / scale
                dy = (hand[point_id][1] - first_hand[point_id][1]) / scale
                dist = (dx * dx + dy * dy) ** 0.5
                max_dist = max(max_dist, dist)

    return max_dist

# ─────────────────────────────
# Prediction Helpers
# ─────────────────────────────
def get_stable_prediction(history):
    if not history:
        return ""

    most_common = Counter(history).most_common(1)
    return most_common[0][0]


def predict_static(classifier, hand_landmarks):
    features = normalize_landmarks(hand_landmarks)
    features = np.array(features).reshape(1, -1)

    probabilities = classifier.predict_proba(features)[0]
    max_index = np.argmax(probabilities)

    label = classifier.classes_[max_index]
    confidence = probabilities[max_index]

    return label, confidence


def predict_motion(classifier, sequence):
    features = sequence_to_features(sequence)
    features = np.array(features).reshape(1, -1)

    probabilities = classifier.predict_proba(features)[0]
    max_index = np.argmax(probabilities)

    label = classifier.classes_[max_index]
    confidence = probabilities[max_index]

    return label, confidence


def predict_phrase_motion(classifier, sequence):
    features = sequence_to_phrase_features(sequence)
    features = np.array(features).reshape(1, -1)

    probabilities = classifier.predict_proba(features)[0]
    max_index = np.argmax(probabilities)

    label = classifier.classes_[max_index]
    confidence = probabilities[max_index]

    return label, confidence

# ─────────────────────────────
# Menu Helpers
# ─────────────────────────────
def is_finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks[tip_id].y < hand_landmarks[pip_id].y


def detect_menu_option(hand_landmarks):
    index_up = is_finger_up(hand_landmarks, 8, 6)
    middle_up = is_finger_up(hand_landmarks, 12, 10)
    ring_up = is_finger_up(hand_landmarks, 16, 14)
    pinky_up = is_finger_up(hand_landmarks, 20, 18)

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "1"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "2"

    if index_up and middle_up and ring_up and not pinky_up:
        return "3"

    return ""


def detect_menu_option_from_hands(all_hand_landmarks):
    for hand_landmarks in all_hand_landmarks:
        option = detect_menu_option(hand_landmarks)

        if option:
            return option

    return ""


def clear_all_buffers():
    alphabet_history.clear()
    number_history.clear()
    phrase_history.clear()
    motion_buffer.clear()

# ─────────────────────────────
# Drawing Helpers
# ─────────────────────────────
def draw_hand(frame, hand_landmarks, width, height):
    points = []

    for lm in hand_landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)


def draw_menu(frame, selected_option, progress):
    cv2.rectangle(frame, (20, 20), (780, 310), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "FILIPINO SIGN LANGUAGE DETECTOR",
        (40, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Show hand sign 1, 2, or 3 to choose a mode",
        (40, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Option 1 - FSL Alphabet",
        (60, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Option 2 - FSL Numbers 0-9",
        (60, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Option 3 - FSL Words / Phrases",
        (60, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    if selected_option:
        cv2.putText(
            frame,
            f"Selecting Option {selected_option}... {progress:.0f}%",
            (40, 295),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
    else:
        cv2.putText(
            frame,
            "Press Q to quit",
            (40, 295),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


def draw_mode_header(frame, mode_name):
    cv2.rectangle(frame, (20, 20), (940, 90), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Mode: {mode_name}",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press M for menu | Press Q to quit",
        (560, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

# ─────────────────────────────
# Main App
# ─────────────────────────────
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not found. Try changing CAMERA_INDEX to 1 or 2.")
        return

    frame_count = 0
    current_mode = MODE_MENU

    menu_candidate = ""
    menu_candidate_start = 0

    last_motion_letter = ""
    last_motion_time = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = frame_count * 33
            frame_count += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_landmarks = None
            all_hand_landmarks = []
            detected_hands = 0

            if result.hand_landmarks:
                all_hand_landmarks = result.hand_landmarks
                detected_hands = len(all_hand_landmarks)
                hand_landmarks = all_hand_landmarks[0]

                for one_hand_landmarks in all_hand_landmarks:
                    draw_hand(frame, one_hand_landmarks, w, h)

                if current_mode == MODE_ALPHABET:
                    motion_buffer.append(raw_landmarks(hand_landmarks))

                elif current_mode == MODE_PHRASES:
                    motion_buffer.append(get_two_hand_frame(all_hand_landmarks))

            # ─────────────────────────────
            # MENU MODE
            # ─────────────────────────────
            if current_mode == MODE_MENU:
                selected_option = ""
                progress = 0

                if all_hand_landmarks:
                    detected_option = detect_menu_option_from_hands(all_hand_landmarks)

                    if detected_option:
                        if detected_option != menu_candidate:
                            menu_candidate = detected_option
                            menu_candidate_start = time.time()
                        else:
                            elapsed = time.time() - menu_candidate_start
                            progress = min((elapsed / MENU_HOLD_SECONDS) * 100, 100)

                            if elapsed >= MENU_HOLD_SECONDS:
                                if detected_option == "1":
                                    current_mode = MODE_ALPHABET
                                    print("Alphabet mode selected.")

                                elif detected_option == "2":
                                    current_mode = MODE_NUMBERS
                                    print("Numbers mode selected.")

                                elif detected_option == "3":
                                    current_mode = MODE_PHRASES
                                    print("Phrases mode selected.")

                                clear_all_buffers()
                                menu_candidate = ""
                                menu_candidate_start = 0
                    else:
                        menu_candidate = ""
                        menu_candidate_start = 0
                else:
                    menu_candidate = ""
                    menu_candidate_start = 0

                selected_option = menu_candidate
                draw_menu(frame, selected_option, progress)

            # ─────────────────────────────
            # ALPHABET MODE
            # ─────────────────────────────
            elif current_mode == MODE_ALPHABET:
                draw_mode_header(frame, "FSL Alphabet")

                static_letter = ""
                static_confidence = 0.0

                motion_letter = ""
                motion_confidence = 0.0
                movement_score = 0.0

                final_letter = ""

                now = time.time()
                motion_hold_active = (
                    last_motion_letter != ""
                    and now - last_motion_time <= MOTION_HOLD_SECONDS
                )

                if hand_landmarks is not None:
                    static_letter, static_confidence = predict_static(
                        alphabet_classifier,
                        hand_landmarks
                    )

                    if static_confidence >= STATIC_CONFIDENCE_THRESHOLD:
                        alphabet_history.append(static_letter)

                    stable_letter = get_stable_prediction(alphabet_history)
                    final_letter = stable_letter

                    if motion_hold_active:
                        final_letter = last_motion_letter
                    else:
                        if motion_classifier is not None and len(motion_buffer) == SEQUENCE_LENGTH:
                            sequence = list(motion_buffer)
                            movement_score = calculate_movement(sequence)

                            if movement_score >= MOTION_MOVEMENT_THRESHOLD:
                                motion_letter, motion_confidence = predict_motion(
                                    motion_classifier,
                                    sequence
                                )

                                motion_letter = str(motion_letter).upper().strip()

                                if motion_letter != "NONE" and motion_confidence >= MOTION_CONFIDENCE_THRESHOLD:
                                    last_motion_letter = motion_letter
                                    last_motion_time = time.time()

                                    final_letter = motion_letter

                                    alphabet_history.clear()
                                    motion_buffer.clear()
                else:
                    alphabet_history.clear()
                    motion_buffer.clear()
                    final_letter = ""

                cv2.rectangle(frame, (20, 110), (700, 260), (0, 0, 0), -1)

                cv2.putText(
                    frame,
                    f"Detected Letter: {final_letter}",
                    (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    f"Static Confidence: {static_confidence:.2f}",
                    (40, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Motion: {motion_letter} {motion_confidence:.2f} | Move: {movement_score:.2f}",
                    (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            # ─────────────────────────────
            # NUMBERS MODE
            # ─────────────────────────────
            elif current_mode == MODE_NUMBERS:
                draw_mode_header(frame, "FSL Numbers 0-9")

                detected_number = ""
                number_confidence = 0.0

                if number_classifier is None:
                    cv2.rectangle(frame, (20, 110), (900, 230), (0, 0, 0), -1)

                    cv2.putText(
                        frame,
                        "Number model not found.",
                        (40, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Create and train models/numbers/fsl_number_model.joblib first.",
                        (40, 205),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (0, 255, 255),
                        2
                    )

                else:
                    if hand_landmarks is not None:
                        number_label, number_confidence = predict_static(
                            number_classifier,
                            hand_landmarks
                        )

                        if number_confidence >= NUMBER_CONFIDENCE_THRESHOLD:
                            number_history.append(number_label)

                        detected_number = get_stable_prediction(number_history)
                    else:
                        number_history.clear()

                    cv2.rectangle(frame, (20, 110), (650, 220), (0, 0, 0), -1)

                    cv2.putText(
                        frame,
                        f"Detected Number: {detected_number}",
                        (40, 165),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        frame,
                        f"Confidence: {number_confidence:.2f}",
                        (40, 205),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2
                    )

            # ─────────────────────────────
            # PHRASES MODE
            # ─────────────────────────────
            elif current_mode == MODE_PHRASES:
                draw_mode_header(frame, "FSL Words / Phrases")

                detected_phrase = ""
                phrase_confidence = 0.0
                movement_score = 0.0
                raw_phrase_label = ""

                if phrase_classifier is None:
                    cv2.rectangle(frame, (20, 110), (900, 250), (0, 0, 0), -1)

                    cv2.putText(
                        frame,
                        "Phrase model not found.",
                        (40, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Create and train models/phrases/fsl_phrase_model.joblib first.",
                        (40, 205),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )

                elif not phrase_model_compatible:
                    cv2.rectangle(frame, (20, 110), (1060, 285), (0, 0, 0), -1)

                    cv2.putText(
                        frame,
                        "Phrase model format is old/incompatible.",
                        (40, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.90,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Recollect phrase data using the 2-hand collect_phrase_data.py.",
                        (40, 205),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "Then run train_phrase_model.py again.",
                        (40, 245),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )

                else:
                    if hand_landmarks is not None and len(motion_buffer) == SEQUENCE_LENGTH:
                        sequence = list(motion_buffer)
                        movement_score = calculate_phrase_movement(sequence)

                        if movement_score >= PHRASE_MOVEMENT_THRESHOLD:
                            raw_phrase_label, phrase_confidence = predict_phrase_motion(
                                phrase_classifier,
                                sequence
                            )

                            raw_phrase_label = str(raw_phrase_label).upper().strip()

                            if phrase_confidence >= PHRASE_CONFIDENCE_THRESHOLD:
                                if raw_phrase_label == "NONE":
                                    phrase_history.clear()
                                else:
                                    phrase_history.append(raw_phrase_label)

                        motion_buffer.clear()

                    elif hand_landmarks is None:
                        phrase_history.clear()
                        motion_buffer.clear()

                    detected_phrase = get_stable_prediction(phrase_history)
                    display_phrase = detected_phrase.replace("_", " ")

                    cv2.rectangle(frame, (20, 110), (930, 295), (0, 0, 0), -1)

                    cv2.putText(
                        frame,
                        f"Detected Phrase: {display_phrase}",
                        (40, 165),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        frame,
                        f"Raw: {raw_phrase_label} | Confidence: {phrase_confidence:.2f}",
                        (40, 210),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Movement Score: {movement_score:.2f} | Hands: {detected_hands}",
                        (40, 245),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "For 2-hand phrases, make sure both hands are visible.",
                        (40, 280),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2
                    )

            cv2.imshow("FSL Detector Menu System", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("m"):
                current_mode = MODE_MENU
                clear_all_buffers()
                menu_candidate = ""
                menu_candidate_start = 0
                print("Returned to main menu.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()