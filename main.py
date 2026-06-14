"""
app.py
------
Flask web app for the Filipino Sign Language Detector.

Adapted from main.py (desktop OpenCV version) — same MediaPipe
HandLandmarker setup, normalization, and model prediction logic,
but streams the camera feed + detection results to a web UI via
Server-Sent Events (SSE) instead of cv2.imshow().

Mode switching (ALPHABET / NUMBERS / PHRASES) is controlled by the
web UI buttons via POST /set_fsl_mode.
"""

import base64
import json
import os
import threading
import time
import urllib.request
import warnings
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)

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

MODE_ALPHABET = "ALPHABET"
MODE_NUMBERS = "NUMBERS"
MODE_PHRASES = "PHRASES"

VALID_MODES = {MODE_ALPHABET, MODE_NUMBERS, MODE_PHRASES}

PHRASE_EXPECTED_FEATURES = SEQUENCE_LENGTH * PHRASE_MAX_HANDS * 21 * 3

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

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
alphabet_classifier = None
motion_classifier = None
number_classifier = None
phrase_classifier = None
phrase_model_compatible = False

if os.path.exists(ALPHABET_MODEL_PATH):
    alphabet_classifier = joblib.load(ALPHABET_MODEL_PATH)
    print("Alphabet model loaded.")
else:
    print("WARNING: No trained alphabet model found at", ALPHABET_MODEL_PATH)

if os.path.exists(MOTION_MODEL_PATH):
    motion_classifier = joblib.load(MOTION_MODEL_PATH)
    print("Alphabet motion model loaded. J and Z detection enabled.")
else:
    print("No alphabet motion model found. J and Z movement detection disabled.")

if os.path.exists(NUMBER_MODEL_PATH):
    number_classifier = joblib.load(NUMBER_MODEL_PATH)
    print("Number model loaded. Number detection enabled.")
else:
    print("No number model found. Number detection disabled.")

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
        print("WARNING: Phrase model is not compatible with 2-hand phrase detection.")
        print(f"Expected features: {PHRASE_EXPECTED_FEATURES}, Model features: {phrase_feature_count}")
else:
    print("No phrase model found. Phrase detection disabled.")

# ─────────────────────────────
# MediaPipe Setup
# ─────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# ─────────────────────────────
# Shared State
# ─────────────────────────────
state_lock = threading.Lock()
state = {
    "fsl_mode": MODE_ALPHABET,
    "fsl_label": "",
    "frame_jpeg_b64": None,
}

# ─────────────────────────────
# Prediction Buffers
# ─────────────────────────────
alphabet_history = deque(maxlen=10)
number_history = deque(maxlen=10)
phrase_history = deque(maxlen=5)
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)

last_motion_letter = ""
last_motion_time = 0


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


# ─────────────────────────────
# Camera + Detection Loop
# ─────────────────────────────
def camera_loop():
    global last_motion_letter, last_motion_time

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not found. Try changing CAMERA_INDEX to 1 or 2.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = frame_count * 33
        frame_count += 1

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_landmarks = None
        all_hand_landmarks = []

        if result.hand_landmarks:
            all_hand_landmarks = result.hand_landmarks
            hand_landmarks = all_hand_landmarks[0]

            for one_hand_landmarks in all_hand_landmarks:
                draw_hand(frame, one_hand_landmarks, w, h)

        with state_lock:
            current_mode = state["fsl_mode"]

        final_label = None

        # ── ALPHABET MODE ──
        if current_mode == MODE_ALPHABET:
            if hand_landmarks is not None:
                motion_buffer.append(raw_landmarks(hand_landmarks))

            if hand_landmarks is not None and alphabet_classifier is not None:
                static_letter, static_confidence = predict_static(alphabet_classifier, hand_landmarks)

                if static_confidence >= STATIC_CONFIDENCE_THRESHOLD:
                    alphabet_history.append(static_letter)

                final_label = get_stable_prediction(alphabet_history)

                now = time.time()
                motion_hold_active = (
                    last_motion_letter != ""
                    and now - last_motion_time <= MOTION_HOLD_SECONDS
                )

                if motion_hold_active:
                    final_label = last_motion_letter
                elif motion_classifier is not None and len(motion_buffer) == SEQUENCE_LENGTH:
                    sequence = list(motion_buffer)
                    movement_score = calculate_movement(sequence)

                    if movement_score >= MOTION_MOVEMENT_THRESHOLD:
                        motion_letter, motion_confidence = predict_motion(motion_classifier, sequence)
                        motion_letter = str(motion_letter).upper().strip()

                        if motion_letter != "NONE" and motion_confidence >= MOTION_CONFIDENCE_THRESHOLD:
                            last_motion_letter = motion_letter
                            last_motion_time = time.time()
                            final_label = motion_letter

                            alphabet_history.clear()
                            motion_buffer.clear()
            else:
                alphabet_history.clear()
                motion_buffer.clear()

        # ── NUMBERS MODE ──
        elif current_mode == MODE_NUMBERS:
            if hand_landmarks is not None and number_classifier is not None:
                number_label, number_confidence = predict_static(number_classifier, hand_landmarks)

                if number_confidence >= NUMBER_CONFIDENCE_THRESHOLD:
                    number_history.append(number_label)

                final_label = get_stable_prediction(number_history)
            else:
                number_history.clear()

        # ── PHRASES MODE ──
        elif current_mode == MODE_PHRASES:
            if hand_landmarks is not None:
                motion_buffer.append(get_two_hand_frame(all_hand_landmarks))

            if phrase_classifier is not None and phrase_model_compatible:
                if hand_landmarks is not None and len(motion_buffer) == SEQUENCE_LENGTH:
                    sequence = list(motion_buffer)
                    movement_score = calculate_phrase_movement(sequence)

                    if movement_score >= PHRASE_MOVEMENT_THRESHOLD:
                        raw_label, phrase_confidence = predict_phrase_motion(phrase_classifier, sequence)
                        raw_label = str(raw_label).upper().strip()

                        if phrase_confidence >= PHRASE_CONFIDENCE_THRESHOLD:
                            if raw_label == "NONE":
                                phrase_history.clear()
                            else:
                                phrase_history.append(raw_label)

                    motion_buffer.clear()

                elif hand_landmarks is None:
                    phrase_history.clear()
                    motion_buffer.clear()

                detected_phrase = get_stable_prediction(phrase_history)
                final_label = detected_phrase.replace("_", " ") if detected_phrase else ""

        # ── Encode frame for streaming ──
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            continue
        frame_b64 = base64.b64encode(buf).decode("utf-8")

        with state_lock:
            state["frame_jpeg_b64"] = frame_b64
            if final_label is not None and final_label != "" and final_label != state["fsl_label"]:
                state["fsl_label"] = final_label

        time.sleep(0.01)


# ─────────────────────────────
# Routes
# ─────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_state")
def get_state():
    with state_lock:
        return jsonify({
            "fsl_mode_current": state["fsl_mode"],
            "fsl_label": state["fsl_label"],
        })


@app.route("/set_fsl_mode", methods=["POST"])
def set_fsl_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "")

    if mode not in VALID_MODES:
        return jsonify({"ok": False, "error": "invalid mode"}), 400

    with state_lock:
        state["fsl_mode"] = mode
        state["fsl_label"] = ""

    clear_all_buffers()

    return jsonify({"ok": True, "mode": mode})


@app.route("/stream")
def stream():
    def event_stream():
        with state_lock:
            init_payload = {
                "fsl_label": state["fsl_label"],
                "fsl_mode": state["fsl_mode"],
            }
        yield f"event: init\ndata: {json.dumps(init_payload)}\n\n"

        last_label = None
        while True:
            with state_lock:
                frame_b64 = state["frame_jpeg_b64"]
                label = state["fsl_label"]
                mode = state["fsl_mode"]

            if frame_b64:
                yield f"event: frame\ndata: {frame_b64}\n\n"

            if label and label != last_label:
                payload = {"label": label, "mode": mode}
                yield f"event: fsl\ndata: {json.dumps(payload)}\n\n"
                last_label = label

            time.sleep(0.05)

    return Response(event_stream(), mimetype="text/event-stream")


# ─────────────────────────────
if __name__ == "__main__":
    worker_thread = threading.Thread(target=camera_loop, daemon=True)
    worker_thread.start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)