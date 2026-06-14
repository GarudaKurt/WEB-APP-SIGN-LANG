"""
fsl_worker.py
-------------
MediaPipe Hands + your trained models for FSL Alphabet, Numbers, and Phrases.

Drop your model files (e.g. fsl_model.joblib, fsl_numbers_model.joblib,
fsl_phrase_model.joblib) next to this file and load them in load_models().

Each predict_* function takes a BGR frame (numpy array from OpenCV) and
returns (label_or_None, annotated_frame).
"""

import cv2
import joblib
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Globals — populated by load_models()
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

alphabet_model = None
numbers_model = None
phrase_model = None

# Tune these to match your data collection / training settings
CONFIDENCE_THRESHOLD = 0.7
DROPOUT_TOLERANCE_FRAMES = 8  # frames a hand can be lost before resetting

_dropout_counter = {"alphabet": 0, "numbers": 0, "phrase": 0}
_last_label = {"alphabet": None, "numbers": None, "phrase": None}


def load_models():
    """Load trained models. Adjust paths/filenames as needed."""
    global alphabet_model, numbers_model, phrase_model

    try:
        alphabet_model = joblib.load("fsl_alphabet_model.joblib")
    except FileNotFoundError:
        print("[fsl_worker] WARNING: alphabet model not found")

    try:
        numbers_model = joblib.load("fsl_numbers_model.joblib")
    except FileNotFoundError:
        print("[fsl_worker] WARNING: numbers model not found")

    try:
        phrase_model = joblib.load("fsl_phrase_model.joblib")
    except FileNotFoundError:
        print("[fsl_worker] WARNING: phrase model not found")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_landmarks(results, frame_shape):
    """
    Flatten hand landmarks into a feature vector.
    Adjust this to match the exact feature layout your model was trained on
    (e.g. single-hand 63-dim, or two-hand 126-dim with zero-padding).
    """
    feature_vec = []
    num_hands_expected = 2  # set to 1 if your model only uses one hand

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:num_hands_expected]:
            for lm in hand_landmarks.landmark:
                feature_vec.extend([lm.x, lm.y, lm.z])

    # Zero-pad if fewer hands than expected
    expected_len = num_hands_expected * 21 * 3
    while len(feature_vec) < expected_len:
        feature_vec.append(0.0)

    return np.array(feature_vec[:expected_len]).reshape(1, -1)


def _draw_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
    return frame


def _run_hands(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    return results


def _predict_with_model(model, results, frame, key):
    """Shared logic: detect, classify, apply confidence + dropout tolerance."""
    frame = _draw_landmarks(frame, results)

    if not results.multi_hand_landmarks:
        _dropout_counter[key] += 1
        if _dropout_counter[key] > DROPOUT_TOLERANCE_FRAMES:
            _last_label[key] = None
        return _last_label[key], frame

    _dropout_counter[key] = 0

    if model is None:
        return None, frame

    features = _extract_landmarks(results, frame.shape)

    try:
        proba = model.predict_proba(features)[0]
        best_idx = int(np.argmax(proba))
        confidence = proba[best_idx]
        label = model.classes_[best_idx]
    except AttributeError:
        # Model without predict_proba — fall back to plain predict
        label = model.predict(features)[0]
        confidence = 1.0

    if confidence < CONFIDENCE_THRESHOLD:
        return _last_label[key], frame

    _last_label[key] = str(label)
    return _last_label[key], frame


# ---------------------------------------------------------------------------
# Public prediction functions (called from app.py)
# ---------------------------------------------------------------------------
def predict_alphabet(frame):
    results = _run_hands(frame)
    return _predict_with_model(alphabet_model, results, frame, "alphabet")


def predict_numbers(frame):
    results = _run_hands(frame)
    return _predict_with_model(numbers_model, results, frame, "numbers")


def predict_phrase(frame):
    results = _run_hands(frame)
    return _predict_with_model(phrase_model, results, frame, "phrase")


def draw_idle_overlay(frame):
    """MENU mode: show hand landmarks without classification."""
    results = _run_hands(frame)
    return _draw_landmarks(frame, results)