import os
import csv
import urllib.request
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

PHRASE_DATA_FILE = os.path.join("data", "phrases", "fsl_phrase_data.csv")

CAMERA_INDEX = 0
SEQUENCE_LENGTH = 30
MAX_HANDS = 2

PHRASE_LABELS = [
    "SALAMAT",
    "KAMUSTA",
    "MAGANDANG_UMAGA",
    "MAHAL_KITA",
    "SORRY_/PASENSYA",
    "OO",
    "HINDI",
    "ANG_PANGALAN_KO_AY_SI",
    "KAMUSTA_KA",
    "AYOS_LANG_/ITS_OKAY",
    "NAIINTINDIHAN_KO",
    "NONE"
]

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def clean_label(label):
    label = str(label).strip().upper()
    label = label.replace("?", "")
    label = label.replace("/", "_")
    label = label.replace("-", "_")
    label = "_".join(label.split())
    return label


def create_csv_if_needed():
    os.makedirs(os.path.dirname(PHRASE_DATA_FILE), exist_ok=True)

    if not os.path.exists(PHRASE_DATA_FILE):
        header = ["label"]

        for i in range(SEQUENCE_LENGTH * MAX_HANDS * 21 * 3):
            header.append(f"f{i}")

        with open(PHRASE_DATA_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)


def load_existing_counts():
    counts = Counter()

    if not os.path.exists(PHRASE_DATA_FILE):
        return counts

    try:
        with open(PHRASE_DATA_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                label = clean_label(row.get("label", ""))
                if label:
                    counts[label] += 1
    except Exception:
        pass

    return counts


def landmarks_to_points(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]


def get_hand_center_x(hand_points):
    return sum(point[0] for point in hand_points) / len(hand_points)


def get_two_hand_frame(all_hand_landmarks):
    hands = []

    for hand_landmarks in all_hand_landmarks:
        hands.append(landmarks_to_points(hand_landmarks))

    hands.sort(key=get_hand_center_x)

    while len(hands) < MAX_HANDS:
        hands.append(None)

    return hands[:MAX_HANDS]


def sequence_to_features(sequence):
    first_frame = sequence[0]

    visible_points = []

    for hand in first_frame:
        if hand is not None:
            visible_points.extend(hand)

    if not visible_points:
        return [0.0] * (SEQUENCE_LENGTH * MAX_HANDS * 21 * 3)

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


def draw_hand(frame, hand_landmarks, width, height):
    points = []

    for lm in hand_landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)


def draw_ui(frame, key_to_label, saved_counts, recording_label, sequence_length, detected_hands):
    cv2.rectangle(frame, (20, 20), (1050, 285), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "FSL Words / Phrases Data Collection - 2 Hands Supported",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press assigned key to record 30 frames | ESC to quit",
        (35, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Detected hands: {detected_hands}",
        (35, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    y = 155
    x = 35

    for key, label in key_to_label.items():
        text = f"{key.upper()} = {label} ({saved_counts[label]})"

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            2
        )

        y += 25

        if y > 260:
            y = 155
            x += 340

    if recording_label is not None:
        cv2.rectangle(frame, (20, 305), (820, 375), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"RECORDING: {recording_label}  {sequence_length}/{SEQUENCE_LENGTH}",
            (35, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3
        )


def main():
    labels = [clean_label(label) for label in PHRASE_LABELS]
    labels = [label for label in labels if label]

    if len(labels) < 2:
        print("Add at least 2 phrase labels in PHRASE_LABELS.")
        return

    if "NONE" not in labels:
        labels.append("NONE")

    if len(set(labels)) != len(labels):
        print("Duplicate labels found in PHRASE_LABELS. Please remove duplicates.")
        return

    keys = list("123456789abcdefghijklmnopqrstuvwxyz")

    if len(labels) > len(keys):
        print("Too many labels. Limit your first batch to 35 labels or fewer.")
        return

    key_to_label = {}

    for i, label in enumerate(labels):
        key_to_label[keys[i]] = label

    create_csv_if_needed()
    saved_counts = load_existing_counts()

    print("")
    print("Phrase key map:")

    for key, label in key_to_label.items():
        print(f"{key.upper()} = {label}")

    print("")
    print("Important:")
    print("- This collector supports 1-hand and 2-hand phrases.")
    print("- For 2-hand phrases, make sure both hands are visible before pressing the key.")
    print("- For 1-hand phrases, show only the required signing hand.")
    print("- For NONE, record neutral/random hand movement that is NOT one of your phrases.")
    print("- Recommended: 50 to 100 samples per phrase, including NONE.")
    print("")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not found. Try changing CAMERA_INDEX to 1 or 2.")
        return

    frame_count = 0
    recording_label = None
    sequence = []

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

            detected_hands = 0
            current_two_hand_frame = None

            if result.hand_landmarks:
                detected_hands = len(result.hand_landmarks)
                current_two_hand_frame = get_two_hand_frame(result.hand_landmarks)

                for hand_landmarks in result.hand_landmarks:
                    draw_hand(frame, hand_landmarks, w, h)

            if recording_label is not None:
                if current_two_hand_frame is not None:
                    sequence.append(current_two_hand_frame)

                if len(sequence) >= SEQUENCE_LENGTH:
                    features = sequence_to_features(sequence)

                    with open(PHRASE_DATA_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([recording_label] + features)

                    saved_counts[recording_label] += 1

                    print(f"Saved phrase sample: {recording_label} | Total: {saved_counts[recording_label]}")

                    recording_label = None
                    sequence = []

            draw_ui(frame, key_to_label, saved_counts, recording_label, len(sequence), detected_hands)

            cv2.imshow("Collect FSL Phrase Data", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if recording_label is None:
                pressed_key = chr(key).lower() if key != 255 else ""

                if pressed_key in key_to_label:
                    if current_two_hand_frame is None:
                        print("Show your hand first before recording.")
                    else:
                        recording_label = key_to_label[pressed_key]
                        sequence = []
                        print(f"Recording phrase: {recording_label}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()