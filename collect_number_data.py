import os
import csv
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

DATA_FILE = os.path.join("data", "numbers", "fsl_number_data.csv")

CAMERA_INDEX = 0

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
    num_hands=1,
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


def create_csv_if_needed():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

    if not os.path.exists(DATA_FILE):
        header = ["label"]

        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])

        with open(DATA_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)


def draw_hand(frame, hand_landmarks, width, height):
    points = []

    for lm in hand_landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 0, 0), 2)


def main():
    create_csv_if_needed()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not found. Try changing CAMERA_INDEX to 1 or 2.")
        return

    frame_count = 0
    saved_count = 0

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

            current_features = None

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                current_features = normalize_landmarks(hand_landmarks)
                draw_hand(frame, hand_landmarks, w, h)

            cv2.rectangle(frame, (20, 20), (820, 120), (0, 0, 0), -1)

            cv2.putText(
                frame,
                "FSL Numbers Data Collection",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Show number sign, then press keyboard 0-9 to save | ESC to quit",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Saved samples this session: {saved_count}",
                (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.imshow("Collect FSL Number Data", frame)

            key = cv2.waitKey(1)

            if key == -1:
                continue

            key = key & 0xFF

            if key == 27:
                break

            if current_features is not None:
                if ord("0") <= key <= ord("9"):
                    number_label = chr(key)

                    with open(DATA_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([number_label] + current_features)

                    saved_count += 1
                    print(f"Saved sample for number: {number_label}")
            else:
                if ord("0") <= key <= ord("9"):
                    print("Show your hand first before saving.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()