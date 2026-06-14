import os
import csv
import urllib.request
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

MOTION_DATA_FILE = os.path.join("data", "alphabet", "fsl_alphabet_motion_data.csv")

CAMERA_INDEX = 0
SEQUENCE_LENGTH = 30  # about 1 second

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

def create_csv_if_needed():
    if not os.path.exists(MOTION_DATA_FILE):
        header = ["label"]

        for i in range(SEQUENCE_LENGTH * 21 * 3):
            header.append(f"f{i}")

        with open(MOTION_DATA_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

def raw_landmarks(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]

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

def main():
    create_csv_if_needed()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not found. Try changing CAMERA_INDEX to 1 or 2.")
        return

    frame_count = 0
    recording_label = None
    sequence = []
    saved_counts = Counter()

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

            current_raw = None

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                current_raw = raw_landmarks(hand_landmarks)

                points = []

                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (255, 0, 0), 2)

            if recording_label is not None:
                if current_raw is not None:
                    sequence.append(current_raw)

                cv2.putText(
                    frame,
                    f"RECORDING {recording_label}: {len(sequence)}/{SEQUENCE_LENGTH}",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                if len(sequence) >= SEQUENCE_LENGTH:
                    features = sequence_to_features(sequence)

                    with open(MOTION_DATA_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([recording_label] + features)

                    saved_counts[recording_label] += 1
                    print(f"Saved movement sample for: {recording_label}")

                    recording_label = None
                    sequence = []

            cv2.rectangle(frame, (20, 20), (780, 100), (0, 0, 0), -1)

            cv2.putText(
                frame,
                "Press J or Z to record movement | Press 0 for NONE | ESC to quit",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Saved: J={saved_counts['J']}  Z={saved_counts['Z']}  NONE={saved_counts['NONE']}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.imshow("Collect J and Z Movement Data", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if recording_label is None:
                if key == ord("j") or key == ord("J"):
                    if current_raw is not None:
                        recording_label = "J"
                        sequence = []
                        print("Recording J movement...")
                    else:
                        print("Show your hand first.")

                elif key == ord("z") or key == ord("Z"):
                    if current_raw is not None:
                        recording_label = "Z"
                        sequence = []
                        print("Recording Z movement...")
                    else:
                        print("Show your hand first.")

                elif key == ord("0"):
                    if current_raw is not None:
                        recording_label = "NONE"
                        sequence = []
                        print("Recording NONE movement...")
                    else:
                        print("Show your hand first.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()