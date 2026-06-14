import os
import csv
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DATA_FILE = os.path.join("data", "alphabet", "fsl_alphabet_data.csv")

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
    if not os.path.exists(DATA_FILE):
        header = ["label"]

        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])

        with open(DATA_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)

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

                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.putText(
                frame,
                "Press A-Z to save sample | Press ESC to quit",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Saved samples: {saved_count}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Collect FSL Alphabet Data", frame)

            key = cv2.waitKey(1)

            if key == -1:
                continue

            key = key & 0xFF

            if key == 27:  # ESC key
                break

            if current_features is not None:
                if ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
                    char = chr(key).upper()

                    with open(DATA_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([char] + current_features)

                    saved_count += 1
                    print(f"Saved sample for letter: {char}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()